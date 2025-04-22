# mppi_planner.py 
import torch
import numpy as np

# Assuming environment defines the Environment class
# from environment import Environment # No direct import needed if env passed to init

class MPPI:
    
    def __init__(self, environment, dynamics_model, cost_function,
                 horizon, num_samples, temperature, control_dim, state_dim, dt, sigma,
                 u_min, u_max, device='cuda'):
        """
        Args:
            environment (Environment): 
            dynamics_model (callable): Function mapping (state, control, dt) -> next_state.
            cost_function (callable): Function mapping (states [K,T+1,D], controls [K,T,C]) -> costs [K].
            horizon (int): T
            num_samples (int): K
            temperature (float): lambda_
            control_dim (int): Dimension of control vector.
            state_dim (int): Dimension of state vector.
            dt (float): Time step.
            sigma (np.array): Control noise std dev [std_v, std_omega].
            u_min (np.array): Min control limits [min_v, min_omega].
            u_max (np.array): Max control limits [max_v, max_omega].
            device (str): 'cpu' or 'cuda'.
        """
        self.env = environment
        self.dynamics = dynamics_model
        self.cost_func = cost_function
        self.K = num_samples
        self.T = horizon
        self.dt = dt
        self.lambda_ = temperature # Map temperature to lambda_
        self.control_dim = control_dim
        self.state_dim = state_dim
        self.device = torch.device(device)
        # Ensure limits are tensors on the correct device
        self.u_min = torch.tensor(u_min, dtype=torch.float32, device=self.device)
        self.u_max = torch.tensor(u_max, dtype=torch.float32, device=self.device)


        # Create covariance matrix Sigma from std dev sigma
        if isinstance(sigma, (list, tuple, np.ndarray)):
             if len(sigma) != self.control_dim:
                 raise ValueError(f"Length of sigma {len(sigma)} != control_dim {self.control_dim}")
             self.Sigma = torch.diag(torch.tensor(sigma, dtype=torch.float32)**2).to(self.device)
        else:
             raise ValueError("sigma must be array-like [std_v, std_omega]")

        self.mean_noise = torch.zeros(self.control_dim, device=self.device)

        self.U_nominal = torch.zeros((self.T, self.control_dim), dtype=torch.float32, device=self.device)
       
        # self.U_nominal[:, 0] = 0.01

        self.robot_radius = 0.15 # Default, GET FROM ROBOT passed to main script later
        self.last_rollouts_np = None # To store rollouts for visualization

        print(f"MPPI Initialized (Our Planner). K={self.K}, T={self.T}, dt={self.dt}, Device={self.device}")
        print(f"Control limits: {u_min} to {u_max}")
        print(f"Noise Sigma (std dev): {sigma}")

    def _compute_rollout_costs(self, current_state_tensor, goal_tensor):
        """ Computes costs for K sampled trajectories. """
        # --- 1. Sample Noise ---
        noise_dist = torch.distributions.MultivariateNormal(self.mean_noise, self.Sigma)
        delta_U = noise_dist.sample((self.K, self.T))

        # --- 2. Simulate Rollouts ---
        state_sequences = torch.zeros((self.K, self.T + 1, self.state_dim), dtype=torch.float32, device=self.device)
        state_sequences[:, 0, :] = current_state_tensor.repeat(self.K, 1)
        V = self.U_nominal.unsqueeze(0) + delta_U # Add noise to nominal sequence

        # Clamp controls using PyTorch tensors
        for d in range(self.control_dim):
             V[:, :, d] = torch.clamp(V[:, :, d], self.u_min[d], self.u_max[d])

        # Simulate T steps using the provided dynamics model
        for t in range(self.T):
            state_sequences[:, t + 1, :] = self.dynamics(
                state_sequences[:, t, :], V[:, t, :], self.dt
            )

        # Store rollouts for visualization
        self.last_rollouts_np = state_sequences.detach().cpu().numpy()

        # --- 3. Calculate Costs using the provided cost function ---
        # The cost function now takes goal tensor, env, robot_radius etc implicitly or explicitly
        # Assuming cost_func is the wrapper defined in main script
        total_costs = self.cost_func(state_sequences, V)

        return total_costs, delta_U, V # Return perturbed controls V as well


    def optimize(self, current_state_np, goal_np, obstacles_list_ignored=None):
        """
        Optimizes the control sequence U using MPPI update rule.
        Matches the call signature in the main function.
        Args:
            current_state_np (np.array): Current robot state [x, y, theta].
            goal_np (np.array): Goal state [x, y, theta?].
            obstacles_list_ignored: main passes obstacles, but our cost func uses self.env.

        Returns:
            np.array: Optimal first control command [v, omega].
        """
        # Ensure tensors
        current_state_tensor = torch.tensor(current_state_np, dtype=torch.float32, device=self.device)
        goal_tensor = torch.tensor(goal_np, dtype=torch.float32, device=self.device) # Cost func needs goal

        # --- Core MPPI Logic ---
        costs, delta_U, V = self._compute_rollout_costs(current_state_tensor, goal_tensor)

        # Weighting & Combining
        min_cost = torch.min(costs)
        exp_costs = torch.exp(-1.0 / self.lambda_ * (costs - min_cost))
        eta = torch.sum(exp_costs)
        if eta < 1e-9: weights = (1.0 / self.K) * torch.ones_like(costs)
        else: weights = exp_costs / eta

        # Update nominal sequence U_nominal using weighted average of V
        self.U_nominal = torch.sum(weights.unsqueeze(1).unsqueeze(2) * V, dim=0)

        # Get the first optimal control command
        optimal_first_control = self.U_nominal[0].clone()

        # Warm Start: Shift nominal sequence
        self.U_nominal = torch.roll(self.U_nominal, shifts=-1, dims=0)
        self.U_nominal[-1] = torch.zeros(self.control_dim, device=self.device) # Reset last

        return optimal_first_control.cpu().numpy() # Return only control

    def get_last_rollouts(self):
        """ Returns the rollouts generated during the last optimize call."""
        # Returns numpy array [K, T+1, state_dim] or None
        return self.last_rollouts_np

    
    def generate_trajectories(self, current_state_np, control_sequence_tensor):
        """
        Generates trajectories based on a given control sequence.
        Needed for the visualization.
        Typically uses U_nominal.
        """
        # Ensure current_state is tensor
        current_state = torch.tensor(current_state_np, dtype=torch.float32, device=self.device)

        # Check control sequence shape
        if len(control_sequence_tensor.shape) == 2 and control_sequence_tensor.shape[0] == self.T:
            # If single sequence (T, C) is passed, unsqueeze it to (1, T, C)
            control_sequence_tensor = control_sequence_tensor.unsqueeze(0)
        elif len(control_sequence_tensor.shape) != 3 or control_sequence_tensor.shape[1] != self.T or control_sequence_tensor.shape[2] != self.control_dim:
             raise ValueError(f"Control sequence shape mismatch: Expected (K, {self.T}, {self.control_dim}), got {control_sequence_tensor.shape}")

        num_seq = control_sequence_tensor.shape[0]

        state_sequences = torch.zeros((num_seq, self.T + 1, self.state_dim), dtype=torch.float32, device=self.device)
        state_sequences[:, 0, :] = current_state.repeat(num_seq, 1)

        # Clamp controls (important if U_nominal is passed directly)
        V = control_sequence_tensor.clone().to(self.device) # Use passed controls
        for d in range(self.control_dim):
             V[:, :, d] = torch.clamp(V[:, :, d], self.u_min[d], self.u_max[d])

        dynamics_func = self.dynamics if self.dynamics else self._dynamics_tensor
        for t in range(self.T):
            state_sequences[:, t + 1, :] = dynamics_func(
                state_sequences[:, t, :], V[:, t, :], self.dt
            )

        # Return states (numpy), controls used (numpy), and None (matching example call)
        return state_sequences.detach().cpu().numpy(), V.detach().cpu().numpy(), None