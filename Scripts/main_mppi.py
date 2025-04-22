# main_mppi.py 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import torch
import os

from matplotlib.animation import FuncAnimation
try:
    import PIL # Used by visualizer for saving animation
except ImportError:
    print("Warning: Pillow library not found. GIF saving might fail.")
    print("Install using: pip install pillow")


try:
    from environment import Environment
    from robot import TurtleBot
    from visualization import Visualizer
except ImportError as e:
    print(f"Error importing classes: {e}")
    print("Make sure environment.py, robot.py, and visualization.py are in the same directory.")
    exit()


try:
    from mppi_planner import MPPI
except ImportError as e:
    print(f"Error importing MPPI planner: {e}")
    print("Make sure mppi_planner.py exists and is adapted correctly.")
    exit()

# --- Configuration ---
# Simulation settings
DT = 0.1                  # Time step
SIM_STEPS = 400           # Max simulation iterations
GOAL_THRESHOLD = 0.3      # Goal proximity threshold
PLOT_INTERVAL = 5         # Update real-time plot every N iterations

# --- Output Configuration ---
SAVE_STATIC_PLOT = True   # Save a PNG of the final path?
STATIC_PLOT_FILENAME = "mppi_final_path.png"
SAVE_ANIMATION = True     # Save a GIF animation?
ANIMATION_FILENAME = "mppi_animation.gif"
ANIMATION_FPS = 15        # GIF frames per second
ANIMATION_DPI = 100       # GIF resolution

# --- Environment Selection ---
# Choose which environment function to call:
ENV_TYPE = 'SIMPLE'
#ENV_TYPE = 'MODERATE'
#ENV_TYPE = 'COMPLEX'


# --- Cost Function Definition (Includes Safety Margin) ---
def mppi_cost_function_compatible(states, controls, goal, env, robot_radius,
                                   goal_weight=25.0, collision_weight=1000.0,
                                   ctrl_v_weight=0.1, ctrl_o_weight=0.1,
                                   goal_orient_weight=1.0,
                                   safety_margin=0.1): # <-- Safety margin added (tune this!)
    """ Calculates cost based on goal distance, collision (with safety margin), and control effort. """
    K, T_plus_1, state_dim = states.shape; T = controls.shape[1]; device = states.device
    total_costs = torch.zeros(K, dtype=torch.float32, device=device)
    effective_radius = robot_radius + safety_margin # Use safety margin

    # Goal Cost
    final_states_pos = states[:, -1, :2]; goal_pos = goal[:2]
    dist_to_goal_sq = torch.sum((final_states_pos - goal_pos)**2, dim=1)
    total_costs += goal_weight * dist_to_goal_sq
    if len(goal) >= 3: # Orientation Cost
       final_theta = states[:, -1, 2]; goal_theta = goal[2]
       angle_error = torch.atan2(torch.sin(final_theta - goal_theta), torch.cos(final_theta - goal_theta))
       total_costs += goal_orient_weight * angle_error**2
    # Control Cost
    total_costs += ctrl_v_weight * torch.sum(controls[:, :, 0]**2, dim=1)
    total_costs += ctrl_o_weight * torch.sum(controls[:, :, 1]**2, dim=1)
    # Collision Cost (Uses effective_radius)
    collision_penalty = torch.zeros(K, dtype=torch.float32, device=device)
    states_np = states.cpu().numpy()
    for k in range(K):
        collision_found = False
        for t in range(T_plus_1):
            x, y = states_np[k, t, 0], states_np[k, t, 1]
            if env.check_collision(x, y, effective_radius): # Check with margin
                collision_found = True; break
        if collision_found: collision_penalty[k] = collision_weight
    total_costs += collision_penalty
    return total_costs

# --- Main Simulation Function ---
def main():
    # Create environment based on selection
    if ENV_TYPE == 'SIMPLE':
        env = Environment.create_simple_environment()
        start_pose_np = np.array([1.0, 1.0, 0.0])
        goal_np = np.array([8.0, 8.0, np.pi/4])
        env_name = "Simple Environment"
    elif ENV_TYPE == 'MODERATE':
        env = Environment.create_moderate_environment()
        start_pose_np = np.array([1.0, 1.0, np.pi/4])
        goal_np = np.array([9.0, 9.0, 0.0])
        env_name = "Moderate Environment"
    elif ENV_TYPE == 'COMPLEX':
        env = Environment.create_complex_cluttered_environment()
        start_pose_np = np.array([0.5, 0.5, np.pi/4])
        goal_np = np.array([9.5, 9.5, -np.pi/4])
        env_name = "Complex Cluttered Environment"
    else:
        raise ValueError(f"Unknown ENV_TYPE: {ENV_TYPE}")

    print(f"Using Environment: {env_name}")

    # Create TurtleBot
    robot = TurtleBot(x=start_pose_np[0], y=start_pose_np[1], theta=start_pose_np[2])

    # --- Setup MPPI controller ---
    def dynamics_model_wrapper(state_tensor, control_tensor, dt):
        next_states = []
        for i in range(state_tensor.shape[0]):
             state_np = state_tensor[i].cpu().numpy(); control_np = control_tensor[i].cpu().numpy()
             next_state_np = robot.dynamics(state_np, control_np, dt)
             next_states.append(torch.tensor(next_state_np, dtype=torch.float32, device=state_tensor.device))
        return torch.stack(next_states)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    goal_tensor = torch.tensor(goal_np, dtype=torch.float32, device=device)
    robot_radius = robot.radius
    def cost_function_wrapper(states, controls):
         # Pass actual robot radius; safety margin is handled inside cost function
         return mppi_cost_function_compatible(states, controls, goal_tensor, env, robot_radius)

    # MPPI Parameters 
    mppi_horizon = 40 if ENV_TYPE == 'COMPLEX' else 30
    mppi_samples = 1500 if ENV_TYPE == 'COMPLEX' else 1000
    mppi_temperature = 0.05

    mppi_controller = MPPI(
        environment=env, dynamics_model=dynamics_model_wrapper, cost_function=cost_function_wrapper,
        horizon=mppi_horizon, num_samples=mppi_samples, temperature=mppi_temperature,
        control_dim=2, state_dim=3, dt=DT,
        sigma=np.array([0.2, 0.5]),
        u_min=np.array([-robot.max_linear_velocity, -robot.max_angular_velocity]),
        u_max=np.array([robot.max_linear_velocity, robot.max_angular_velocity]),
        device=device)
    mppi_controller.robot_radius = robot_radius # Pass actual radius

    
    visualizer = Visualizer(env) 

    # --- Simulation loop setup ---
    robot_states = [robot.get_state().copy()] # Store history
    all_sampled_trajectories_for_gif = [] # For animation

    print("Starting Simulation...")
    

    # --- Main control loop ---
    for i in range(SIM_STEPS):

        # --- Check if Visualizer window was closed ---
        if visualizer.fig is not None and not plt.fignum_exists(visualizer.fig.number):
             print("Plot window closed by user, stopping simulation.")
             break # Stop the loop

        print(f"Iteration {i+1}/{SIM_STEPS}")
        current_state = robot.get_state()

        # Check goal
        dist_to_goal = np.sqrt((current_state[0] - goal_np[0])**2 + (current_state[1] - goal_np[1])**2)
        if dist_to_goal < GOAL_THRESHOLD:
            print(f"Goal reached at iteration {i+1}!")
            # Optionally update plot one last time
            if visualizer.fig is None or not plt.fignum_exists(visualizer.fig.number):
                # Create figure if it was closed right before goal
                 visualizer.setup_plot(robot, goal_np)
            if visualizer.fig: # Check if figure exists
                 visualizer.update_time_text(f'Step: {i}/{SIM_STEPS}\nTime: {i * DT:.1f}s\nGoal Reached!')
                 visualizer.fig.canvas.draw(); visualizer.fig.canvas.flush_events()
                 plt.pause(1.0) # Show goal state briefly
            break # Exit loop

        # Optimize control
        start_time = time.time()
        optimal_control = mppi_controller.optimize(current_state, goal_np)
        current_rollouts = mppi_controller.get_last_rollouts()
        end_time = time.time()
        print(f"MPPI optimization took {(end_time - start_time):.4f} seconds")

        # Apply control & store state
        robot.update(optimal_control, DT)
        current_state_after_update = robot.get_state().copy()
        robot_states.append(current_state_after_update)
        visualizer.trajectory.append(current_state_after_update[:2]) # Update viz internal history

        # --- Real-time Visualization ---
        if i % PLOT_INTERVAL == 0:
             print("Updating plot...")
             
             visualizer.setup_plot(robot, goal_np) # Pass current robot state

             # 2. Update sampled trajectories overlay (plots on the axes setup above)
             if current_rollouts is not None:
                 visualizer.update_sampled_trajectories(current_rollouts)
                 # Store subset for final GIF animation
                 num_to_store = min(20, current_rollouts.shape[0])
                 indices = np.linspace(0, current_rollouts.shape[0]-1, num_to_store, dtype=int)
                 all_sampled_trajectories_for_gif.append(current_rollouts[indices, :, :2])

             # 3. Update time text
             visualizer.update_time_text(f'Step: {i}/{SIM_STEPS}\nTime: {i * DT:.1f}s')

             # 4. Display plot updates using the visualizer's figure
             if visualizer.fig: # Ensure figure exists
                 plt.show(block=False) # Ensure window is shown
                 visualizer.fig.canvas.draw_idle() # More robust way to draw
                 visualizer.fig.canvas.flush_events() # Process events
                 # plt.pause(0.01) # Pause can sometimes be unreliable
             else:
                 print("Warning: Visualizer figure doesn't exist for drawing.")


    # --- Post-Simulation ---
    print("Simulation finished.")
    plt.ioff() # Turn off interactive mode
    # Close the real-time window if it's still open
    if visualizer.fig is not None and plt.fignum_exists(visualizer.fig.number):
         plt.close(visualizer.fig)
    visualizer.fig, visualizer.ax = None, None # Reset visualizer state

    # Save static plot
    if SAVE_STATIC_PLOT and len(robot_states) > 1:
        print(f"Saving static path visualization to '{STATIC_PLOT_FILENAME}'...")
        plt.figure(figsize=(8, 8)); ax_static = plt.gca() # Use a new figure
        ax_static.set_xlim(0, env.width); ax_static.set_ylim(0, env.height)
        ax_static.set_aspect('equal'); ax_static.grid(True, linestyle='-', color='0.8')
        ax_static.add_patch(patches.Rectangle((0, 0), env.width, env.height, lw=2, ec='black', fc='none'))
        for obs in env.obstacles: # Plot obstacles
            if 'type' in obs and obs['type'] == 'rectangle': ax_static.add_patch(patches.Rectangle((obs['x'], obs['y']), obs['width'], obs['height'], lw=1, ec='black', fc='gray'))
            else: ax_static.add_patch(patches.Circle((obs['x'], obs['y']), obs['radius'], lw=1, ec='black', fc='gray'))
        path = np.array(robot_states) # Plot path
        ax_static.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Robot Path') 
        ax_static.plot(path[0, 0], path[0, 1], 'go', markersize=10, label='Start') 
        ax_static.plot(goal_np[0], goal_np[1], 'ro', markersize=10, label='Goal') 
        ax_static.set_title(f'MPPI Path Visualization ({env_name})'); ax_static.set_xlabel('X (m)'); ax_static.set_ylabel('Y (m)')
        ax_static.legend()
        plt.savefig(STATIC_PLOT_FILENAME, dpi=150)
        print(f"Static path saved to '{STATIC_PLOT_FILENAME}'"); plt.close()

    
    if SAVE_ANIMATION and len(robot_states) > 1:
        print("Creating animation GIF...")
        try:
            # Re-create visualizer instance for clean animation figure
            anim_visualizer = Visualizer(env)
            animation = anim_visualizer.create_animation(
                robot_states=robot_states, goal=goal_np, filename=ANIMATION_FILENAME, fps=ANIMATION_FPS,
                all_sampled_trajectories=all_sampled_trajectories_for_gif, sample_interval=PLOT_INTERVAL )
        except Exception as e:
            print(f"Error creating animation: {e}"); import traceback; traceback.print_exc()
    else: print("Animation saving skipped.")

    print("Script finished.")
    # plt.show() # Keep plot windows open if needed (often not necessary)

if __name__ == "__main__":
    main()