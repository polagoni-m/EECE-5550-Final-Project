# comparison_script.py
# Runs MPPI-only and GP-MPPI simulations sequentially in a chosen environment,
# gathers statistics, and prints a comparison table.

import numpy as np
import torch
import time
import traceback

# --- Import Components ---
try:
    from environment import Environment
    from robot import TurtleBot
    from mppi_planner import MPPI
    from gp_mapper import GPMapper
    from sensor import LidarSensor
    # Check for scikit-learn
    import sklearn
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Ensure all .py files are present and libraries (sklearn, torch, numpy, matplotlib) are installed.")
    exit()

# --- Configuration ---
# Choose Environment
# ENV_TYPE = 'SIMPLE'
# ENV_TYPE = 'MODERATE'
ENV_TYPE = 'COMPLEX'

# Simulation settings
DT = 0.1
SIM_STEPS = 600           # Max iterations
GOAL_THRESHOLD = 0.35     
ROBOT_RADIUS = 0.15       # Default from TurtleBot class

# Common MPPI Params 
MPPI_HORIZON = 35
MPPI_SAMPLES = 1500 # More samples generally better for complex envs
MPPI_TEMPERATURE = 0.05
MPPI_SIGMA = np.array([0.2, 0.5]) # Noise std dev [v, omega]


MPPI_COST_PARAMS = {
    'goal_weight': 25.0,
    'collision_weight': 1500.0, # High penalty for known collisions
    'ctrl_v_weight': 0.1,
    'ctrl_o_weight': 0.1,
    'goal_orient_weight': 1.0,
    'safety_margin': 0.1 # Safety margin for known obstacles
}
# For GP-MPPI (using GP map)
GPMPPI_COST_PARAMS = {
    'goal_weight': 25.0,
    'gp_collision_weight': 800.0, # Weight for GP-predicted collisions
    'ctrl_v_weight': 0.1,
    'ctrl_o_weight': 0.1,
    'goal_orient_weight': 1.0
    # safety_margin is implicit in GP map learning/prediction
}
# GP & Sensor Params (for GP-MPPI only)
GP_MAX_POINTS = 750
SENSOR_NUM_BEAMS = 40
SENSOR_MAX_RANGE = 5.0
SENSOR_ANGLE_SPAN = np.pi * 1.5

# --- Cost Function Definitions ---

# 1. Cost Function for Standard MPPI 
def mppi_cost_function_basic(states, controls, goal, env, robot_radius,
                              goal_weight, collision_weight, ctrl_v_weight,
                              ctrl_o_weight, goal_orient_weight, safety_margin):
    """ Calculates cost using known environment map for collisions. """
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
    # Collision Cost (Uses effective_radius and known env.check_collision)
    collision_penalty = torch.zeros(K, dtype=torch.float32, device=device)
    states_np = states.cpu().numpy()
    for k in range(K):
        collision_found = False
        for t in range(T_plus_1):
            x, y = states_np[k, t, 0], states_np[k, t, 1]
            if env.check_collision(x, y, effective_radius): # Check known map
                collision_found = True; break
        if collision_found: collision_penalty[k] = collision_weight
    total_costs += collision_penalty
    return total_costs

# 2. Cost Function for GP-MPPI (uses GP map)
def gp_mppi_cost_function(states, controls, goal, gp_mapper, robot_radius, # Takes gp_mapper
                           goal_weight, gp_collision_weight, ctrl_v_weight,
                           ctrl_o_weight, goal_orient_weight):
    """ Calculates cost using GP map for collisions. """
    K, T_plus_1, state_dim = states.shape; T = controls.shape[1]; device = states.device
    total_costs = torch.zeros(K, dtype=torch.float32, device=device)
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
    # GP Collision Cost
    query_points_flat = states[:, :, :2].reshape(-1, 2).cpu().numpy()
    gp_mean_flat, _ = gp_mapper.predict(query_points_flat)
    gp_mean = torch.tensor(gp_mean_flat.reshape(K, T_plus_1), dtype=torch.float32, device=device)
    collision_costs = torch.sum(gp_mean, dim=1) * gp_collision_weight # Sum probability
    total_costs += collision_costs
    return total_costs


# --- Simulation Runner Function ---
def run_simulation(planner_type, env, start_pose_np, goal_np, robot_radius, common_mppi_params, cost_params, sim_config):
    """ Runs one simulation trial and returns statistics. """
    print(f"\n--- Running Simulation: {planner_type} ---")
    stats = { 'planner_type': planner_type, 'success': False, 'collision': False, 'steps': 0,
              'time': 0.0, 'path_len': 0.0, 'min_clear': float('inf'), 'avg_plan_t': 0.0,
              'avg_gp_t': 0.0, 'avg_sens_t': 0.0, 'avg_step_t': 0.0 }
    history = {'poses': [start_pose_np.copy()]}
    timings = {'planner': [], 'gp_update': [], 'sensor': [], 'step': []}

    # Create Robot
    robot = TurtleBot(x=start_pose_np[0], y=start_pose_np[1], theta=start_pose_np[2])

    # Device and Goal Tensor
    device = common_mppi_params['device']
    goal_tensor = torch.tensor(goal_np, dtype=torch.float32, device=device)

    # Setup specific planner components
    if planner_type == 'GP-MPPI':
        gp_mapper = GPMapper(max_points=GP_MAX_POINTS)
        sensor = LidarSensor(num_beams=SENSOR_NUM_BEAMS, max_range=SENSOR_MAX_RANGE, angle_span=SENSOR_ANGLE_SPAN)
        def cost_wrapper(states, controls):
            # Pass only the required params for GP cost function
            return gp_mppi_cost_function(states, controls, goal_tensor, gp_mapper, robot_radius, **cost_params)
    elif planner_type == 'MPPI':
        # Standard MPPI doesn't need GP or Sensor
        gp_mapper = None
        sensor = None
        def cost_wrapper(states, controls):
            # Pass only the required params for basic cost function
             return mppi_cost_function_basic(states, controls, goal_tensor, env, robot_radius, **cost_params)
    else: raise ValueError("Unknown planner_type")

    
    def dynamics_model_wrapper(state_tensor, control_tensor, dt):
        next_states = []; state_np=state_tensor.cpu().numpy(); control_np=control_tensor.cpu().numpy()
        for i in range(state_tensor.shape[0]): next_states.append(torch.tensor(robot.dynamics(state_np[i], control_np[i], dt), dtype=torch.float32, device=state_tensor.device))
        return torch.stack(next_states)

    # Create Planner
    mppi_controller = MPPI( environment=env, dynamics_model=dynamics_model_wrapper, cost_function=cost_wrapper, **common_mppi_params )
    mppi_controller.robot_radius = robot_radius

    # Simulation Loop
    for i in range(sim_config['max_steps']):
        step_start_time = time.time()
        current_state = robot.get_state()

        # Goal Check
        dist_to_goal = np.sqrt((current_state[0] - goal_np[0])**2 + (current_state[1] - goal_np[1])**2)
        if dist_to_goal < sim_config['goal_threshold']:
            print(f"Goal reached at iteration {i+1}!")
            stats['success'] = True; stats['steps'] = i; break

        # Sensing & Mapping (GP-MPPI only)
        if planner_type == 'GP-MPPI':
            sens_start = time.time()
            gp_update_data = sensor.get_scan_data_for_gp(current_state, env)
            timings['sensor'].append(time.time() - sens_start)
            map_start = time.time()
            gp_mapper.update(gp_update_data)
            timings['gp_update'].append(time.time() - map_start)

        # Planning
        plan_start = time.time()
        optimal_control = mppi_controller.optimize(current_state, goal_np)
        timings['planner'].append(time.time() - plan_start)

        # Step
        robot.update(optimal_control, sim_config['dt'])
        next_state = robot.get_state()
        history['poses'].append(next_state.copy())
        stats['steps'] = i + 1

        # Collision Check (Ground Truth)
        if env.check_collision(next_state[0], next_state[1], robot_radius):
             print(f"COLLISION DETECTED (Ground Truth) at step {i+1}!")
             stats['collision'] = True; # break # Optionally stop

        # Record step time
        timings['step'].append(time.time() - step_start_time)

        # Print progress minimally
        if i % 50 == 0: print(f"  Step {i}...")

    # --- Calculate Final Stats ---
    stats['time'] = stats['steps'] * sim_config['dt']
    path_array = np.array(history['poses'])
    if len(path_array) > 1:
        stats['path_len'] = np.sum(np.sqrt(np.sum(np.diff(path_array[:, :2], axis=0)**2, axis=1)))
        for pose in path_array:
             dist = env.distance_to_obstacles(pose[0], pose[1])
             stats['min_clear'] = min(stats['min_clear'], dist)
    if timings['planner']: stats['avg_plan_t'] = np.mean(timings['planner'])
    if timings['gp_update']: stats['avg_gp_t'] = np.mean(timings['gp_update'])
    if timings['sensor']: stats['avg_sens_t'] = np.mean(timings['sensor'])
    if timings['step']: stats['avg_step_t'] = np.mean(timings['step'])

    print(f"--- {planner_type} Run Finished ---")
    return stats


# --- Main Execution ---
if __name__ == "__main__":

    # --- Define Configurations ---
    sim_config = {
        'dt': DT,
        'max_steps': SIM_STEPS,
        'goal_threshold': GOAL_THRESHOLD
    }
    robot_config = {
        'radius': ROBOT_RADIUS
    }
    common_mppi_params = {
        'horizon': MPPI_HORIZON,
        'num_samples': MPPI_SAMPLES,
        'temperature': MPPI_TEMPERATURE,
        'control_dim': 2,
        'state_dim': 3,
        'dt': DT,
        'sigma': MPPI_SIGMA, # Pass std dev array
        'u_min': np.array([-TurtleBot().max_linear_velocity, -TurtleBot().max_angular_velocity]), # Use default limits from class
        'u_max': np.array([TurtleBot().max_linear_velocity, TurtleBot().max_angular_velocity]),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # --- Select Environment ---
    if ENV_TYPE == 'SIMPLE': env = Environment.create_simple_environment(); start_p=np.array([1.,1.,0.]); goal_p=np.array([8.,8.,np.pi/4]); name="Simple"
    elif ENV_TYPE == 'MODERATE': env = Environment.create_moderate_environment(); start_p=np.array([1.,1.,np.pi/4]); goal_p=np.array([9.,9.,0.]); name="Moderate"
    elif ENV_TYPE == 'COMPLEX': env = Environment.create_complex_cluttered_environment(); start_p=np.array([0.5,0.5,np.pi/4]); goal_p=np.array([9.5,9.5,-np.pi/4]); name="Complex"
    else: raise ValueError(f"Unknown ENV_TYPE: {ENV_TYPE}")
    print(f"===== Starting Comparison for Environment: {name} =====")

    # --- Run MPPI Simulation ---
    mppi_stats = run_simulation(
        planner_type='MPPI',
        env=env, start_pose_np=start_p, goal_np=goal_p,
        robot_radius=ROBOT_RADIUS, common_mppi_params=common_mppi_params,
        cost_params=MPPI_COST_PARAMS, # Pass MPPI specific cost weights
        sim_config=sim_config
    )

    # --- Run GP-MPPI Simulation ---
    gp_mppi_stats = run_simulation(
        planner_type='GP-MPPI',
        env=env, start_pose_np=start_p, goal_np=goal_p, # Same start/goal/env
        robot_radius=ROBOT_RADIUS, common_mppi_params=common_mppi_params, # Same core MPPI params
        cost_params=GPMPPI_COST_PARAMS, # Pass GP-MPPI specific cost weights
        sim_config=sim_config
    )

    # --- Print Comparison Table ---
    print("\n\n===== Comparison Results =====")
    print(f"{'Metric':<25} | {'MPPI':<15} | {'GP-MPPI':<15}")
    print("-" * 60)
    print(f"{'Success (Goal Reached)':<25} | {str(mppi_stats['success']):<15} | {str(gp_mppi_stats['success']):<15}")
    print(f"{'Collision Occurred':<25} | {str(mppi_stats['collision']):<15} | {str(gp_mppi_stats['collision']):<15}")
    print(f"{'Steps Taken':<25} | {mppi_stats['steps']:<15} | {gp_mppi_stats['steps']:<15}")
    print(f"{'Simulated Time (s)':<25} | {mppi_stats['time']:.2f}{'' if mppi_stats['success'] else ' (DNF)' :<15} | {gp_mppi_stats['time']:.2f}{'' if gp_mppi_stats['success'] else ' (DNF)' :<15}")
    print(f"{'Path Length (m)':<25} | {mppi_stats['path_len']:.3f}{'' if mppi_stats['success'] else ' (N/A)' :<15} | {gp_mppi_stats['path_len']:.3f}{'' if gp_mppi_stats['success'] else ' (N/A)' :<15}")
    print(f"{'Min Obstacle Clearance (m)':<25} | {mppi_stats['min_clear']:.3f}{'' if mppi_stats['path_len']>0 else ' (N/A)' :<15} | {gp_mppi_stats['min_clear']:.3f}{'' if gp_mppi_stats['path_len']>0 else ' (N/A)' :<15}")
    print("-" * 60)
    print("Avg. Computation Time per Step (s):")
    print(f"{'  Planner':<23} | {mppi_stats['avg_plan_t']:.4f}{'':<15} | {gp_mppi_stats['avg_plan_t']:.4f}{'':<15}")
    print(f"{'  Sensor':<23} | {'N/A':<15} | {gp_mppi_stats['avg_sens_t']:.4f}{'':<15}")
    print(f"{'  GP Map Update':<23} | {'N/A':<15} | {gp_mppi_stats['avg_gp_t']:.4f}{'':<15}")
    print(f"{'  Total Step':<23} | {mppi_stats['avg_step_t']:.4f}{'':<15} | {gp_mppi_stats['avg_step_t']:.4f}{'':<15}")
    print("-" * 60)