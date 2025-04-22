# main_gpmppi.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import torch
import os
import traceback 

from matplotlib.animation import FuncAnimation
try:
    import PIL # Used by visualizer for saving animation
except ImportError:
    print("Warning: Pillow library not found. GIF saving might fail.")
    print("Install using: pip install pillow")
try:
    import sklearn 
except ImportError:
     print("Error: scikit-learn not found. Please install it: pip install scikit-learn")
     exit()


# Import components
try:
    from environment import Environment
    from robot import TurtleBot
    from visualization import Visualizer 
    from mppi_planner import MPPI 
    from gp_mapper import GPMapper 
    from sensor import LidarSensor 
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Ensure all .py files (environment, robot, visualization, mppi_planner, gp_mapper, sensor) are present and correct.")
    exit()


# --- Configuration ---
# Simulation settings
DT = 0.1                      
SIM_STEPS = 600               
GOAL_THRESHOLD = 0.3          
PLOT_INTERVAL = 3             

# --- Output Configuration ---
SAVE_STATIC_PLOT = True       # Save a PNG of the final path?
STATIC_PLOT_FILENAME = "gp_mppi_final_path.png" # Filename for GP-MPPI static plot
SAVE_ANIMATION = True         # Save a GIF animation?
ANIMATION_FILENAME = "gp_mppi_animation.gif" # Filename for GP-MPPI animation
ANIMATION_FPS = 15            # GIF frames per second
ANIMATION_DPI = 100           # GIF resolution

# --- Environment Selection ---
# Choose which environment function to call:
#ENV_TYPE = 'SIMPLE'
#ENV_TYPE = 'MODERATE'
ENV_TYPE = 'COMPLEX'

# --- GP & Sensor Parameters ---
GP_MAX_POINTS = 750 # Max points for GP training dataset
SENSOR_NUM_BEAMS = 40
SENSOR_MAX_RANGE = 5.0
SENSOR_ANGLE_SPAN = np.pi * 1.5 # 270 degrees


# --- Cost Function Definition (Uses GP Map) ---
def gp_mppi_cost_function(states, controls, goal, gp_mapper, robot_radius, # Pass gp_mapper
                          goal_weight=25.0, gp_collision_weight=800.0, # Tuned GP cost weight
                          ctrl_v_weight=0.1, ctrl_o_weight=0.1,
                          goal_orient_weight=1.0):
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
    if gp_mapper is not None and gp_mapper.fitted: # Check if mapper is fitted
        query_points_flat = states[:, :, :2].reshape(-1, 2).cpu().numpy()
        # Ensure prediction happens only if points exist
        if query_points_flat.shape[0] > 0:
            try:
                gp_mean_flat, _ = gp_mapper.predict(query_points_flat) # Ignore std dev for now
                # Clamp predictions to avoid extreme values if necessary (optional)
                # gp_mean_flat = np.clip(gp_mean_flat, 0.0, 1.0)
                gp_mean = torch.tensor(gp_mean_flat.reshape(K, T_plus_1), dtype=torch.float32, device=device)
                # Sum predicted occupancy probability along the trajectory
                collision_costs = torch.sum(gp_mean, dim=1) * gp_collision_weight
                total_costs += collision_costs
            except Exception as e: # Catch potential errors during GP prediction
                print(f"Warning: GP prediction failed during cost calculation: {e}")
                # Optionally add a default high cost or handle differently
        else:
             # No points to query, no collision cost added
             pass
    # else: # If GP not fitted, maybe add a small default collision cost or rely on other costs
    #    pass

    return total_costs

# --- Main Simulation Function ---
def main():
    # Create environment
    if ENV_TYPE == 'SIMPLE': env = Environment.create_simple_environment(); start_pose_np=np.array([1.0,1.0,0.0]); goal_np=np.array([8.0,8.0,np.pi/4]); env_name="Simple"
    elif ENV_TYPE == 'MODERATE': env = Environment.create_moderate_environment(); start_pose_np=np.array([1.0,1.0,np.pi/4]); goal_np=np.array([9.0,9.0,0.0]); env_name="Moderate"
    elif ENV_TYPE == 'COMPLEX': env = Environment.create_complex_cluttered_environment(); start_pose_np=np.array([0.5,0.5,np.pi/4]); goal_np=np.array([9.5,9.5,-np.pi/4]); env_name="Complex"
    else: raise ValueError(f"Unknown ENV_TYPE: {ENV_TYPE}")
    print(f"Using Environment: {env_name}")

    # Create Robot, Sensor, Mapper
    robot = TurtleBot(x=start_pose_np[0], y=start_pose_np[1], theta=start_pose_np[2])
    sensor = LidarSensor(num_beams=SENSOR_NUM_BEAMS, max_range=SENSOR_MAX_RANGE, angle_span=SENSOR_ANGLE_SPAN)
    gp_mapper = GPMapper(max_points=GP_MAX_POINTS)

    # --- Setup MPPI controller ---
    def dynamics_model_wrapper(state_tensor, control_tensor, dt): # Inefficient loop version
        next_states = []; state_np=state_tensor.cpu().numpy(); control_np=control_tensor.cpu().numpy()
        # Make sure robot's internal state is updated before calling dynamics if needed
        # (Current TurtleBot dynamics uses the passed state directly, so this is okay)
        temp_next_states_np = []
        for i in range(state_np.shape[0]):
            temp_next_states_np.append(robot.dynamics(state_np[i], control_np[i], dt))
        # Convert list of numpy arrays back to a tensor
        next_states_tensor = torch.tensor(np.array(temp_next_states_np), dtype=torch.float32, device=state_tensor.device)
        return next_states_tensor


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    goal_tensor = torch.tensor(goal_np, dtype=torch.float32, device=device)
    robot_radius = robot.radius

    # Cost function wrapper now captures gp_mapper
    def cost_function_wrapper(states, controls):
         # Pass gp_mapper instance to the cost function
         # Adjust GP weight based on environment complexity
         gp_weight = 750.0 
         if ENV_TYPE == 'COMPLEX':
             gp_weight = 1200.0 # Higher weight for complex environment
         elif ENV_TYPE == 'MODERATE':
              gp_weight = 900.0 # Slightly higher for moderate

         return gp_mppi_cost_function(states, controls, goal_tensor, gp_mapper, robot_radius,
                                        gp_collision_weight=gp_weight)

    # MPPI Parameters may need tuning for GP-MPPI
    mppi_horizon = 30 # Planning horizon steps
    mppi_samples = 1000 # Number of rollouts (K)
    mppi_temperature = 0.05 # Exploration factor (lambda_) - lower is less exploration

    # Adjust sigma (control noise) - potentially smaller if GP map is reliable
    mppi_sigma = np.array([0.2, 0.5]) # [linear_vel_std, angular_vel_std]

    mppi_controller = MPPI(
        environment=env, dynamics_model=dynamics_model_wrapper, cost_function=cost_function_wrapper, # Use GP cost func
        horizon=mppi_horizon, num_samples=mppi_samples, temperature=mppi_temperature,
        control_dim=2, state_dim=3, dt=DT, sigma=mppi_sigma,
        u_min=np.array([-robot.max_linear_velocity, -robot.max_angular_velocity]),
        u_max=np.array([robot.max_linear_velocity, robot.max_angular_velocity]), device=device)
    mppi_controller.robot_radius = robot_radius

    # Create visualizer
    visualizer = Visualizer(env)

    # --- Simulation loop setup ---
    history = { 'poses': [robot.get_state().copy()], 'rollouts': [], 'gp_map_data': [],
                'goal_reached': False, 'collision_detected': False, 'steps_taken': 0 }

    print("Starting GP-MPPI Simulation...")
    plt.ion(); # Turn interactive mode on

    # --- Main GP-MPPI control loop ---
    for i in range(SIM_STEPS):
        
        if visualizer.fig is not None and not plt.fignum_exists(visualizer.fig.number):
             print("Plot window closed by user, stopping simulation.")
             break

        loop_start_time = time.time()
        print(f"\n--- Iteration {i+1}/{SIM_STEPS} ---")
        current_state = robot.get_state()
        print(f"Current State: x={current_state[0]:.2f}, y={current_state[1]:.2f}, th={current_state[2]:.2f}")


        # Goal Check
        dist_to_goal = np.sqrt((current_state[0] - goal_np[0])**2 + (current_state[1] - goal_np[1])**2)
        print(f"Distance to goal: {dist_to_goal:.3f}")
        if dist_to_goal < GOAL_THRESHOLD:
            print(f"\nGoal reached at iteration {i+1}!")
            history['goal_reached'] = True;
            history['steps_taken'] = i # Store the number of steps completed
            break

        # 1. Sense
        sense_start_time = time.time()
        gp_update_data = sensor.get_scan_data_for_gp(current_state, env)
        # print(f"Sensing time: {(time.time() - sense_start_time):.4f}s")
        # print(f"GP Data Points: {len(gp_update_data['X'])}")


        # 2. Update GP Map
        map_update_start_time = time.time()
        try:
            gp_mapper.update(gp_update_data)
            # print(f"GP Map Update time: {(time.time() - map_update_start_time):.4f}s")
            if gp_mapper.fitted:
                 print(f"GP Map fitted with {gp_mapper.gp.X_train_.shape[0]} points.")
            else:
                 print("GP Map not fitted yet.")
        except Exception as e:
             print(f"Error during GP Map update: {e}")
             traceback.print_exc()
             # Continue simulation even if map update fails? Or break?
             # break


        # 3. Store GP Map data for GIF
        map_bounds = (env.width * -0.1, env.width * 1.1, env.height * -0.1, env.height * 1.1)
        if gp_mapper.fitted:
            try:
                xx, yy, zz_mean = gp_mapper.get_map_data_for_plotting(map_bounds, grid_res=0.3)
                history['gp_map_data'].append({'xx':xx, 'yy':yy, 'zz_mean':zz_mean} if xx is not None else None)
            except Exception as e:
                 print(f"Error getting GP map data for plotting: {e}")
                 history['gp_map_data'].append(None) # Append None if plotting fails
        else:
            history['gp_map_data'].append(None) # Append None if not fitted


        # 4. Plan using GP-informed Cost Function
        plan_start_time = time.time()
        current_rollouts = None # Initialize
        try:
            # Ensure current_state is numpy array for optimize function if needed
            current_state_np = current_state if isinstance(current_state, np.ndarray) else np.array(current_state)
            optimal_control = mppi_controller.optimize(current_state_np, goal_np)
            current_rollouts = mppi_controller.get_last_rollouts() # Get rollouts ( K x T+1 x state_dim )
            print(f"Planner time: {(time.time() - plan_start_time):.4f}s")
            print(f"Optimal control: v={optimal_control[0]:.3f}, w={optimal_control[1]:.3f}")
        except Exception as e:
             print(f"\n!!! Error during MPPI optimization: {e} !!!")
             traceback.print_exc()
             print("!!! Setting control to zero and attempting to continue !!!")
             optimal_control = np.array([0.0, 0.0]) # Set zero control on failure
             # break # Optionally stop simulation on planner failure


        # 5. Apply control & store state
        robot.update(optimal_control, DT)
        current_state_after_update = robot.get_state().copy()
        history['poses'].append(current_state_after_update)
        # Ensure rollouts are stored correctly (convert from torch if needed)
        if current_rollouts is not None and isinstance(current_rollouts, torch.Tensor):
             history['rollouts'].append(current_rollouts.cpu().numpy())
        else:
              history['rollouts'].append(current_rollouts) # Store None or numpy array

        history['steps_taken'] = i + 1 # Record step completed
        visualizer.trajectory.append(current_state_after_update[:2]) # Update viz internal history

        # 6. Collision Check (Ground Truth - for failure detection)
        if env.check_collision(current_state_after_update[0], current_state_after_update[1], robot_radius):
             print(f"COLLISION DETECTED (Ground Truth) at step {i+1}!")
             history['collision_detected'] = True;
             # break # Optionally stop simulation on collision

        # --- Real-time Visualization ---
        if i % PLOT_INTERVAL == 0 or history['collision_detected'] : # Update on interval or if collision happens
            plot_start_time = time.time()
            print("Updating plot...")
            try:
                visualizer.setup_plot(robot, goal_np) 
                visualizer.update_gp_map_display(gp_mapper, map_bounds, grid_res=0.3) 
                if current_rollouts is not None:
                    # Convert rollouts to numpy if they are tensors for visualization
                    rollouts_np = current_rollouts.cpu().numpy() if isinstance(current_rollouts, torch.Tensor) else current_rollouts
                    visualizer.update_sampled_trajectories(rollouts_np) # Add rollouts overlay

                visualizer.update_time_text(f'Step: {i+1}/{SIM_STEPS}\nTime: {(i+1) * DT:.1f}s') # Update text
                plt.title(f'GP-MPPI Navigation ({env_name}) - Step {i+1}') # Update title

                if visualizer.fig:
                    plt.show(block=False);
                    visualizer.fig.canvas.draw_idle();
                    visualizer.fig.canvas.flush_events()
                # print(f"Plotting time: {(time.time() - plot_start_time):.4f}s")
            except Exception as e:
                print(f"Error during real-time plotting: {e}")
                traceback.print_exc()
                # Close the figure if plotting fails badly?
                if visualizer.fig is not None and plt.fignum_exists(visualizer.fig.number):
                    plt.close(visualizer.fig)
                visualizer.fig, visualizer.ax = None, None # Reset visualizer state

        print(f"Iteration {i+1} total time: {(time.time() - loop_start_time):.4f}s")


    # --- Post-Simulation ---
    # If loop finished naturally (max steps reached) and goal wasn't reached
    if not history['goal_reached'] and history['steps_taken'] == SIM_STEPS:
         print(f"\nMax simulation steps ({SIM_STEPS}) reached.")
         # Ensure final map/rollout data corresponds to the last completed step
         # The loop already appended data for the last step (SIM_STEPS - 1)

    # If goal was reached, the loop broke *before* appending data for the breaking step `i`.
    # The last recorded step is history['steps_taken'] = i.
    # We have poses up to history['steps_taken'].
    # We have map/rollout data up to history['steps_taken'] - 1.
    # We might want to add the final GP map state if goal was reached.
    elif history['goal_reached']:
        print(f"Simulation ended: Goal Reached after {history['steps_taken']} steps.")
        final_state = history['poses'][-1]
        print(f"Final State: x={final_state[0]:.2f}, y={final_state[1]:.2f}, th={final_state[2]:.2f}")
        # Add final GP map data based on the state where goal was reached
        if gp_mapper.fitted:
            try:
                xx, yy, zz_mean = gp_mapper.get_map_data_for_plotting(map_bounds, grid_res=0.3)
                history['gp_map_data'].append({'xx':xx, 'yy':yy, 'zz_mean':zz_mean} if xx is not None else None)
            except Exception as e:
                print(f"Error getting final GP map data for plotting: {e}")
                history['gp_map_data'].append(None)
        else:
             history['gp_map_data'].append(None)


    elif history['collision_detected']:
         print(f"Simulation ended: Collision Detected after {history['steps_taken']} steps.")
         # Similar logic, data exists up to the step *before* collision if loop broke.


    print("Simulation finished."); plt.ioff() # Turn interactive mode off
    if visualizer.fig is not None and plt.fignum_exists(visualizer.fig.number):
        plt.close(visualizer.fig) # Close the real-time plot window
    visualizer.fig, visualizer.ax = None, None # Clear visualizer state


    # --- Save Static Plot (with final GP Map) ---
    if SAVE_STATIC_PLOT and len(history['poses']) > 1:
        print(f"\nSaving static path visualization to '{STATIC_PLOT_FILENAME}'...")
        try:
            plt.figure(figsize=(8, 8)); ax_static = plt.gca()
            ax_static.set_xlim(0, env.width); ax_static.set_ylim(0, env.height); ax_static.set_aspect('equal');
            ax_static.grid(True, linestyle='-', color='0.8', zorder=0) # Add grid

            # Plot FINAL GP Map if available
            # Ensure index is valid before accessing
            final_map_index = len(history['gp_map_data']) - 1
            if final_map_index >= 0 and history['gp_map_data'][final_map_index]:
                 gp_data = history['gp_map_data'][final_map_index]
                 # Check if data inside dictionary is valid before plotting
                 if gp_data['xx'] is not None and gp_data['yy'] is not None and gp_data['zz_mean'] is not None:
                     ax_static.contourf(gp_data['xx'], gp_data['yy'], gp_data['zz_mean'],
                                         levels=np.linspace(0, 1, 11), cmap='viridis', alpha=0.4, zorder=-1, extend='neither')
                 else:
                      print("Warning: Final GP map data dictionary contains None values, map not plotted.")

            # Plot Env boundaries and ground truth obstacles lightly
            ax_static.add_patch(patches.Rectangle((0, 0), env.width, env.height, lw=2, ec='black', fc='none', zorder=1))
            for obs in env.obstacles:
                 if 'type' in obs and obs['type'] == 'rectangle': ax_static.add_patch(patches.Rectangle((obs['x'], obs['y']), obs['width'], obs['height'], lw=1, ec='maroon', fc='red', alpha=0.25, zorder=2)) # Make obstacles more visible
                 else: ax_static.add_patch(patches.Circle((obs['x'], obs['y']), obs['radius'], lw=1, ec='maroon', fc='red', alpha=0.25, zorder=2))

            # Plot Path, Start, Goal
            path = np.array(history['poses']);
            ax_static.plot(path[:, 0], path[:, 1], color='#2ca02c', linestyle='-', linewidth=2.5, label='Robot Path', zorder=5); # Thicker, green path
            ax_static.plot(path[0, 0], path[0, 1], marker='o', color='blue', markersize=10, label='Start', zorder=6, linestyle='None') # Blue start
            ax_static.plot(goal_np[0], goal_np[1], marker='*', color='red', markersize=14, label='Goal', zorder=6, linestyle='None') # Red star goal
            # Plot final robot position if different from goal
            if not history['goal_reached'] or dist_to_goal >= GOAL_THRESHOLD:
                 ax_static.plot(path[-1, 0], path[-1, 1], marker='x', color='#17becf', markersize=10, label='End', zorder=6, linestyle='None') # Cyan end marker

            ax_static.set_title(f'GP-MPPI Path ({env_name}) - {"Goal Reached" if history["goal_reached"] else ("Collision" if history["collision_detected"] else "Finished")}')
            ax_static.set_xlabel('X (m)'); ax_static.set_ylabel('Y (m)'); ax_static.legend(loc='best')
            plt.savefig(STATIC_PLOT_FILENAME, dpi=150, bbox_inches='tight'); print(f"Static path saved to '{STATIC_PLOT_FILENAME}'"); plt.close()
        except Exception as e:
             print(f"Error saving static plot: {e}")
             traceback.print_exc()
             if plt.fignum_exists(plt.gcf().number): plt.close() # Close figure if error occurs


    # --- Create Animation with GP Map ---
    if SAVE_ANIMATION and len(history['poses']) > 1:
        print("\nCreating animation GIF (with GP map)...")
        try:
            
            # Ensure all data lists match the number of poses exactly.
            num_poses = len(history['poses']) # Number of frames needed
            print(f"Poses recorded: {num_poses}")
            print(f"GP Maps recorded before padding: {len(history['gp_map_data'])}")
            print(f"Rollouts recorded before padding: {len(history['rollouts'])}")


            while len(history['gp_map_data']) < num_poses:
                 history['gp_map_data'].append(None) # Pad with None
            # Ensure we only keep as many map entries as poses (truncation safety)
            history['gp_map_data'] = history['gp_map_data'][:num_poses]

            while len(history['rollouts']) < num_poses:
                 history['rollouts'].append(None) # Pad with None
            # Ensure we only keep as many rollout entries as poses (truncation safety)
            history['rollouts'] = history['rollouts'][:num_poses]
            # --- !! END ROBUST PADDING !! ---

            # Check lengths just before animation call (for debugging)
            print(f"Animation data lengths AFTER padding: poses={len(history['poses'])}, gp_maps={len(history['gp_map_data'])}, rollouts={len(history['rollouts'])}")

            if not all(len(lst) == num_poses for lst in [history['poses'], history['gp_map_data'], history['rollouts']]):
                 raise ValueError("Padding failed! List lengths do not match number of poses.")

            anim_visualizer = Visualizer(env) # Fresh visualizer for animation
            # Call the animation method that includes GP map plotting
            animation = anim_visualizer.create_animation_with_gp(
                 robot_states=history['poses'],
                 goal=goal_np,
                 all_gp_map_data=history['gp_map_data'], # Pass GP map history
                 filename=ANIMATION_FILENAME,
                 fps=ANIMATION_FPS,
                 all_sampled_trajectories=history['rollouts'], # Pass rollouts history
                 sample_interval=1 # Plot data associated with each frame
             )
            

        except AttributeError as ae:
             print(f"\nAttributeError creating animation: {ae}")
             print("Ensure 'create_animation_with_gp' method exists and is correctly named in visualization.py\n")
             traceback.print_exc()
        except ValueError as ve: # Catch padding errors
             print(f"\nValueError creating animation: {ve}")
             traceback.print_exc()
        except Exception as e:
             print(f"\nError creating or saving animation: {e}")
             traceback.print_exc()
    elif not SAVE_ANIMATION:
         print("\nAnimation saving skipped (SAVE_ANIMATION=False).")
    else: # len(history['poses']) <= 1
         print("\nAnimation saving skipped (not enough poses).")


    print("\nScript finished.")
    # plt.show() # Keep commented out unless debugging interactively

if __name__ == "__main__":
    main()