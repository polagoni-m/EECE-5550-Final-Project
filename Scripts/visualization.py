# visualization.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import os

# Define consistent colors for the target style
COLOR_PATH = '#2ca02c'       # Green
COLOR_ROBOT = '#17becf'      # Teal/Cyan
COLOR_GOAL = '#d62728'       # Red
COLOR_ROLLOUTS = '#17becf'   # Teal/Cyan
COLOR_OBSTACLE_CIRCLE_FILL = 'red'  # Fill color for circles
COLOR_OBSTACLE_RECT_FILL = 'red'  # Fill color for rectangles 
# COLOR_OBSTACLE_RECT_FILL = 'black' # Or keep black if preferred
COLOR_OBSTACLE_EDGE = 'black'  # Outline color
COLOR_GRID = '0.8'           # Light grey
GP_MAP_CMAP = 'viridis'      # Colormap for GP map

class Visualizer:
    """ Visualization tools, styled to match the target image, includes GP map plotting. """
    def __init__(self, environment, figsize=(8, 8)):
        """ Initialize visualizer. """
        self.env = environment
        self.figsize = figsize
        self.fig = None
        self.ax = None
        self.robot_circle = None
        self.trajectory_line = None
        self.sampled_lines = []
        self.time_text = None
        self.gp_contour_plot = None 
        self.trajectory = [] # Store path history internally
        self.current_sampled_trajectories = None 

    def _plot_obstacles(self):
        # Plot obstacles using defined colors with transparency
        for obs in self.env.obstacles:
            if 'type' in obs and obs['type'] == 'rectangle':
                self.ax.add_patch(patches.Rectangle(
                    (obs['x'], obs['y']), obs['width'], obs['height'],
                    linewidth=1, edgecolor=COLOR_OBSTACLE_EDGE, 
                    facecolor=COLOR_OBSTACLE_RECT_FILL, alpha=0.5, zorder=1))  
            else:
                self.ax.add_patch(patches.Circle(
                    (obs['x'], obs['y']), obs['radius'],
                    linewidth=1, edgecolor=COLOR_OBSTACLE_EDGE, 
                    facecolor=COLOR_OBSTACLE_CIRCLE_FILL, alpha=0.5, zorder=1))  

    def setup_plot(self, robot, goal):
        """ Set up plot or clear and re-setup for updates, matching target style. """
        if self.fig is None or self.ax is None or not plt.fignum_exists(self.fig.number):
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
            plt.ion() 
        else:
             self.ax.clear() 

        # Plot limits and appearance
        self.ax.set_xlim(0, self.env.width); self.ax.set_ylim(0, self.env.height)
        self.ax.set_aspect('equal', adjustable='box'); self.ax.grid(True, linestyle='-', color=COLOR_GRID, zorder=0) # Light grey grid
        self.ax.add_patch(patches.Rectangle((0, 0), self.env.width, self.env.height, linewidth=2, edgecolor='black', facecolor='none', zorder=0))
        self._plot_obstacles()

        # Plot goal 
        self.ax.plot(goal[0], goal[1], marker='*', color=COLOR_GOAL, markersize=15, label='Goal', zorder=6, linestyle='None')

        # Plot robot 
        state = robot.get_state(); robot_x, robot_y, robot_theta = state
        self.robot_circle = patches.Circle((robot_x, robot_y), robot.radius, linewidth=1, edgecolor=COLOR_ROBOT, facecolor=COLOR_ROBOT, zorder=10)
        self.ax.add_patch(self.robot_circle)

        # Plot trajectory 
        if self.trajectory:
            trajectory_array = np.array(self.trajectory)
            self.trajectory_line, = self.ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], color=COLOR_PATH, linewidth=2, label='Robot Path', zorder=5)
        else:
             self.trajectory_line, = self.ax.plot([], [], color=COLOR_PATH, linewidth=2, label='Robot Path', zorder=5)

        
        self.sampled_lines.clear()
        # Re-plot current stored sampled trajectories if any (allows update_sampled_trajectories to work with setup_plot)
        if self.current_sampled_trajectories is not None:
             self.update_sampled_trajectories(self.current_sampled_trajectories) # Call helper

        # Time text placeholder
        self.time_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, fontsize=9, va='top', zorder=20)

        # Clear GP map plot handle as axes were cleared
        self.gp_contour_plot = None

        self.ax.set_title('GP-MPPI Navigation') 
        self.ax.set_xlabel('X (m)'); self.ax.set_ylabel('Y (m)')  
        self.ax.legend(loc='upper right')  

    def update_robot_visuals(self, robot):
         """Only update the robot circle visual."""
         if not all([self.robot_circle, self.ax]): return
         state = robot.get_state()
         self.robot_circle.center = state[:2]
         # NO orientation arrow

    def update_path_visuals(self):
         """Update the path line visual based on self.trajectory."""
         if self.trajectory_line is None or not self.trajectory: return
         path_arr = np.array(self.trajectory)
         self.trajectory_line.set_data(path_arr[:, 0], path_arr[:, 1])

    def update_sampled_trajectories(self, trajectories_np, max_rollouts=50, alpha=0.1, color=COLOR_ROLLOUTS, linewidth=0.5, zorder=2):
        """ Clear previous and plot new sampled trajectories (Teal/Cyan). """
        if self.ax is None: return
        self.current_sampled_trajectories = trajectories_np 
        for line in self.sampled_lines: line.remove()
        self.sampled_lines.clear()
        if trajectories_np is None or len(trajectories_np) == 0: return
        num_rollouts = trajectories_np.shape[0]
        num_to_plot = min(num_rollouts, max_rollouts)
        # Prevent error if num_rollouts is less than num_to_plot (although min should handle it)
        if num_rollouts == 0: return
        # Ensure num_to_plot is not greater than available rollouts
        num_to_plot = min(num_to_plot, num_rollouts)
        indices = np.random.choice(num_rollouts, num_to_plot, replace=False)
        for i in indices:
            rollout = trajectories_np[i]
            line, = self.ax.plot(rollout[:, 0], rollout[:, 1], color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)
            self.sampled_lines.append(line)

    def update_time_text(self, text):
        """Update the time/step text."""
        if self.time_text: self.time_text.set_text(text)

    def update_gp_map_display(self, gp_mapper, bounds, grid_res=0.25, cmap=GP_MAP_CMAP, alpha=0.4, zorder=-1):
        """ Plots the GP map mean prediction as a background contour plot. """
        if self.ax is None or gp_mapper is None: return None

        # Remove previous GP map contours if they exist
        if self.gp_contour_plot is not None:
             if hasattr(self.gp_contour_plot, 'collections'): # For contourf
                 for coll in self.gp_contour_plot.collections: coll.remove()
             # Add check for other plot types if needed
             self.gp_contour_plot = None

        # Get map data from mapper
        xx, yy, zz_mean = gp_mapper.get_map_data_for_plotting(bounds, grid_res=grid_res)

        if xx is not None:
            self.gp_contour_plot = self.ax.contourf(xx, yy, zz_mean, levels=np.linspace(0, 1, 11),
                                                    cmap=cmap, alpha=alpha, zorder=zorder, extend='neither') # Add extend='neither'
            return self.gp_contour_plot
        return None

    # Original animation method (without GP map)
    def create_animation(self, robot_states, goal, filename=None, fps=10, all_sampled_trajectories=None, sample_interval=5):
        """ Create animation (GIF) using stored history, styled like target image (NO GP MAP). """
        fig_anim, ax_anim = plt.subplots(figsize=self.figsize)
        ax_anim.set_xlim(0, self.env.width); ax_anim.set_ylim(0, self.env.height)
        ax_anim.set_aspect('equal'); ax_anim.grid(True, linestyle='-', color=COLOR_GRID, zorder=0)
        ax_anim.add_patch(patches.Rectangle((0, 0), self.env.width, self.env.height, lw=2, ec='black', fc='none', zorder=0))
        # Plot obstacles and goal (static elements)
        temp_viz = Visualizer(self.env); temp_viz.ax = ax_anim; temp_viz._plot_obstacles()
        ax_anim.plot(goal[0], goal[1], marker='*', color=COLOR_GOAL, markersize=15, label='Goal', zorder=6, linestyle='None')
        ax_anim.set_title('GP-MPPI Animation')
        ax_anim.set_xlabel('X (m)'); ax_anim.set_ylabel('Y (m)')
        ax_anim.legend(loc='upper right')

        # Initialize dynamic elements
        from robot import TurtleBot # Local import
        temp_robot = TurtleBot(x=robot_states[0][0], y=robot_states[0][1], theta=robot_states[0][2])
        anim_robot_circle = patches.Circle(robot_states[0][:2], temp_robot.radius, color=COLOR_ROBOT, zorder=10)
        ax_anim.add_patch(anim_robot_circle)
        anim_trajectory_line, = ax_anim.plot([], [], color=COLOR_PATH, linewidth=2, zorder=5)
        anim_sampled_lines_handles = []
        anim_time_text = ax_anim.text(0.02, 0.98, '', transform=ax_anim.transAxes, fontsize=9, va='top', zorder=20)

        print(f"Creating animation with {len(robot_states)} frames...")
        def update(frame): # Update function for this animation
            anim_robot_circle.center = robot_states[frame][:2]
            path_arr = np.array(robot_states[:frame+1])
            anim_trajectory_line.set_data(path_arr[:, 0], path_arr[:, 1])
            for line in anim_sampled_lines_handles: line.remove()
            anim_sampled_lines_handles.clear()
            traj_index = frame // sample_interval
            if all_sampled_trajectories is not None and traj_index < len(all_sampled_trajectories):
                current_rollouts = all_sampled_trajectories[traj_index]
                num_to_plot = min(50, current_rollouts.shape[0])
                indices = np.random.choice(current_rollouts.shape[0], num_to_plot, replace=False)
                for i in indices:
                    rollout = current_rollouts[i]
                    line, = ax_anim.plot(rollout[:, 0], rollout[:, 1], color=COLOR_ROLLOUTS, alpha=0.1, linewidth=0.5, zorder=2)
                    anim_sampled_lines_handles.append(line)
            anim_time_text.set_text(f'Step: {frame}/{len(robot_states)-1}\nTime: {frame * 0.1:.1f}s')
            return [anim_robot_circle, anim_trajectory_line, anim_time_text] + anim_sampled_lines_handles

        anim = FuncAnimation(fig_anim, update, frames=len(robot_states), interval=max(1, 1000//fps), blit=False)
        if filename:
            try: anim.save(filename, writer='pillow', fps=fps, dpi=100); print(f"Animation saved to '{filename}'")
            except Exception as e: print(f"Error saving animation: {e}")
        else: print("No filename provided, animation not saved.")
        plt.close(fig_anim); return anim

    # Fixed animation method with GP map
    def create_animation_with_gp(self, robot_states, goal, all_gp_map_data,
                                filename=None, fps=10, all_sampled_trajectories=None,
                                sample_interval=1):
        """ Create animation (GIF) including the evolving GP map. """
        fig_anim, ax_anim = plt.subplots(figsize=self.figsize)
        ax_anim.set_xlim(0, self.env.width); ax_anim.set_ylim(0, self.env.height)
        ax_anim.set_aspect('equal'); ax_anim.grid(True, linestyle='-', color=COLOR_GRID, zorder=0)
        ax_anim.add_patch(patches.Rectangle((0, 0), self.env.width, self.env.height, lw=2, ec='black', fc='none', zorder=0))
        # Plot static obstacles and goal using updated colors
        temp_viz = Visualizer(self.env); temp_viz.ax = ax_anim; temp_viz._plot_obstacles()
        
        # Updated goal marker to star to match static image
        ax_anim.plot(goal[0], goal[1], marker='*', color=COLOR_GOAL, markersize=15, label='Goal', zorder=6, linestyle='None')
        ax_anim.set_title('GP-MPPI Animation')
        ax_anim.set_xlabel('X (m)'); ax_anim.set_ylabel('Y (m)')  # Added units
        ax_anim.legend(loc='upper right')  # Changed to upper right

        # Initialize dynamic elements
        from robot import TurtleBot  # Local import
        temp_robot = TurtleBot(x=robot_states[0][0], y=robot_states[0][1], theta=robot_states[0][2])
        
        # Updated robot color to blue to match static image  
        anim_robot_circle = patches.Circle(robot_states[0][:2], temp_robot.radius, color=COLOR_ROBOT, zorder=10)
        ax_anim.add_patch(anim_robot_circle)
        
        # First frame, add a blue start marker to match static image
        ax_anim.plot(robot_states[0][0], robot_states[0][1], 'o', color=COLOR_ROBOT, 
                    markersize=10, label='Start', zorder=6)
        
        anim_trajectory_line, = ax_anim.plot([], [], color=COLOR_PATH, linewidth=2, label='Robot Path', zorder=5)
        anim_sampled_lines_handles = []
        anim_time_text = ax_anim.text(0.02, 0.98, '', transform=ax_anim.transAxes, fontsize=9, va='top', zorder=20)
        anim_gp_contour = None  # Handle for GP contour plot

        # Ensure all data arrays have the same length by padding if necessary
        num_frames = len(robot_states)
        
        # Defensive check for all_gp_map_data
        if all_gp_map_data is None:
            all_gp_map_data = [None] * num_frames
        elif len(all_gp_map_data) < num_frames:
            print(f"GP Maps recorded before padding: {len(all_gp_map_data)}")
            # Pad with None values to match robot_states length
            all_gp_map_data.extend([None] * (num_frames - len(all_gp_map_data)))
        
        # Defensive check for all_sampled_trajectories
        if all_sampled_trajectories is None:
            all_sampled_trajectories = [None] * num_frames
        elif len(all_sampled_trajectories) < num_frames:
            print(f"Rollouts recorded before padding: {len(all_sampled_trajectories)}")
            # Pad with None values to match robot_states length
            all_sampled_trajectories.extend([None] * (num_frames - len(all_sampled_trajectories)))
        
        print(f"Animation data lengths AFTER padding: poses={len(robot_states)}, "
              f"gp_maps={len(all_gp_map_data)}, rollouts={len(all_sampled_trajectories)}")
        print(f"Creating GP animation with {num_frames} frames...")

        # Animation update function for GP animation
        def update_gp(frame):
            nonlocal anim_sampled_lines_handles, anim_gp_contour  # Modify handles

            # Safety check to avoid index errors
            if frame >= num_frames:
                print(f"Warning: Frame {frame} exceeds available data length {num_frames}")
                frame = num_frames - 1
                
            # Update robot position & path
            anim_robot_circle.center = (robot_states[frame][0], robot_states[frame][1])
            path_arr = np.array(robot_states[:frame+1])
            anim_trajectory_line.set_data(path_arr[:, 0], path_arr[:, 1])

            # Clear and Update sampled trajectories
            for line in anim_sampled_lines_handles:
                line.remove()
            anim_sampled_lines_handles.clear()
            
            # Determine index for rollouts with safety check
            rollout_frame_index = min(frame, len(all_sampled_trajectories) - 1)
            if all_sampled_trajectories is not None and rollout_frame_index >= 0 and all_sampled_trajectories[rollout_frame_index] is not None:
                current_rollouts = all_sampled_trajectories[rollout_frame_index]
                if hasattr(current_rollouts, 'shape') and len(current_rollouts.shape) >= 2 and current_rollouts.shape[0] > 0:  # Additional safety check
                    num_to_plot = min(50, current_rollouts.shape[0])
                    indices = np.random.choice(current_rollouts.shape[0], num_to_plot, replace=False)
                    for i in indices:
                        if i < len(current_rollouts):  # Safety check
                            rollout = current_rollouts[i]  # Should be (T+1, 2)
                            line, = ax_anim.plot(rollout[:, 0], rollout[:, 1], color=COLOR_ROLLOUTS, 
                                             alpha=0.1, linewidth=0.5, zorder=2)
                            anim_sampled_lines_handles.append(line)

            # --- Clear and Update GP Map Display ---
            if anim_gp_contour is not None:  # Clear previous contour
                try:
                    for collection in anim_gp_contour.collections:
                        collection.remove()
                except (AttributeError, TypeError):
                    # If collections attribute doesn't exist or causes other issues
                    # Try the alternative way to remove
                    try:
                        anim_gp_contour.remove()
                    except Exception as e:
                        print(f"Warning: Could not clear previous contour: {e}")
                        # Just ignore and clear the reference
                        pass
                anim_gp_contour = None

            # Determine index for GP map data with safety check
            gp_frame_index = min(frame, len(all_gp_map_data) - 1)
            if gp_frame_index >= 0 and all_gp_map_data[gp_frame_index] is not None:
                gp_data = all_gp_map_data[gp_frame_index]
                # Use same parameters as update_gp_map_display
                if 'xx' in gp_data and 'yy' in gp_data and 'zz_mean' in gp_data:
                    try:
                        anim_gp_contour = ax_anim.contourf(gp_data['xx'], gp_data['yy'], gp_data['zz_mean'],
                                                         levels=np.linspace(0, 1, 11), cmap=GP_MAP_CMAP, 
                                                         alpha=0.4, zorder=-1, extend='neither')
                    except Exception as e:
                        print(f"Error plotting GP contour at frame {frame}: {e}")

            # Update time text
            anim_time_text.set_text(f'Step: {frame}/{num_frames-1}\nTime: {frame * 0.1:.1f}s')

            # Return all artists that could possibly change
            artists = [anim_robot_circle, anim_trajectory_line, anim_time_text] + anim_sampled_lines_handles
            
            # Fixed: Handle contour plot artist correctly
            if anim_gp_contour is not None:
                # In matplotlib, contourf returns a QuadContourSet, which itself should be included
                # as an artist, not its collections attribute
                artists.append(anim_gp_contour)
                
            return artists

        # Create animation object
        try:
            anim = FuncAnimation(fig_anim, update_gp, frames=num_frames, interval=max(1, 1000//fps), blit=True)
            
            # Save animation
            if filename:
                try:
                    # Use PillowWriter explicitly for GIF
                    from matplotlib.animation import PillowWriter
                    writer = PillowWriter(fps=fps)
                    anim.save(filename, writer=writer, dpi=100)
                    print(f"Animation saved to '{filename}'")
                except Exception as e:
                    print(f"Error saving animation: {e}")
                    
                    import traceback
                    traceback.print_exc()
            else:
                print("No filename provided, animation not saved.")
                
            plt.close(fig_anim)
            return anim
            
        except Exception as e:
            print(f"Error creating animation: {e}")
            import traceback
            traceback.print_exc()
            plt.close(fig_anim)
            return None