# sensor.py
import numpy as np

# Assumes environment defines the Environment class
# from environment import Environment

class LidarSensor:
    def __init__(self, num_beams=30, max_range=8.0, angle_span=np.pi * 1.5, noise_stddev=0.05):
        """
        Initializes a simulated Lidar sensor.

        Args:
            num_beams (int): Number of scan lines.
            max_range (float): Maximum sensing distance.
            angle_span (float): Total angular width of the scan (radians).
            noise_stddev (float): Standard deviation of Gaussian noise added to ranges.
        """
        self.num_beams = num_beams
        self.max_range = max_range
        self.angle_min = -angle_span / 2
        self.angle_max = angle_span / 2
        self.angles = np.linspace(self.angle_min, self.angle_max, num_beams)
        self.noise_stddev = noise_stddev
        # How many points to mark as free space along each beam before a hit
        self.free_points_per_beam = 3

    def get_scan_data_for_gp(self, robot_pose, environment):
        """
        Simulates a Lidar scan and returns points formatted for GP training.

        Args:
            robot_pose (np.array): Current robot pose [x, y, theta].
            environment (Environment): The *true* environment object (from environment.py)
                                      used for ray casting ground truth.

        Returns:
            list: A list of tuples [(x, y, occupancy), ...], where occupancy is 0.0 or 1.0.
        """
        x_robot, y_robot, theta_robot = robot_pose
        gp_points = []
        scan_endpoints = [] # Optional: For plotting raw scan later

        for angle_offset in self.angles:
            scan_angle = theta_robot + angle_offset
            cos_a, sin_a = np.cos(scan_angle), np.sin(scan_angle)

            # --- Simple Ray Casting ---
            # Check points along the ray incrementally
            hit_dist = self.max_range
            hit_detected = False
            resolution = 0.05 
            for d in np.arange(resolution, self.max_range + resolution, resolution):
                check_x = x_robot + d * cos_a
                check_y = y_robot + d * sin_a
                
                if environment.check_collision(check_x, check_y, robot_radius=0.0):
                    hit_dist = d
                    hit_detected = True
                    break

            
            if hit_detected:
                 hit_dist = max(0, hit_dist + np.random.normal(0, self.noise_stddev))
                 hit_dist = min(hit_dist, self.max_range) 

            # Determine the actual measured range for this beam
            measured_range = hit_dist if hit_detected else self.max_range
            endpoint_x = x_robot + measured_range * cos_a
            endpoint_y = y_robot + measured_range * sin_a
            scan_endpoints.append((endpoint_x, endpoint_y))

            # --- Generate points for GP training ---
            # 1. Free space points along the beam (before hit or max_range)
            for i in range(1, self.free_points_per_beam + 1):
                free_d = measured_range * (i / (self.free_points_per_beam + 1.0))
                # Only add points substantially before the endpoint, prevents overlap
                if free_d < measured_range - resolution:
                    free_x = x_robot + free_d * cos_a
                    free_y = y_robot + free_d * sin_a
                    gp_points.append((free_x, free_y, 0.0)) # 0.0 for free space

            # 2. Hit point (if an obstacle was hit within max_range)
            if hit_detected and hit_dist < self.max_range:
                 # Add point slightly *before* the actual hit surface for stability? Optional.
                 # hit_d_gp = max(0, hit_dist - resolution)
                 # hit_x = x_robot + hit_d_gp * cos_a
                 # hit_y = y_robot + hit_d_gp * sin_a
                 gp_points.append((endpoint_x, endpoint_y, 1.0)) # 1.0 for occupied

        # return gp_points, scan_endpoints # Optionally return raw scan too
        return gp_points