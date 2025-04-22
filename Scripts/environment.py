# environment.py
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.patches import Circle, Rectangle

COLOR_OBSTACLE_CIRCLE = 'grey'
COLOR_OBSTACLE_RECT = 'black' 

class Environment:
    """
    Environment with obstacles for TurtleBot navigation.
    """
    def __init__(self, width=10.0, height=10.0):
        """
        Initialize environment.

        Args:
            width: Environment width
            height: Environment height
        """
        self.width = width
        self.height = height
        self.obstacles = [] 

    def add_obstacle(self, x, y, radius):
        """ Add circular obstacle. """
        self.obstacles.append({'x': x, 'y': y, 'radius': radius, 'type': 'circle'})

    def add_rectangular_obstacle(self, x, y, width, height):
        """ Add rectangular obstacle. """
        self.obstacles.append({'x': x, 'y': y, 'width': width, 'height': height, 'type': 'rectangle'})

    def check_collision(self, x, y, robot_radius):
        """ Check if position collides with any obstacle or boundary. """
        # Boundary collision
        if x - robot_radius < 0 or x + robot_radius > self.width or \
           y - robot_radius < 0 or y + robot_radius > self.height:
            return True
        # Obstacle collision
        for obs in self.obstacles:
            if 'type' in obs and obs['type'] == 'rectangle':
                rect_x, rect_y = obs['x'], obs['y']
                rect_w, rect_h = obs['width'], obs['height']
                closest_x = max(rect_x, min(x, rect_x + rect_w))
                closest_y = max(rect_y, min(y, rect_y + rect_h))
                dist_x = x - closest_x; dist_y = y - closest_y
                if (dist_x**2 + dist_y**2) <= robot_radius**2: return True
            else: # Circle
                if ((x - obs['x'])**2 + (y - obs['y'])**2) <= (robot_radius + obs['radius'])**2: return True
        return False

    def distance_to_obstacles(self, x, y):
        """ Calculate distance to nearest obstacle edge/boundary. """
        min_dist = float('inf')
        min_dist = min(min_dist, x, self.width - x, y, self.height - y)
        for obs in self.obstacles:
            if 'type' in obs and obs['type'] == 'rectangle':
                rect_x, rect_y = obs['x'], obs['y']
                rect_w, rect_h = obs['width'], obs['height']
                closest_x = max(rect_x, min(x, rect_x + rect_w))
                closest_y = max(rect_y, min(y, rect_y + rect_h))
                dist = np.sqrt((x - closest_x)**2 + (y - closest_y)**2)
            else: # Circle
                dist = np.sqrt((x - obs['x'])**2 + (y - obs['y'])**2) - obs['radius']
            min_dist = min(min_dist, max(0, dist))
        return min_dist

    # --- Environment Creation Functions as class methods ---
    @classmethod
    def create_simple_environment(cls):
        env = cls(width=10.0, height=10.0)
        env.add_obstacle(3.0, 3.0, 0.5)
        env.add_obstacle(5.0, 7.0, 0.7)
        env.add_obstacle(7.0, 4.0, 0.6)
        env.add_rectangular_obstacle(1.0, 5.0, 1.0, 3.0)
        return env

    @classmethod
    def create_moderate_environment(cls):
        """ Creates an environment with more obstacles and tighter passages. """
        env = cls(width=10.0, height=10.0)
        env.add_rectangular_obstacle(4.0, 4.0, 2.0, 2.0) # Central block
        env.add_obstacle(2.0, 6.0, 0.6); env.add_obstacle(8.0, 4.0, 0.6) # Offset pillars
        env.add_obstacle(4.0, 8.0, 0.5); env.add_obstacle(6.0, 2.0, 0.5)
        env.add_rectangular_obstacle(1.0, 1.5, 2.0, 0.5) # Diagonal barriers
        env.add_rectangular_obstacle(7.0, 7.5, 2.0, 0.5)
        env.add_obstacle(1.0, 8.5, 0.4); env.add_obstacle(8.5, 1.0, 0.4) # Corner clutter
        return env

    @classmethod
    def create_complex_cluttered_environment(cls):
        """ Creates a densely cluttered environment with narrow gaps. """
        env = cls(width=10.0, height=10.0)
        obstacles_params = [
            ('c', 1.5, 1.5, 0.5), ('c', 3.0, 0.8, 0.4), ('r', 4.0, 1.0, 1.5, 0.5),
            ('c', 0.8, 3.5, 0.6), ('r', 2.0, 2.5, 0.5, 2.0), ('c', 3.8, 4.0, 0.5),
            ('c', 5.5, 2.5, 0.7), ('r', 6.5, 0.5, 0.5, 2.5), ('c', 8.0, 2.0, 0.6),
            ('c', 9.2, 4.0, 0.5), ('r', 8.5, 5.0, 1.0, 1.5), ('c', 7.0, 5.5, 0.6),
            ('r', 5.0, 4.5, 1.5, 0.5), ('c', 3.0, 6.0, 0.7), ('r', 0.5, 6.5, 2.0, 0.5),
            ('c', 1.5, 8.5, 0.6), ('c', 4.0, 7.5, 0.5), ('r', 5.5, 8.0, 1.0, 1.5),
            ('c', 7.5, 9.0, 0.6), ('c', 9.0, 7.0, 0.5) ]
        for params in obstacles_params:
            type = params[0]
            if type == 'c': env.add_obstacle(params[1], params[2], params[3])
            elif type == 'r': env.add_rectangular_obstacle(params[1], params[2], params[3], params[4])
        return env