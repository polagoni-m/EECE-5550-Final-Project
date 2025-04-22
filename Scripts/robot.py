import numpy as np

class TurtleBot:
    """
    TurtleBot model with differential drive dynamics.
    """
    def __init__(self, x=0.0, y=0.0, theta=0.0, max_linear_velocity=0.5, max_angular_velocity=1.0):
        """
        Initialize TurtleBot.
        
        Args:
            x: Initial x position
            y: Initial y position
            theta: Initial orientation (radians)
            max_linear_velocity: Maximum linear velocity
            max_angular_velocity: Maximum angular velocity
        """
        self.state = np.array([x, y, theta])  # [x, y, theta]
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        
        # Robot dimensions
        self.radius = 0.1  # Robot radius in meters
    
    def dynamics(self, state, control, dt):
        """
        Differential drive dynamics model for TurtleBot.
        
        Args:
            state: Current state [x, y, theta]
            control: Control input [v, omega] (linear and angular velocity)
            dt: Time step
            
        Returns:
            next_state: Next state after applying control
        """
        x, y, theta = state
        v, omega = control
        
        # Clip control inputs to respect limits
        v = np.clip(v, -self.max_linear_velocity, self.max_linear_velocity)
        omega = np.clip(omega, -self.max_angular_velocity, self.max_angular_velocity)
        
        # Differential drive dynamics
        x_next = x + v * np.cos(theta) * dt
        y_next = y + v * np.sin(theta) * dt
        theta_next = theta + omega * dt
        
        # Normalize angle to [-pi, pi]
        theta_next = ((theta_next + np.pi) % (2 * np.pi)) - np.pi
        
        return np.array([x_next, y_next, theta_next])
    
    def update(self, control, dt):
        """
        Update robot state based on control input.
        
        Args:
            control: Control input [v, omega]
            dt: Time step
        """
        self.state = self.dynamics(self.state, control, dt)
        
    def get_state(self):
        """
        Get current robot state.
        
        Returns:
            state: Current state [x, y, theta]
        """
        return self.state
    
    def set_state(self, state):
        """
        Set robot state.
        
        Args:
            state: New state [x, y, theta]
        """
        self.state = state