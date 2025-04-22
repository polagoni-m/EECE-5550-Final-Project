# gp_mapper.py
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

class GPMapper:
    def __init__(self, kernel=None, max_points=750, subsample_factor=0.2, noise_level=0.05):
        """
        Initializes the Gaussian Process Mapper.

        Args:
            kernel: Custom scikit-learn GP kernel. If None, a default RBF kernel is used.
            max_points (int): Maximum number of training points to keep (limits memory/computation).
            subsample_factor (float): Fraction of new points to randomly keep per update (0 to 1).
            noise_level (float): Assumed noise level for the WhiteKernel part of the GP.
        """
        if kernel is None:
            # Default Kernel: RBF for spatial correlation + WhiteKernel for observation noise
            # Length scale needs tuning (start around expected obstacle size or free space patch size)
            kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(0.2, 5.0)) \
                     + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-5, 1e-1))

        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, # Small regularization
                                           n_restarts_optimizer=0, 
                                           normalize_y=False) 
        self.train_X = None # Stores (N, 2) training locations [[x1,y1], [x2,y2]...]
        self.train_y = None # Stores (N,) training occupancy values [o1, o2...] (0=free, 1=occupied)
        self.max_points = max_points
        self.subsample_factor = subsample_factor
        self.scaler = StandardScaler() 
        self.fitted = False # Flag to track if GP has been fitted at least once
        print(f"GPMapper Initialized. Max points={max_points}, Subsample={subsample_factor}")

    def update(self, new_points_list):
        """
        Updates the GP with new sensor data.

        Args:
            new_points_list: List of tuples [(x, y, occupancy), ...]
                             where occupancy is typically 0.0 (free) or 1.0 (occupied).
        """
        if not new_points_list:
            return

        new_X = np.array([[p[0], p[1]] for p in new_points_list])
        new_y = np.array([p[2] for p in new_points_list])

        # Optional: Subsample new points before adding
        if 0 < self.subsample_factor < 1.0 and len(new_X) > 10:
             num_to_keep = max(1, int(len(new_X) * self.subsample_factor))
             indices = np.random.choice(len(new_X), num_to_keep, replace=False)
             new_X = new_X[indices]
             new_y = new_y[indices]

        # Add new data to training set
        if self.train_X is None:
            self.train_X = new_X
            self.train_y = new_y
        else:
            self.train_X = np.vstack((self.train_X, new_X))
            self.train_y = np.hstack((self.train_y, new_y))

        # Limit total number of points
        if len(self.train_X) > self.max_points:
            overflow = len(self.train_X) - self.max_points
            self.train_X = self.train_X[overflow:]
            self.train_y = self.train_y[overflow:]

        if len(self.train_X) < 10: # Need a minimum number of points to fit
             print(f"GP waiting for more data ({len(self.train_X)} points)...")
             self.fitted = False
             return

       
        with warnings.catch_warnings():
             warnings.simplefilter("ignore", category=ConvergenceWarning)
             try:
                 X_scaled = self.scaler.fit_transform(self.train_X) # Fit scaler on current data
                 self.gp.fit(X_scaled, self.train_y)
                 self.fitted = True
                
             except Exception as e:
                 print(f"!!! GP Fit Error: {e}. Data size: {self.train_X.shape if self.train_X is not None else 'None'}")
                 
                 self.fitted = False


    def predict(self, query_points_np):
        """
        Predicts occupancy mean and std dev at query points.

        Args:
            query_points_np: (M, 2) numpy array of (x, y) coordinates.

        Returns:
            tuple: (mean, std_dev) - both (M,) numpy arrays.
                   Mean is clipped to [0, 1]. Returns default (0.5, high_std) if not fitted.
        """
        default_std = 1.0 # Default uncertainty standard deviation
        if not self.fitted or query_points_np.shape[0] == 0:
            return np.full(len(query_points_np), 0.5), np.full(len(query_points_np), default_std)

        query_points_np = np.atleast_2d(query_points_np)
        try:
            # Scale query points using the fitted scaler
            query_scaled = self.scaler.transform(query_points_np)
            mean, std_dev = self.gp.predict(query_scaled, return_std=True)

            mean = np.clip(mean, 0.0, 1.0) # Ensure probability bounds
            std_dev = np.maximum(std_dev, 1e-6) # Ensure non-negative std dev

            return mean, std_dev
        except Exception as e:
            print(f"!!! GP Predict Error: {e}. Query shape: {query_points_np.shape}")
            return np.full(len(query_points_np), 0.5), np.full(len(query_points_np), default_std)

    def get_map_data_for_plotting(self, bounds, grid_res=0.25):
        """ Generates map data (mean prediction) over a grid for visualization. """
        if not self.fitted: return None, None, None

        x_min, x_max, y_min, y_max = bounds
        xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_res),
                             np.arange(y_min, y_max, grid_res))
        grid_points = np.vstack([xx.ravel(), yy.ravel()]).T

        if grid_points.shape[0] == 0: return None, None, None

        # Use self.predict which handles scaling and error checking
        zz_mean, _ = self.predict(grid_points) 
        zz_mean = zz_mean.reshape(xx.shape)

        return xx, yy, zz_mean