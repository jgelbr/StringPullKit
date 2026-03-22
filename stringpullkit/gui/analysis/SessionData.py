import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

class SessionData:
    """
    Represents a single DLC-processed string pull session from one mouse.
    Holds DLC coordinate data, derived kinematics and computed metrics.
    """

    def __init__(self, video_path=None, dlc_paths=None, save_dir=None, fps=120, session_id=None, total_frames=0, likelihood_threshold=0.6,
                 smoothing_window=25, smoothing_poly=2, scale_factor=None, height=None, show_plot=False):
        """
        Parameters
        ----------
        video_path: str
            Path to raw video file
        dlc_paths: dict
            Dictionary of DLC csv paths ({'hands': 'hands.csv', 'nose': 'nose.csv', 'ears': 'ears.csv', ...})
        save_dir: str, optional
            Directory for saving outputs (defaults to same as video)
        fps: int
            Video frames per second
        likelihood_threshold: float
            Minimum DLC confidence to use a point in calculations
        smoothing_window: float
            Smoothing parameter for coordinate signals (Savitzky–Golay)
        smoothing_poly: float
            Polynomial order for Savitzky–Golay smoothing
        scale_factor: float
            Spatial scaling (if the video was scaled in the GUI)
        height: float
            Video height (px)
        show_plot: bool
            Whether to show plots during analysis
        """
        self.video_path = video_path
        self.dlc_paths = dlc_paths
        self.save_dir = save_dir if save_dir is not None else os.path.dirname(video_path)
        self.fps = fps
        self.session_id = session_id
        self.total_frames = total_frames
        self.likelihood_threshold = likelihood_threshold
        self.smoothing_window = smoothing_window
        self.smoothing_poly = smoothing_poly
        self.scale_factor = scale_factor
        self.height = height
        self.show_plot = show_plot

        self.data = {}
        self.coords = {}
        self.likelihoods = {}
        self.metrics = {}
        self.phase_metrics = {}
        

    # -------------------------------- #
    # Data Loading and Cleaning
    # -------------------------------- #

    def load_data(self):
        """Load all DLC csvs into pandas DataFrames."""
        for key, path in self.dlc_paths.items():
            df = pd.read_csv(path, header=[1, 2])  # Skip first row, use second and third as header
            df.columns = ['_'.join(col).strip() for col in df.columns.values]  # Flatten the multi-index
            self.data[key] = df
        print(f"DLC data loaded for: {list(self.data.keys())}")

    def clean_data(self):
        """
        Filter by likelihood, smooth and scale coordinates.
        Populates self.coords[key][bodypart] = np.array([[x, y], ...]) with un-smooth coordinates
        Smoothed coordinates are stored in self.metrics as '{bodypart}_x_trajectory' and '{bodypart}_y_trajectory and '{bodypart}_likelihood'

        """
        for key, df in self.data.items():
            coords = {}
            likelihoods = {}

            # Extract base bodypart names (remove '_x', '_y', '_likelihood')
            bodyparts = sorted([col.rsplit('_', 1)[0] for col in df.columns if col.endswith('_x')])

            for bp in bodyparts:
                x_col, y_col, likelihood_col = f"{bp}_x", f"{bp}_y", f"{bp}_likelihood"

                # Extract coordinates
                x = df[x_col].values
                y = df[y_col].values
                lik = df[likelihood_col].values

                # Flip y-axis first (since DLC y=0 is top)
                if self.height is not None:
                    y = self.height - y
              
                # Apply scaling (same factor used for display in GUI)
                if self.scale_factor is not None:
                    x *= self.scale_factor
                    y *= self.scale_factor
                
                coords[bp] = np.vstack((x, y)).T
                likelihoods[bp] = lik 

                # Filter by likelihood, interpolate missing data and smooth
               # x[lik < self.likelihood_threshold] = np.nan
               # y[lik < self.likelihood_threshold] = np.nan

                # if np.any(np.isnan(x)): 
                #     x = pd.Series(x).interpolate(limit_direction='both').to_numpy()                   
                # if np.any(np.isnan(y)):
                #     y = pd.Series(y).interpolate(limit_direction='both').to_numpy()

                try:
                    x_smooth = savgol_filter(x, self.smoothing_window, self.smoothing_poly)
                    y_smooth = savgol_filter(y, self.smoothing_window, self.smoothing_poly)
                    self.metrics[f"{bp.lower()}_x_trajectory"] = x_smooth
                    self.metrics[f"{bp.lower()}_y_trajectory"] = y_smooth       
                    self.metrics[f"{bp.lower()}_likelihood"] = lik
                except ValueError:
                     # Skip smoothing if not enough points
                    self.metrics[f"{bp.lower()}_x_trajectory"] = x
                    self.metrics[f"{bp.lower()}_y_trajectory"] = y
                    self.metrics[f"{bp.lower()}_likelihood"] = lik  

            self.coords[key] = coords
            self.coords[f"{key}_likelihood"] = likelihoods

    def __getitem__(self, key):
        """Allow session['hand_l'] or session['hand_l_x'] access."""
        part = key.lower()
        axis = None
        if part.endswith('_x'):
            axis = 0
            part = part[:-2]
        elif part.endswith('_y'):
            axis = 1
            part = part[:-2]

        for g, parts in self.coords.items():
            if part in parts:
                data = self.coords[g][part]
                return data[:, axis] if axis is not None else data
        raise KeyError(f"{key} not found.")

