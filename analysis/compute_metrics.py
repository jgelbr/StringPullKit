#import utils
import numpy as np
from stringpullkit.analysis import utils

# ===================================== #
# Validation Functions
# ===================================== #

def validate_trajectory(trajectory, name):
    """Validate trajectory data before processing."""
    if trajectory is None or len(trajectory) == 0:
        print(f"WARNING: {name} is empty or None")
        return False
    if np.all(np.isnan(trajectory)):
        print(f"WARNING: {name} contains only NaN values")
        return False
    return True

def validate_phase_ranges(ranges, trajectory_length, phase_name):
    """Validate that phase ranges are within bounds."""
    valid_ranges = []
    for start, end in ranges:
        if start < 0 or end > trajectory_length:
            print(f"WARNING: {phase_name} range ({start}, {end}) out of bounds for trajectory length {trajectory_length}")
            continue
        if start >= end:
            print(f"WARNING: {phase_name} range ({start}, {end}) invalid (start >= end)")
            continue
        valid_ranges.append((start, end))
    return valid_ranges

def safe_compute_stats(data, metric_name):
    """Safely compute mean and CV with validation."""
    data = np.asarray(data)
    if len(data) == 0 or np.all(np.isnan(data)):
        print(f"WARNING: Cannot compute statistics for {metric_name} - no valid data")
        return np.nan, np.nan
    return np.nanmean(data), utils.coefficient_of_variation(data)

# ===================================== #
# Likelihood Masking Functions
# ===================================== #

def apply_likelihood_mask(data, likelihood, threshold):
    """
    Apply likelihood threshold to data, setting low-confidence values to NaN.
    
    Args:
        data: numpy array of values
        likelihood: numpy array of likelihood values (0-1)
        threshold: minimum likelihood threshold
    
    Returns:
        Masked data array with low-confidence values set to NaN
    """
    if likelihood is None:
        return data
    
    data_masked = data.copy()
    mask = likelihood < threshold
    data_masked[mask] = np.nan
    return data_masked

def apply_likelihood_mask_2d(x, y, likelihood, threshold):
    """
    Apply likelihood mask to paired x,y trajectories.
    
    Returns:
        Tuple of (x_masked, y_masked)
    """
    x_masked = apply_likelihood_mask(x, likelihood, threshold)
    y_masked = apply_likelihood_mask(y, likelihood, threshold)
    return x_masked, y_masked

def get_masked_trajectories(session, bodypart):
    """
    Get x, y trajectories with likelihood masking applied.
    
    Args:
        session: SessionData object
        bodypart: name of bodypart (e.g., 'hand_l', 'nose')
    
    Returns:
        Tuple of (x_masked, y_masked, likelihood)
    """
    x = session.metrics[f'{bodypart}_x_trajectory']
    y = session.metrics[f'{bodypart}_y_trajectory']
    likelihood = session.metrics[f'{bodypart}_likelihood']
    
    x_masked, y_masked = apply_likelihood_mask_2d(x, y, likelihood, session.likelihood_threshold)
    
    return x_masked, y_masked, likelihood

def store_unmasked_for_plotting(session, metric_name, masked_data, unmasked_data):
    """
    Store both masked (for analysis) and unmasked (for plotting) versions of a metric.
    
    Args:
        session: SessionData object
        metric_name: name of the metric
        masked_data: masked version (with NaNs)
        unmasked_data: unmasked version (for visualization)
    """
    session.metrics[metric_name] = masked_data
    session.metrics[f'{metric_name}_unmasked'] = unmasked_data

# ===================================== #
# Helper Functions
# ===================================== #

def stack(session, key):
    """Stack x and y trajectories into Nx2 array."""
    x = session.metrics[f'{key}_x_trajectory']
    y = session.metrics[f'{key}_y_trajectory']
    return np.vstack((x, y)).T

def stack_masked(session, key):
    """Stack masked x and y trajectories into Nx2 array."""
    x_masked, y_masked, _ = get_masked_trajectories(session, key)
    return np.vstack((x_masked, y_masked)).T

def get_for_plotting(session, metric_name):
    """
    Get metric for plotting - returns unmasked version if available, otherwise masked.
    Use this in your plotting code to automatically get the right version.
    
    Example:
        yaw = get_for_plotting(session, 'head_yaw')  # Returns head_yaw_unmasked if it exists
    """
    unmasked_key = f'{metric_name}_unmasked'
    if unmasked_key in session.metrics:
        return session.metrics[unmasked_key]
    return session.metrics.get(metric_name)

def safe_concatenate_lists(list1, list2):
    """Safely concatenate two lists, converting to arrays if needed."""
    arr1 = np.asarray(list1) if not isinstance(list1, np.ndarray) else list1
    arr2 = np.asarray(list2) if not isinstance(list2, np.ndarray) else list2
    return np.concatenate([arr1, arr2])

# ===================================== #
# Hand Metrics
# ===================================== #

def compute_hand_metrics(session):
    """Compute hand-based metrics for the session"""
    print("Computing hand-based metrics...")

    hands = ['l', 'r']
    
    for hand in hands:
        full_hand = 'left' if hand == 'l' else 'right'

        # Get masked trajectories for computation
        x_masked, y_masked, likelihood = get_masked_trajectories(session, f'hand_{hand}')
        
        # Get unmasked trajectories  
        x_unmasked = session.metrics[f'hand_{hand}_x_trajectory']
        y_unmasked = session.metrics[f'hand_{hand}_y_trajectory']

        # Validate trajectories
        if not validate_trajectory(y_masked, f'hand_{hand}_y') or \
           not validate_trajectory(x_masked, f'hand_{hand}_x'):
            continue

        # Velocity, Acceleration, Jerk (compute from both masked and unmasked)
        vel_masked = utils.compute_velocity(y_masked, session.fps)
        vel_unmasked = utils.compute_velocity(y_unmasked, session.fps)
        store_unmasked_for_plotting(session, f'hand_{hand}_velocity', vel_masked, vel_unmasked)
        
        acc_masked = utils.compute_acceleration(vel_masked, session.fps)
        acc_unmasked = utils.compute_acceleration(vel_unmasked, session.fps)
        store_unmasked_for_plotting(session, f'hand_{hand}_acceleration', acc_masked, acc_unmasked)
        
        jerk_masked = utils.compute_jerk(acc_masked, session.fps)
        jerk_unmasked = utils.compute_jerk(acc_unmasked, session.fps)
        store_unmasked_for_plotting(session, f'hand_{hand}_jerk', jerk_masked, jerk_unmasked)

        # Phase Frame Ranges (use unmasked for phase detection)
        session.phase_metrics.setdefault(f'{full_hand}_withdraw', {})
        session.phase_metrics.setdefault(f'{full_hand}_reach', {})

        phase_data = utils.detect_cycles_and_phases(y_unmasked, session.fps, session.scale_factor)

        withdraw_ranges = phase_data['withdraw_ranges']
        reach_ranges = phase_data['reach_ranges']
        withdraw_durations = phase_data['withdraw_durations']
        reach_durations = phase_data['reach_durations']
        cycle_durations = phase_data['cycle_durations']
        cycles = phase_data['cycles']
        
        # Validate phase ranges
        withdraw_ranges = validate_phase_ranges(withdraw_ranges, len(y_masked), f'{full_hand}_withdraw')
        reach_ranges = validate_phase_ranges(reach_ranges, len(y_masked), f'{full_hand}_reach')
        
        session.phase_metrics[f'{full_hand}_withdraw']['ranges'] = withdraw_ranges
        session.phase_metrics[f'{full_hand}_reach']['ranges'] = reach_ranges

        # Durations & Count
        session.phase_metrics[f'{full_hand}_withdraw']['duration'] = withdraw_durations
        session.phase_metrics[f'{full_hand}_reach']['duration'] = reach_durations
        session.metrics[f'hand_{hand}_cycle_duration'] = cycle_durations
        session.metrics[f'hand_{hand}_cycle_count'] = len(cycle_durations)
        session.metrics[f'hand_{hand}_cycles'] = cycles
        session.metrics[f'hand_{hand}_withdraw_count'] = len(withdraw_durations)
        session.metrics[f'hand_{hand}_reach_count'] = len(reach_durations)
        
        # Duration statistics
        session.metrics[f'hand_{hand}_cycle_duration_mean'], \
        session.metrics[f'hand_{hand}_cycle_duration_cv'] = safe_compute_stats(cycle_durations, f'hand_{hand}_cycle_duration')
        
        session.metrics[f'hand_{hand}_withdraw_duration_mean'], \
        session.metrics[f'hand_{hand}_withdraw_duration_cv'] = safe_compute_stats(withdraw_durations, f'hand_{hand}_withdraw_duration')
        
        session.metrics[f'hand_{hand}_reach_duration_mean'], \
        session.metrics[f'hand_{hand}_reach_duration_cv'] = safe_compute_stats(reach_durations, f'hand_{hand}_reach_duration')

        # Proportions
        withdraw_proportion_mean, withdraw_proportion_cv = utils.compute_phase_proportion_stats(
            withdraw_durations, cycle_durations)
        reach_proportion_mean, reach_proportion_cv = utils.compute_phase_proportion_stats(
            reach_durations, cycle_durations)
        session.metrics[f'hand_{hand}_withdraw_proportion_mean'] = withdraw_proportion_mean
        session.metrics[f'hand_{hand}_withdraw_proportion_cv'] = withdraw_proportion_cv
        session.metrics[f'hand_{hand}_reach_proportion_mean'] = reach_proportion_mean
        session.metrics[f'hand_{hand}_reach_proportion_cv'] = reach_proportion_cv

        # Withdraw/Reach Ratio
        if len(withdraw_durations) > 0 and len(reach_durations) > 0:
            min_len = min(len(withdraw_durations), len(reach_durations))
            ratios = np.array(withdraw_durations[:min_len]) / np.array(reach_durations[:min_len])
            session.metrics[f'hand_{hand}_withdraw_reach_ratio_mean'] = np.nanmean(ratios)
            session.metrics[f'hand_{hand}_withdraw_reach_ratio_cv'] = utils.coefficient_of_variation(ratios)
        else:
            session.metrics[f'hand_{hand}_withdraw_reach_ratio_mean'] = np.nan
            session.metrics[f'hand_{hand}_withdraw_reach_ratio_cv'] = np.nan

        # Speed & Phasewise Speed 
        session.metrics[f'hand_{hand}_speed'] = utils.compute_speed(x_masked, y_masked, session.fps)

        withdraw_mean_speed, withdraw_peak_speed = utils.compute_phasewise_speed(
            session.metrics[f'hand_{hand}_speed'], withdraw_ranges)
        reach_mean_speed, reach_peak_speed = utils.compute_phasewise_speed(
            session.metrics[f'hand_{hand}_speed'], reach_ranges)

        session.phase_metrics[f'{full_hand}_withdraw']['mean_speed'] = withdraw_mean_speed
        session.phase_metrics[f'{full_hand}_reach']['mean_speed'] = reach_mean_speed
        session.phase_metrics[f'{full_hand}_withdraw']['peak_speed'] = withdraw_peak_speed
        session.phase_metrics[f'{full_hand}_reach']['peak_speed'] = reach_peak_speed

        # Handle empty arrays gracefully
        if len(withdraw_mean_speed) == 0:
            print(f"WARNING: No valid withdraw phases for hand_{hand}")
            withdraw_mean_speed = np.array([np.nan])
            withdraw_peak_speed = np.array([np.nan])
        
        if len(reach_mean_speed) == 0:
            print(f"WARNING: No valid reach phases for hand_{hand}")
            reach_mean_speed = np.array([np.nan])
            reach_peak_speed = np.array([np.nan])

        session.metrics[f'hand_{hand}_withdraw_speed_mean'], \
        session.metrics[f'hand_{hand}_withdraw_speed_cv'] = safe_compute_stats(withdraw_mean_speed, f'hand_{hand}_withdraw_speed')
        
        session.metrics[f'hand_{hand}_withdraw_speed_peak_mean'], \
        session.metrics[f'hand_{hand}_withdraw_speed_peak_cv'] = safe_compute_stats(withdraw_peak_speed, f'hand_{hand}_withdraw_speed_peak')
        
        session.metrics[f'hand_{hand}_reach_speed_mean'], \
        session.metrics[f'hand_{hand}_reach_speed_cv'] = safe_compute_stats(reach_mean_speed, f'hand_{hand}_reach_speed')
        
        session.metrics[f'hand_{hand}_reach_speed_peak_mean'], \
        session.metrics[f'hand_{hand}_reach_speed_peak_cv'] = safe_compute_stats(reach_peak_speed, f'hand_{hand}_reach_speed_peak')

        # Displacement & Phasewise Displacement (use masked data for computation)
        disp_masked = utils.compute_displacement(y_masked)
        disp_unmasked = utils.compute_displacement(y_unmasked)
        store_unmasked_for_plotting(session, f'hand_{hand}_displacement', disp_masked, disp_unmasked)

        withdraw_dx, withdraw_dy, withdraw_euclidean = utils.compute_phasewise_displacement(
            x_masked, y_masked, withdraw_ranges)
        reach_dx, reach_dy, reach_euclidean = utils.compute_phasewise_displacement(
            x_masked, y_masked, reach_ranges)
        
        session.phase_metrics[f'{full_hand}_withdraw']['displacement'] = {
            'dx': withdraw_dx,
            'dy': withdraw_dy,
            'euclidean': withdraw_euclidean
        }
        session.phase_metrics[f'{full_hand}_reach']['displacement'] = {
            'dx': reach_dx,
            'dy': reach_dy,
            'euclidean': reach_euclidean
        }

        # Store as arrays for easier access
        for displacement in ['dx', 'dy', 'euclidean']:
            session.metrics[f'hand_{hand}_withdraw_{displacement}_displacement'] = np.array(
                session.phase_metrics[f'{full_hand}_withdraw']['displacement'][displacement])
            session.metrics[f'hand_{hand}_reach_{displacement}_displacement'] = np.array(
                session.phase_metrics[f'{full_hand}_reach']['displacement'][displacement])

        session.metrics[f'hand_{hand}_withdraw_euclidean_displacement_mean'], \
        session.metrics[f'hand_{hand}_withdraw_euclidean_displacement_cv'] = safe_compute_stats(
            withdraw_euclidean, f'hand_{hand}_withdraw_euclidean_displacement')
        
        session.metrics[f'hand_{hand}_reach_euclidean_displacement_mean'], \
        session.metrics[f'hand_{hand}_reach_euclidean_displacement_cv'] = safe_compute_stats(
            reach_euclidean, f'hand_{hand}_reach_euclidean_displacement')

        # Stroke Detection (use unmasked for detection robustness)
        session.metrics[f'hand_{hand}_strokes'] = utils.detect_strokes(y_unmasked, session.scale_factor)
        session.metrics[f'hand_{hand}_stroke_count'] = len(session.metrics[f'hand_{hand}_strokes'])
        session.metrics[f'hand_{hand}_stroke_rate'] = session.metrics[f'hand_{hand}_stroke_count'] / (
            len(y_masked) / session.fps) if len(y_masked) > 0 else 0
        
        # Stroke amplitude (use masked data)
        session.metrics[f'hand_{hand}_stroke_amplitude'] = utils.compute_stroke_amplitudes(
            y_masked, session.phase_metrics[f'{full_hand}_withdraw']['ranges'])
        
        session.metrics[f'hand_{hand}_stroke_amplitude_mean'], \
        session.metrics[f'hand_{hand}_stroke_amplitude_cv'] = safe_compute_stats(
            session.metrics[f'hand_{hand}_stroke_amplitude'], f'hand_{hand}_stroke_amplitude')

        # Range of Motion (use masked data)
        session.metrics[f'hand_{hand}_range_of_motion'] = utils.compute_range_of_motion(y_masked)

        # Path Length & Circuity (use masked data)
        session.phase_metrics[f'{full_hand}_withdraw']['path_length'] = utils.compute_path_length(
            x_masked, y_masked, withdraw_ranges)
        session.phase_metrics[f'{full_hand}_reach']['path_length'] = utils.compute_path_length(
            x_masked, y_masked, reach_ranges)
       
        session.phase_metrics[f'{full_hand}_withdraw']['circuity'] = utils.compute_circuity(
            session.phase_metrics[f'{full_hand}_withdraw']['path_length'],
            session.phase_metrics[f'{full_hand}_withdraw']['displacement']['euclidean'])
        session.phase_metrics[f'{full_hand}_reach']['circuity'] = utils.compute_circuity(
            session.phase_metrics[f'{full_hand}_reach']['path_length'],
            session.phase_metrics[f'{full_hand}_reach']['displacement']['euclidean'])
        
        session.metrics[f'hand_{hand}_withdraw_circuity_mean'], \
        session.metrics[f'hand_{hand}_withdraw_circuity_cv'] = safe_compute_stats(
            session.phase_metrics[f'{full_hand}_withdraw']['circuity'], f'hand_{hand}_withdraw_circuity')
        
        session.metrics[f'hand_{hand}_reach_circuity_mean'], \
        session.metrics[f'hand_{hand}_reach_circuity_cv'] = safe_compute_stats(
            session.phase_metrics[f'{full_hand}_reach']['circuity'], f'hand_{hand}_reach_circuity')

        # Movement Scaling Correlation
        session.phase_metrics[f'{full_hand}_withdraw']['movement_scaling_correlation'] = \
            utils.compute_movement_scaling_correlation(
                session.phase_metrics[f'{full_hand}_withdraw']['displacement']['euclidean'], 
                session.phase_metrics[f'{full_hand}_withdraw']['peak_speed'])

        session.phase_metrics[f'{full_hand}_reach']['movement_scaling_correlation'] = \
            utils.compute_movement_scaling_correlation(
                session.phase_metrics[f'{full_hand}_reach']['displacement']['euclidean'], 
                session.phase_metrics[f'{full_hand}_reach']['peak_speed'])
    
        # Heading Direction & Concentration (use masked data)
        session.phase_metrics[f'{full_hand}_withdraw']['heading_direction'] = utils.compute_heading_direction(
            x_masked, y_masked, withdraw_ranges)
        session.phase_metrics[f'{full_hand}_reach']['heading_direction'] = utils.compute_heading_direction(
            x_masked, y_masked, reach_ranges)

        session.phase_metrics[f'{full_hand}_withdraw']['heading_concentration'] = utils.compute_heading_concentration(
            session.phase_metrics[f'{full_hand}_withdraw']['heading_direction'])
        session.phase_metrics[f'{full_hand}_reach']['heading_concentration'] = utils.compute_heading_concentration(
            session.phase_metrics[f'{full_hand}_reach']['heading_direction'])

        session.metrics[f'hand_{hand}_withdraw_heading_direction_mean'] = utils.compute_circular_mean(
            session.phase_metrics[f'{full_hand}_withdraw']['heading_direction'])
        session.metrics[f'hand_{hand}_reach_heading_direction_mean'] = utils.compute_circular_mean(
            session.phase_metrics[f'{full_hand}_reach']['heading_direction'])

        # Hand-Nose Distance (use masked data for both)
        nose_x_masked, nose_y_masked, _ = get_masked_trajectories(session, 'nose')
        nose_x_unmasked = session.metrics['nose_x_trajectory']
        nose_y_unmasked = session.metrics['nose_y_trajectory']
        
        if validate_trajectory(nose_x_masked, 'nose_x') and validate_trajectory(nose_y_masked, 'nose_y'):
            hnd_masked = utils.compute_euclidean_distance(x_masked, y_masked, nose_x_masked, nose_y_masked)
            hnd_unmasked = utils.compute_euclidean_distance(x_unmasked, y_unmasked, nose_x_unmasked, nose_y_unmasked)
            store_unmasked_for_plotting(session, f'hand_{hand}_nose_distance', hnd_masked, hnd_unmasked)
            
            session.metrics[f'hand_{hand}_nose_distance_mean'], \
            session.metrics[f'hand_{hand}_nose_distance_cv'] = safe_compute_stats(
                hnd_masked, f'hand_{hand}_nose_distance')
        else:
            session.metrics[f'hand_{hand}_nose_distance'] = np.full_like(y_masked, np.nan)
            session.metrics[f'hand_{hand}_nose_distance_unmasked'] = np.full_like(y_unmasked, np.nan)
            session.metrics[f'hand_{hand}_nose_distance_mean'] = np.nan
            session.metrics[f'hand_{hand}_nose_distance_cv'] = np.nan

    # Bimanual Metrics (combining left and right)
    right_x_masked, right_y_masked, _ = get_masked_trajectories(session, 'hand_r')
    left_x_masked, left_y_masked, _ = get_masked_trajectories(session, 'hand_l')
    right_x_unmasked = session.metrics['hand_r_x_trajectory']
    right_y_unmasked = session.metrics['hand_r_y_trajectory']
    left_x_unmasked = session.metrics['hand_l_x_trajectory']
    left_y_unmasked = session.metrics['hand_l_y_trajectory']

    left_reach_phase_ranges = session.phase_metrics['left_reach']['ranges']
    right_reach_phase_ranges = session.phase_metrics['right_reach']['ranges']
    left_withdraw_phase_ranges = session.phase_metrics['left_withdraw']['ranges']
    right_withdraw_phase_ranges = session.phase_metrics['right_withdraw']['ranges']
    left_cycles = session.metrics['hand_l_cycles']
    right_cycles = session.metrics['hand_r_cycles']

    # Displacement Combined
    session.metrics['hand_withdraw_displacement_dx'] = safe_concatenate_lists(
        session.phase_metrics['left_withdraw']['displacement']['dx'],
        session.phase_metrics['right_withdraw']['displacement']['dx'])
    
    session.metrics['hand_withdraw_displacement_dy'] = safe_concatenate_lists(
        session.phase_metrics['left_withdraw']['displacement']['dy'],
        session.phase_metrics['right_withdraw']['displacement']['dy'])
    
    session.metrics['hand_withdraw_displacement_euclidean'] = safe_concatenate_lists(
        session.phase_metrics['left_withdraw']['displacement']['euclidean'],
        session.phase_metrics['right_withdraw']['displacement']['euclidean'])

    session.metrics['hand_reach_displacement_dx'] = safe_concatenate_lists(
        session.phase_metrics['left_reach']['displacement']['dx'],
        session.phase_metrics['right_reach']['displacement']['dx'])
    
    session.metrics['hand_reach_displacement_dy'] = safe_concatenate_lists(
        session.phase_metrics['left_reach']['displacement']['dy'],
        session.phase_metrics['right_reach']['displacement']['dy'])
    
    session.metrics['hand_reach_displacement_euclidean'] = safe_concatenate_lists(
        session.phase_metrics['left_reach']['displacement']['euclidean'],
        session.phase_metrics['right_reach']['displacement']['euclidean'])

    # Symmetry (use masked data for computation, store unmasked for plotting)
    sym_masked = utils.compute_symmetry(left_y_masked, right_y_masked)
    sym_unmasked = utils.compute_symmetry(left_y_unmasked, right_y_unmasked)
    store_unmasked_for_plotting(session, 'symmetry', sym_masked, sym_unmasked)
    
    session.metrics['symmetry_mean'], session.metrics['symmetry_cv'] = safe_compute_stats(
        sym_masked, 'symmetry')
    session.metrics['symmetry_index'] = utils.compute_symmetry_index(left_y_masked, right_y_masked)

    # Bimanual Coordination (use masked displacements)
    left_displacement_masked = utils.compute_displacement(left_y_masked)
    right_displacement_masked = utils.compute_displacement(right_y_masked)
    
    r_val, p_val, slope, intercept = utils.compute_bimanual_coordination(
        left_displacement_masked, right_displacement_masked)
    
    session.metrics.setdefault('bimanual_coordination', {})
    session.metrics['bimanual_coordination']['r_value'] = r_val
    session.metrics['bimanual_coordination']['p_value'] = p_val
    session.metrics['bimanual_coordination']['slope'] = slope
    session.metrics['bimanual_coordination']['intercept'] = intercept

    # Cyclewise Bimanual Coordination: compute correlation within each cycle and average across cycles
    session.metrics.setdefault('cyclewise_bimanual_coordination', {})
    cyclewise_results = utils.compute_cyclewise_bimanual_coordination(left_cycles, right_cycles, 
                                left_displacement_masked, right_displacement_masked)
    session.metrics['cyclewise_bimanual_coordination']['correlations'] = cyclewise_results['correlations']
    

    # Interhand Distance (use masked data for computation, store unmasked for plotting)
    ihd_masked = utils.compute_euclidean_distance(left_x_masked, left_y_masked, right_x_masked, right_y_masked)
    ihd_unmasked = utils.compute_euclidean_distance(left_x_unmasked, left_y_unmasked, right_x_unmasked, right_y_unmasked)
    store_unmasked_for_plotting(session, 'interhand_distance', ihd_masked, ihd_unmasked)
    
    withdraw_l_interhand_distances, withdraw_r_interhand_distances, \
    reach_l_interhand_distances, reach_r_interhand_distances = \
        utils.compute_phasewise_euclidean_limb_distance(
            left_y_masked, left_x_masked, right_y_masked, right_x_masked,
            left_withdraw_phase_ranges, left_reach_phase_ranges, 
            right_withdraw_phase_ranges, right_reach_phase_ranges)

    session.phase_metrics['left_withdraw']['interhand_distance'] = withdraw_l_interhand_distances
    session.phase_metrics['right_withdraw']['interhand_distance'] = withdraw_r_interhand_distances
    session.phase_metrics['left_reach']['interhand_distance'] = reach_l_interhand_distances
    session.phase_metrics['right_reach']['interhand_distance'] = reach_r_interhand_distances

    session.metrics['interhand_distance_withdraw'] = safe_concatenate_lists(
        withdraw_l_interhand_distances, withdraw_r_interhand_distances)
    session.metrics['interhand_distance_reach'] = safe_concatenate_lists(
        reach_l_interhand_distances, reach_r_interhand_distances)

    session.metrics['interhand_distance_withdraw_mean'], \
    session.metrics['interhand_distance_withdraw_cv'] = safe_compute_stats(
        session.metrics['interhand_distance_withdraw'], 'interhand_distance_withdraw')
    
    session.metrics['interhand_distance_reach_mean'], \
    session.metrics['interhand_distance_reach_cv'] = safe_compute_stats(
        session.metrics['interhand_distance_reach'], 'interhand_distance_reach')

    # Cross-Correlation (use masked displacements)
    best_lag, max_correlation, lags, correlations = utils.compute_lag(
        left_displacement_masked, right_displacement_masked, session.fps)
    
    session.metrics.setdefault('cross_correlation', {})
    session.metrics['cross_correlation']['best_lag'] = best_lag
    session.metrics['cross_correlation']['max_correlation'] = max_correlation
    session.metrics['cross_correlation']['lags'] = lags 
    session.metrics['cross_correlation']['correlation'] = correlations

    print("Hand-based metrics computation complete.")


def compute_head_metrics(session):
    """Compute head-based metrics for the session"""
    print("Computing head-based metrics...")

    # Get masked and unmasked trajectories
    nose_masked = stack_masked(session, "nose")
    ear_l_masked = stack_masked(session, "ear_l")
    ear_r_masked = stack_masked(session, "ear_r")
    
    nose_unmasked = stack(session, "nose")
    ear_l_unmasked = stack(session, "ear_l")
    ear_r_unmasked = stack(session, "ear_r")
    
    nose_x_masked, nose_y_masked, _ = get_masked_trajectories(session, 'nose')
    string_x_masked, string_y_masked, _ = get_masked_trajectories(session, 'string')
    
    nose_x_unmasked = session.metrics['nose_x_trajectory']
    nose_y_unmasked = session.metrics['nose_y_trajectory']
    string_x_unmasked = session.metrics['string_x_trajectory']
    string_y_unmasked = session.metrics['string_y_trajectory']

    left_reach_phase_ranges = session.phase_metrics['left_reach']['ranges']
    right_reach_phase_ranges = session.phase_metrics['right_reach']['ranges']
    left_withdraw_phase_ranges = session.phase_metrics['left_withdraw']['ranges']
    right_withdraw_phase_ranges = session.phase_metrics['right_withdraw']['ranges']

    # Head Yaw, Pitch, Roll (compute from both masked and unmasked, store both)
    yaw_masked = utils.compute_yaw(nose_masked, ear_l_masked, ear_r_masked)
    yaw_unmasked = utils.compute_yaw(nose_unmasked, ear_l_unmasked, ear_r_unmasked)
    store_unmasked_for_plotting(session, 'head_yaw', yaw_masked, yaw_unmasked)
    
    pitch_masked = utils.compute_pitch(nose_masked, ear_l_masked, ear_r_masked)
    pitch_unmasked = utils.compute_pitch(nose_unmasked, ear_l_unmasked, ear_r_unmasked)
    store_unmasked_for_plotting(session, 'head_pitch', pitch_masked, pitch_unmasked)
    
    roll_masked = utils.compute_roll(ear_l_masked, ear_r_masked)
    roll_unmasked = utils.compute_roll(ear_l_unmasked, ear_r_unmasked)
    store_unmasked_for_plotting(session, 'head_roll', roll_masked, roll_unmasked)

    session.metrics['head_yaw_mean'], session.metrics['head_yaw_cv'] = safe_compute_stats(
        yaw_masked, 'head_yaw')
    session.metrics['head_pitch_mean'], session.metrics['head_pitch_cv'] = safe_compute_stats(
        pitch_masked, 'head_pitch')
    session.metrics['head_roll_mean'], session.metrics['head_roll_cv'] = safe_compute_stats(
        roll_masked, 'head_roll')

    # Phasewise Head Movement Averages/Change (use masked for stats)
    head_metrics = {
        'yaw': yaw_masked,
        'pitch': pitch_masked,
        'roll': roll_masked
    }
    phase_dict = {
        'left_withdraw': left_withdraw_phase_ranges,
        'right_withdraw': right_withdraw_phase_ranges,
        'left_reach': left_reach_phase_ranges,
        'right_reach': right_reach_phase_ranges
    }

    avg_metrics, change_metrics = utils.compute_head_metrics_phasewise(head_metrics, phase_dict)

    for phase in phase_dict.keys():
        session.phase_metrics[phase]['head_yaw_average'] = avg_metrics[phase]['yaw']
        session.phase_metrics[phase]['head_yaw_change'] = change_metrics[phase]['yaw']
        session.phase_metrics[phase]['head_pitch_average'] = avg_metrics[phase]['pitch']
        session.phase_metrics[phase]['head_pitch_change'] = change_metrics[phase]['pitch']
        session.phase_metrics[phase]['head_roll_average'] = avg_metrics[phase]['roll']
        session.phase_metrics[phase]['head_roll_change'] = change_metrics[phase]['roll']

    # Phasewise Nose Displacement (use masked data)
    for hand in ['left', 'right']:
        withdraw_phase_ranges = session.phase_metrics[f'{hand}_withdraw']['ranges']
        reach_phase_ranges = session.phase_metrics[f'{hand}_reach']['ranges']

        nose_dx_withdraw, nose_dy_withdraw, nose_euclidean_withdraw = utils.compute_phasewise_displacement(
            nose_x_masked, nose_y_masked, withdraw_phase_ranges)
        nose_dx_reach, nose_dy_reach, nose_euclidean_reach = utils.compute_phasewise_displacement(
            nose_x_masked, nose_y_masked, reach_phase_ranges)

        session.phase_metrics[f'{hand}_withdraw']['nose_displacement'] = {
            'dx': nose_dx_withdraw,
            'dy': nose_dy_withdraw,
            'euclidean': nose_euclidean_withdraw
        }
        session.phase_metrics[f'{hand}_reach']['nose_displacement'] = {
            'dx': nose_dx_reach,
            'dy': nose_dy_reach,
            'euclidean': nose_euclidean_reach
        }

    for displacement in ['dx', 'dy', 'euclidean']:
        session.metrics[f'nose_withdraw_displacement_{displacement}'] = safe_concatenate_lists(
            session.phase_metrics['left_withdraw']['nose_displacement'][displacement],
            session.phase_metrics['right_withdraw']['nose_displacement'][displacement])
        
        session.metrics[f'nose_reach_displacement_{displacement}'] = safe_concatenate_lists(
            session.phase_metrics['left_reach']['nose_displacement'][displacement],
            session.phase_metrics['right_reach']['nose_displacement'][displacement])

    session.metrics['nose_withdraw_displacement_euclidean_mean'], \
    session.metrics['nose_withdraw_displacement_euclidean_cv'] = safe_compute_stats(
        session.metrics['nose_withdraw_displacement_euclidean'], 'nose_withdraw_displacement_euclidean')
    
    session.metrics['nose_reach_displacement_euclidean_mean'], \
    session.metrics['nose_reach_displacement_euclidean_cv'] = safe_compute_stats(
        session.metrics['nose_reach_displacement_euclidean'], 'nose_reach_displacement_euclidean')

    # Nose-String Distance & Correlation (use masked data)
    if validate_trajectory(string_x_masked, 'string_x') and \
       validate_trajectory(string_y_masked, 'string_y'):
        string_masked = stack_masked(session, 'string')
        string_unmasked = stack(session, 'string')

        nsd_masked = utils.compute_euclidean_distance(nose_x_masked, nose_y_masked, string_x_masked, string_y_masked)
        nsd_unmasked = utils.compute_euclidean_distance(nose_x_unmasked, nose_y_unmasked, string_x_unmasked, string_y_unmasked)
        store_unmasked_for_plotting(session, 'nose_string_distance', nsd_masked, nsd_unmasked)
        
        session.metrics['nose_string_distance_mean'], \
        session.metrics['nose_string_distance_cv'] = safe_compute_stats(
            nsd_masked, 'nose_string_distance')
        
        nose_displacement, string_displacement, r_squared, p_value = \
            utils.compute_nose_string_tracking_correlation(nose_masked, string_masked)

        session.metrics['nose_string_tracking_correlation'] = {
            'r_squared': r_squared,
            'p_value': p_value,
            'nose_displacement': nose_displacement,
            'string_displacement': string_displacement
        }
    else:
        print("WARNING: String trajectory not valid, skipping nose-string metrics")
        session.metrics['nose_string_distance'] = np.full_like(nose_y_masked, np.nan)
        session.metrics['nose_string_distance_unmasked'] = np.full_like(nose_y_unmasked, np.nan)
        session.metrics['nose_string_distance_mean'] = np.nan
        session.metrics['nose_string_distance_cv'] = np.nan
        session.metrics['nose_string_tracking_correlation'] = {
            'r_squared': np.nan,
            'p_value': np.nan
        }

    print("Head-based metrics computation complete.")


def compute_postural_metrics(session):
    """Compute postural metrics for the session, including phasewise metrics and correlations."""
    print("Computing postural metrics...")

    # Prepare keypoints with masking
    upper_l_masked = stack_masked(session, 'upper_l')
    upper_r_masked = stack_masked(session, 'upper_r')
    lower_l_masked = stack_masked(session, 'lower_l')
    lower_r_masked = stack_masked(session, 'lower_r')
    spine_1_masked = stack_masked(session, 'spine_1')
    spine_2_masked = stack_masked(session, 'spine_2')
    spine_3_masked = stack_masked(session, 'spine_3')
    ear_l_masked = stack_masked(session, 'ear_l')
    ear_r_masked = stack_masked(session, 'ear_r')
    nose_masked = stack_masked(session, 'nose')
    foot_l_masked = stack_masked(session, 'foot_l')
    foot_r_masked = stack_masked(session, 'foot_r')
    hand_l_masked = stack_masked(session, 'hand_l')
    hand_r_masked = stack_masked(session, 'hand_r')
    
    # Also get unmasked versions
    upper_l_unmasked = stack(session, 'upper_l')
    upper_r_unmasked = stack(session, 'upper_r')
    lower_l_unmasked = stack(session, 'lower_l')
    lower_r_unmasked = stack(session, 'lower_r')
    spine_1_unmasked = stack(session, 'spine_1')
    spine_2_unmasked = stack(session, 'spine_2')
    spine_3_unmasked = stack(session, 'spine_3')
    ear_l_unmasked = stack(session, 'ear_l')
    ear_r_unmasked = stack(session, 'ear_r')
    nose_unmasked = stack(session, 'nose')
    foot_l_unmasked = stack(session, 'foot_l')
    foot_r_unmasked = stack(session, 'foot_r')

    phase_dict = {
        'left_reach': session.phase_metrics['left_reach']['ranges'],
        'right_reach': session.phase_metrics['right_reach']['ranges'],
        'left_withdraw': session.phase_metrics['left_withdraw']['ranges'],
        'right_withdraw': session.phase_metrics['right_withdraw']['ranges']
    }

    # Body vectors (use masked data for computation)
    torso_upper_masked = utils.compute_midpoint(upper_l_masked, upper_r_masked)
    torso_lower_masked = utils.compute_midpoint(lower_l_masked, lower_r_masked)
    body_vector_masked = torso_upper_masked - torso_lower_masked
    
    torso_upper_unmasked = utils.compute_midpoint(upper_l_unmasked, upper_r_unmasked)
    torso_lower_unmasked = utils.compute_midpoint(lower_l_unmasked, lower_r_unmasked)
    body_vector_unmasked = torso_upper_unmasked - torso_lower_unmasked

    ear_mid_masked = utils.compute_midpoint(ear_l_masked, ear_r_masked)
    head_vector_masked = ear_mid_masked - nose_masked
    
    ear_mid_unmasked = utils.compute_midpoint(ear_l_unmasked, ear_r_unmasked)
    head_vector_unmasked = ear_mid_unmasked - nose_unmasked

    top_displacement = utils.compute_displacement(spine_1_masked)
    mid_displacement = utils.compute_displacement(spine_2_masked)
    bot_displacement = utils.compute_displacement(spine_3_masked)

    left_displacement = utils.compute_displacement(hand_l_masked)
    right_displacement = utils.compute_displacement(hand_r_masked)
    hand_displacements = {
        'left': left_displacement,
        'right': right_displacement
    }

    # Body Length (use masked data)
    bl_masked = utils.compute_body_length(ear_l_masked, ear_r_masked, foot_l_masked, foot_r_masked)
    bl_unmasked = utils.compute_body_length(ear_l_unmasked, ear_r_unmasked, foot_l_unmasked, foot_r_unmasked)
    store_unmasked_for_plotting(session, 'body_length', bl_masked, bl_unmasked)
    
    session.metrics['body_length_mean'], session.metrics['body_length_cv'] = safe_compute_stats(
        bl_masked, 'body_length')

    # Phasewise Body Length
    for phase_name, ranges in phase_dict.items():
        phase_bl, phase_bl_cv = utils.compute_phasewise_bodylength(bl_masked, ranges)
        session.phase_metrics[phase_name]['body_length'] = phase_bl
        session.phase_metrics[phase_name]['body_length_cv'] = phase_bl_cv
    
    # Inter-foot distance (use masked data)
    foot_l_x_masked, foot_l_y_masked, _ = get_masked_trajectories(session, 'foot_l')
    foot_r_x_masked, foot_r_y_masked, _ = get_masked_trajectories(session, 'foot_r')
    
    session.metrics['interfoot_distance'] = utils.compute_euclidean_distance(
        foot_l_x_masked, foot_l_y_masked, foot_r_x_masked, foot_r_y_masked
    )

    # Phasewise interfoot distance
    withdraw_l, withdraw_r, reach_l, reach_r = utils.compute_phasewise_euclidean_limb_distance(
        foot_l_y_masked, foot_l_x_masked, foot_r_y_masked, foot_r_x_masked,
        phase_dict['left_withdraw'], phase_dict['left_reach'],
        phase_dict['right_withdraw'], phase_dict['right_reach']
    )

    session.phase_metrics['left_withdraw']['interfoot_distance'] = withdraw_l
    session.phase_metrics['right_withdraw']['interfoot_distance'] = withdraw_r
    session.phase_metrics['left_reach']['interfoot_distance'] = reach_l
    session.phase_metrics['right_reach']['interfoot_distance'] = reach_r

    # Core body signals (compute from both masked and unmasked)
    body_angle_masked = utils.compute_body_angle(body_vector_masked)
    body_angle_unmasked = utils.compute_body_angle(body_vector_unmasked)
    store_unmasked_for_plotting(session, 'body_angle', body_angle_masked, body_angle_unmasked)
    
    utr_masked = utils.compute_roll(upper_l_masked, upper_r_masked)
    utr_unmasked = utils.compute_roll(upper_l_unmasked, upper_r_unmasked)
    store_unmasked_for_plotting(session, 'upper_torso_roll', utr_masked, utr_unmasked)
    
    ltr_masked = utils.compute_roll(lower_l_masked, lower_r_masked)
    ltr_unmasked = utils.compute_roll(lower_l_unmasked, lower_r_unmasked)
    store_unmasked_for_plotting(session, 'lower_torso_roll', ltr_masked, ltr_unmasked)
    
    utp_masked = utils.compute_pitch(spine_2_masked, upper_l_masked, upper_r_masked)
    utp_unmasked = utils.compute_pitch(spine_2_unmasked, upper_l_unmasked, upper_r_unmasked)
    store_unmasked_for_plotting(session, 'upper_torso_pitch', utp_masked, utp_unmasked)
    
    uty_masked = utils.compute_yaw(spine_2_masked, upper_l_masked, upper_r_masked)
    uty_unmasked = utils.compute_yaw(spine_2_unmasked, upper_l_unmasked, upper_r_unmasked)
    store_unmasked_for_plotting(session, 'upper_torso_yaw', uty_masked, uty_unmasked)
    
    sc_masked = utils.compute_spine_curvature(spine_1_masked, spine_2_masked, spine_3_masked)
    sc_unmasked = utils.compute_spine_curvature(spine_1_unmasked, spine_2_unmasked, spine_3_unmasked)
    store_unmasked_for_plotting(session, 'spine_curvature', sc_masked, sc_unmasked)
    
    hba_masked = utils.compute_head_body_alignment(head_vector_masked, body_vector_masked)
    hba_unmasked = utils.compute_head_body_alignment(head_vector_unmasked, body_vector_unmasked)
    store_unmasked_for_plotting(session, 'head_body_alignment', hba_masked, hba_unmasked)

    # Phasewise body metrics and correlations
    body_signals = [
        'body_angle', 'upper_torso_roll', 'lower_torso_roll',
        'upper_torso_pitch', 'upper_torso_yaw', 'spine_curvature', 'head_body_alignment'
    ]

    corr_pairs = [
        ('upper_torso_roll', 'lower_torso_roll'),
        ('upper_torso_roll', 'head_roll'),
        ('upper_torso_pitch', 'head_pitch'),
        ('upper_torso_yaw', 'head_yaw')
    ]

    for sig in body_signals:
        values = session.metrics[sig]
        session.metrics[f'{sig}_mean'], session.metrics[f'{sig}_cv'] = safe_compute_stats(values, sig)

        # Phasewise metrics
        for phase_name, ranges in phase_dict.items():
            phase_vals = utils.compute_phasewise_body_angles(values, ranges)
            session.phase_metrics[phase_name][f'{sig}_mean'] = phase_vals['angles_mean']
            session.phase_metrics[phase_name][f'{sig}_cv'] = phase_vals['angles_cv']
            session.phase_metrics[phase_name][f'{sig}'] = phase_vals['angles'] 

    # Phasewise correlations
    for phase_name, ranges in phase_dict.items():
        for sig_a, sig_b in corr_pairs:
            corr_dict = utils.compute_phasewise_correlation(
                session.metrics[sig_a],
                session.metrics[sig_b],
                ranges
            )
            key_name = f'{sig_a}_{sig_b}_correlation'
            session.phase_metrics[phase_name][key_name] = corr_dict['phase_corrs']

    # Overall correlations
    for sig_a, sig_b in corr_pairs:
        key_name = f'{sig_a}_{sig_b}_correlation'
        session.metrics[key_name] = utils.compute_phasewise_correlation(
            session.metrics[sig_a],
            session.metrics[sig_b]
        )['overall_corr']

    # Body Recruitment
    norm_factor = np.nanmean(session.metrics['body_length'])

    for disp, region in zip([top_displacement, mid_displacement, bot_displacement],['upper', 'mid', 'lower']):
        for hand in ['left', 'right']:
            for phase in ['reach', 'withdraw']:
                recruitment = utils.compute_body_recruitment(disp, hand_displacements[hand], 
                    phase_dict[f'{hand}_{phase}'], norm_factor)
                session.phase_metrics[f'{hand}_{phase}'][f'{region}_torso_recruitment_ratio'] = recruitment['phase_ratios']
    
    print("Postural metrics computation complete.")


def compute_arm_metrics(session):
    """Compute arm segment and joint kinematics."""
    print("Computing arm metrics...")

    arms = ['left', 'right']
    joints = ['shoulder', 'elbow']
    phases = ['reach', 'withdraw']

    for arm in arms:
        # Get masked and unmasked coordinates
        shoulder_masked = stack_masked(session, f'shoulder_{arm[0]}')
        elbow_masked = stack_masked(session, f'elbow_{arm[0]}')
        wrist_masked = stack_masked(session, f'wrist_{arm[0]}')
        
        shoulder_unmasked = stack(session, f'shoulder_{arm[0]}')
        elbow_unmasked = stack(session, f'elbow_{arm[0]}')
        wrist_unmasked = stack(session, f'wrist_{arm[0]}')

        # Segment lengths (compute from both masked and unmasked)
        upper_arm_masked = np.linalg.norm(elbow_masked - shoulder_masked, axis=1)
        forearm_masked = np.linalg.norm(wrist_masked - elbow_masked, axis=1)
        full_arm_masked = np.linalg.norm(wrist_masked - shoulder_masked, axis=1)
        
        upper_arm_unmasked = np.linalg.norm(elbow_unmasked - shoulder_unmasked, axis=1)
        forearm_unmasked = np.linalg.norm(wrist_unmasked - elbow_unmasked, axis=1)
        full_arm_unmasked = np.linalg.norm(wrist_unmasked - shoulder_unmasked, axis=1)

        session.metrics[f'arm_{arm[0]}_upper_arm_length'] = upper_arm_masked
        session.metrics[f'arm_{arm[0]}_forearm_length'] = forearm_masked
        store_unmasked_for_plotting(session, f'arm_{arm[0]}_full_arm_length', full_arm_masked, full_arm_unmasked)

        # Torso reference (use masked data)
        torso_ref_masked = stack_masked(session, 'spine_1')
        torso_ref_unmasked = stack(session, 'spine_1')

        # Joint angles (compute from both masked and unmasked)
        elbow_angle_masked = utils.compute_joint_angle(shoulder_masked, elbow_masked, wrist_masked)
        elbow_angle_unmasked = utils.compute_joint_angle(shoulder_unmasked, elbow_unmasked, wrist_unmasked)
        store_unmasked_for_plotting(session, f'arm_{arm[0]}_elbow_angle', elbow_angle_masked, elbow_angle_unmasked)
        
        shoulder_angle_masked = utils.compute_joint_angle(elbow_masked, shoulder_masked, torso_ref_masked)
        shoulder_angle_unmasked = utils.compute_joint_angle(elbow_unmasked, shoulder_unmasked, torso_ref_unmasked)
        store_unmasked_for_plotting(session, f'arm_{arm[0]}_shoulder_angle', shoulder_angle_masked, shoulder_angle_unmasked)

        # Summary metrics (use masked for stats)
        session.metrics[f'arm_{arm[0]}_shoulder_angle_mean'], \
        session.metrics[f'arm_{arm[0]}_shoulder_angle_cv'] = safe_compute_stats(
            shoulder_angle_masked, f'arm_{arm[0]}_shoulder_angle')
        
        session.metrics[f'arm_{arm[0]}_elbow_angle_mean'], \
        session.metrics[f'arm_{arm[0]}_elbow_angle_cv'] = safe_compute_stats(
            elbow_angle_masked, f'arm_{arm[0]}_elbow_angle')

        # Angular velocity (use masked angles)
        session.metrics[f'arm_{arm[0]}_shoulder_angular_velocity'] = utils.compute_velocity(
            shoulder_angle_masked, session.fps)
        session.metrics[f'arm_{arm[0]}_elbow_angular_velocity'] = utils.compute_velocity(
            elbow_angle_masked, session.fps)

        # Extension ratio (compute from both masked and unmasked)
        ext_ratio_masked = utils.compute_arm_extension_ratio(full_arm_masked)
        ext_ratio_unmasked = utils.compute_arm_extension_ratio(full_arm_unmasked)
        store_unmasked_for_plotting(session, f'arm_{arm[0]}_extension_ratio', ext_ratio_masked, ext_ratio_unmasked)

        # Phasewise arm metrics (use masked angles)
        for phase in phases:
            ranges = session.phase_metrics[f'{arm}_{phase}']['ranges']

            for joint in joints:
                angle_key = f'arm_{arm[0]}_{joint}_angle'
                angle = session.metrics[angle_key]  # This is the masked version
                phase_metrics = utils.compute_phasewise_arm_metrics(angle, ranges, session.fps)
        
                # Add phase summaries
                session.phase_metrics[f'{arm}_{phase}'][f'{joint}_angle_mean'] = phase_metrics['angle_mean']
                session.phase_metrics[f'{arm}_{phase}'][f'{joint}_mean_speed'] = phase_metrics['speed_mean']
                session.phase_metrics[f'{arm}_{phase}'][f'{joint}_peak_speed'] = phase_metrics['speed_peak']
                session.phase_metrics[f'{arm}_{phase}'][f'{joint}_angle_change'] = phase_metrics['angle_change']
            
            ext_ratio = session.metrics[f'arm_{arm[0]}_extension_ratio']  # Masked version
            phase_ext_metrics = utils.compute_phasewise_extension_ratio(ext_ratio, ranges)
            session.phase_metrics[f'{arm}_{phase}']['extension_ratio_mean'] = phase_ext_metrics['mean']
            session.phase_metrics[f'{arm}_{phase}']['extension_ratio_min'] = phase_ext_metrics['min']
            session.phase_metrics[f'{arm}_{phase}']['extension_ratio_max'] = phase_ext_metrics['max']

    print("Arm metrics computation complete.")


def compute_all_metrics(session):
    """Compute all metrics for the session"""
    print("="*50)
    print("Starting comprehensive metrics computation...")
    print(f"Using likelihood threshold: {session.likelihood_threshold}")
    print("="*50)
    
    compute_hand_metrics(session)
    compute_head_metrics(session)
    compute_postural_metrics(session)
    compute_arm_metrics(session)
    
    print("="*50)
    print("All metrics computation complete.")
    print(f"Total metrics computed: {len(session.metrics)}")
    print(f"Phase-specific metrics: {len(session.phase_metrics)}")
    print("="*50)
    #print("\nNote: Metrics ending in '_unmasked' are for visualization only.")
    #print("Use the standard metric names (without '_unmasked') for analysis.")