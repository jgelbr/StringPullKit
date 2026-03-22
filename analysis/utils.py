import numpy as np 
from scipy.signal import correlate, find_peaks
from scipy.stats import pearsonr

# ===================================== #
# Geometry / Vector Utilities
# ===================================== #

def compute_euclidean_distance(x1, y1, x2, y2):
    """Euclidean distance between two 2D points (NaN-aware)"""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def compute_displacement(coords):
    """
    Compute displacement relative to first valid frame for 1D or 2D arrays.
    NaN-aware: uses first valid (non-NaN) frame as reference.
    """
    coords = np.asarray(coords)
    
    if coords.ndim == 1:
        # Find first valid value
        valid_idx = np.where(~np.isnan(coords))[0]
        if len(valid_idx) == 0:
            return np.full_like(coords, np.nan)
        first_valid = coords[valid_idx[0]]
        return coords - first_valid
    
    elif coords.ndim == 2:
        # Find first valid row (both x and y are valid)
        valid_rows = np.where(~np.isnan(coords).any(axis=1))[0]
        if len(valid_rows) == 0:
            return np.full(len(coords), np.nan)
        first_valid = coords[valid_rows[0]]
        displacement = np.linalg.norm(coords - first_valid, axis=1)
        return displacement


def compute_distances_for_ranges(left_y, left_x, right_y, right_x, ranges):
    """Compute mean Euclidean distances for a list of frame ranges (NaN-aware)."""
    dists = []
    for s, e in ranges:
        if s >= e or e > len(left_x):
            dists.append(np.nan)
            continue
        dist = np.sqrt((left_x[s:e] - right_x[s:e])**2 + (left_y[s:e] - right_y[s:e])**2)
        dists.append(np.nanmean(dist))
    return dists

def compute_path_length(x, y, phase_range):
    """
    Total path length for given phase ranges (NaN-aware).
    Skips NaN values in path calculation.
    """
    lengths = []
    for start, end in phase_range:
        x_seg = x[start:end+1]
        y_seg = y[start:end+1]
        
        # Only compute differences where both current and next points are valid
        valid_mask = ~np.isnan(x_seg[:-1]) & ~np.isnan(x_seg[1:]) & \
                     ~np.isnan(y_seg[:-1]) & ~np.isnan(y_seg[1:])
        
        if np.sum(valid_mask) == 0:
            lengths.append(np.nan)
            continue
            
        dx = np.diff(x_seg)[valid_mask]
        dy = np.diff(y_seg)[valid_mask]
        diffs = np.sqrt(dx**2 + dy**2)
        lengths.append(np.sum(diffs))
    return lengths

def compute_midpoint(a, b):
    """Midpoint of two arrays (NaN propagates)"""
    return (a + b) / 2.0

def compute_angle_from_vector(dx, dy):
    """Angle from vector components in degrees"""
    return np.degrees(np.arctan2(dy, dx))

def compute_range_of_motion(y):
    """Range of motion in y-direction (NaN-aware)"""
    return np.nanmax(y) - np.nanmin(y)

def compute_joint_angle(a, b, c):
    """
    Compute joint angle at point b (in degrees). 
    a, b, c are Nx2 arrays representing 2D coordinates (NaN-aware).
    """
    ba = a - b
    bc = c - b

    # Compute norms
    ba_norm_mag = np.linalg.norm(ba, axis=1, keepdims=True)
    bc_norm_mag = np.linalg.norm(bc, axis=1, keepdims=True)
    
    # Avoid division by zero
    ba_norm_mag = np.where(ba_norm_mag == 0, np.nan, ba_norm_mag)
    bc_norm_mag = np.where(bc_norm_mag == 0, np.nan, bc_norm_mag)
    
    ba_norm = ba / ba_norm_mag
    bc_norm = bc / bc_norm_mag

    cos_angle = np.einsum('ij,ij->i', ba_norm, bc_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angles = np.degrees(np.arccos(cos_angle))
    return angles

# ===================================== #
# Circular / Statistical Utilities
# ===================================== #

def coefficient_of_variation(data):
    """Coefficient of variation (std/mean) (NaN-aware)"""
    data = np.asarray(data)
    mean_val = np.nanmean(data)
    if len(data) > 0 and mean_val != 0 and not np.isnan(mean_val):
        return np.nanstd(data) / mean_val
    return np.nan

def compute_circular_mean(angles):
    """Compute circular mean of angles in degrees (NaN-aware)"""
    angles = np.asarray(angles)
    angles = angles[~np.isnan(angles)]
    if len(angles) == 0:
        return np.nan
    
    radians = np.radians(angles)
    sin_sum = np.sum(np.sin(radians))
    cos_sum = np.sum(np.cos(radians))
    mean_angle = compute_angle_from_vector(cos_sum, sin_sum)
    return mean_angle % 360

# ===================================== #
# Kinematics
# ===================================== #

def compute_velocity(y, fps):
    """
    Velocity from y-position (NaN-aware).
    Uses forward differences, preserves NaNs.
    """
    return np.gradient(y) * fps

def compute_acceleration(velocity, fps):
    """Acceleration from velocity (NaN-aware)"""
    return np.gradient(velocity) * fps

def compute_jerk(acceleration, fps):
    """Jerk from acceleration (NaN-aware)"""
    return np.gradient(acceleration) * fps

def compute_speed(x, y, fps):
    """Speed from x and y positions (NaN-aware)"""
    vx = compute_velocity(x, fps)
    vy = compute_velocity(y, fps)
    return np.sqrt(vx**2 + vy**2)

def compute_phasewise_speed(speed, phase_range):
    """Compute mean and peak speed for given phase ranges (NaN-aware)"""
    mean_speeds, peak_speeds = [], []
    for start, end in phase_range:
        if start < end <= len(speed):
            phase_speed = speed[start:end]
            # Require at least some valid data
            if np.sum(~np.isnan(phase_speed)) > 0:
                mean_speeds.append(np.nanmean(phase_speed))
                peak_speeds.append(np.nanmax(phase_speed))
            else:
                mean_speeds.append(np.nan)
                peak_speeds.append(np.nan)
        else:
            mean_speeds.append(np.nan)
            peak_speeds.append(np.nan)
    return mean_speeds, peak_speeds

def compute_phasewise_arm_metrics(angle, ranges, fps):
    """
    Compute per-phase angular metrics: mean angle, change, mean/peak speed, etc. (NaN-aware)
    """
    # Precompute angular velocity once
    angular_velocity = np.gradient(angle) * fps
    angular_speed = np.abs(angular_velocity)

    phase_angles = []
    phase_velocities = []
    phase_speeds = []
    mean_angles = []
    angle_changes = []
    mean_speeds = []
    peak_speeds = []

    for s, e in ranges:
        if e <= s or e > len(angle):
            phase_angles.append(np.nan)
            phase_velocities.append(np.nan)
            phase_speeds.append(np.nan)
            mean_angles.append(np.nan)
            angle_changes.append(np.nan)
            mean_speeds.append(np.nan)
            peak_speeds.append(np.nan)
            continue

        phase_angle = angle[s:e]
        phase_vel = angular_velocity[s:e]
        phase_speed = angular_speed[s:e]

        # Check for sufficient valid data
        if np.sum(~np.isnan(phase_angle)) < 2:
            phase_angles.append(phase_angle)
            phase_velocities.append(phase_vel)
            phase_speeds.append(phase_speed)
            mean_angles.append(np.nan)
            angle_changes.append(np.nan)
            mean_speeds.append(np.nan)
            peak_speeds.append(np.nan)
            continue

        phase_angles.append(phase_angle)
        phase_velocities.append(phase_vel)
        phase_speeds.append(phase_speed)

        mean_angles.append(np.nanmean(phase_angle))
        
        # Angle change: use first and last valid values
        valid_idx = np.where(~np.isnan(phase_angle))[0]
        if len(valid_idx) >= 2:
            angle_changes.append(phase_angle[valid_idx[-1]] - phase_angle[valid_idx[0]])
        else:
            angle_changes.append(np.nan)
            
        mean_speeds.append(np.nanmean(phase_speed))
        peak_speeds.append(np.nanmax(phase_speed))

    return {
        "angle": phase_angles,
        "velocity": phase_velocities,
        "speed": phase_speeds,
        "angle_mean": mean_angles,
        "angle_change": angle_changes,
        "speed_mean": mean_speeds,
        "speed_peak": peak_speeds
    }

# ===================================== #
# Phase-based Computations
# ===================================== #

def compute_phase_proportion_stats(component_durations, cycle_durations):
    """Compute mean proportion of component durations relative to cycle durations (NaN-aware)"""
    proportions = []
    for comp_dur, cycle_dur in zip(component_durations, cycle_durations):
        if cycle_dur > 0 and not np.isnan(comp_dur) and not np.isnan(cycle_dur):
            proportions.append(comp_dur / cycle_dur)
        else:
            proportions.append(np.nan)
    
    if len(proportions) == 0 or all(np.isnan(proportions)):
        return np.nan, np.nan
        
    mean_prop = np.nanmean(proportions)
    std_prop = np.nanstd(proportions)
    cv_prop = std_prop / mean_prop if mean_prop > 0 else np.nan
    return mean_prop, cv_prop

def compute_circuity(euclidean_distance, path_length):
    """Compute circuity safely: elementwise ratio with NaN for zero/invalid distances"""
    euclidean_distance = np.asarray(euclidean_distance, dtype=float)
    path_length = np.asarray(path_length, dtype=float)
  
    with np.errstate(divide='ignore', invalid='ignore'):
        circuity = np.divide(path_length, euclidean_distance)
        circuity[~np.isfinite(circuity)] = np.nan
    return circuity

def compute_movement_scaling_correlation(euclidean_distances, max_speeds):
    """Compute correlation between Euclidean distances and max speeds (NaN-aware)"""
    euclidean_distances = np.asarray(euclidean_distances, dtype=float)
    max_speeds = np.asarray(max_speeds, dtype=float)

    # Remove pairs where either value is NaN
    valid_mask = ~np.isnan(euclidean_distances) & ~np.isnan(max_speeds)
    if np.sum(valid_mask) < 2:
        return np.nan

    correlation, _ = pearsonr(max_speeds[valid_mask], euclidean_distances[valid_mask])
    return correlation

def compute_phasewise_euclidean_limb_distance(left_y, left_x, right_y, right_x,
                                           left_withdraw, left_reach, right_withdraw, right_reach):
    """Compute mean interlimb distances during withdraw/reach phases for both hands (NaN-aware)"""
    withdraw_l = compute_distances_for_ranges(left_y, left_x, right_y, right_x, left_withdraw)
    reach_l = compute_distances_for_ranges(left_y, left_x, right_y, right_x, left_reach)
    withdraw_r = compute_distances_for_ranges(left_y, left_x, right_y, right_x, right_withdraw)
    reach_r = compute_distances_for_ranges(left_y, left_x, right_y, right_x, right_reach)
    return withdraw_l, withdraw_r, reach_l, reach_r

def compute_phasewise_displacement(x, y, phase_range):
    """
    Compute dx, dy, and Euclidean displacement for each phase (NaN-aware).
    Uses first and last valid points in each phase.
    """
    dx_list, dy_list, euclidean_list = [], [], []
    for start, end in phase_range:
        if start < end <= len(x):
            x_seg = x[start:end]
            y_seg = y[start:end]
            
            # Find valid indices
            valid_x = np.where(~np.isnan(x_seg))[0]
            valid_y = np.where(~np.isnan(y_seg))[0]
            
            if len(valid_x) >= 2 and len(valid_y) >= 2:
                # Use first and last valid points
                dx = x_seg[valid_x[-1]] - x_seg[valid_x[0]]
                dy = y_seg[valid_y[-1]] - y_seg[valid_y[0]]
                dx_list.append(dx)
                dy_list.append(dy)
                euclidean_list.append(np.sqrt(dx**2 + dy**2))
            else:
                dx_list.append(np.nan)
                dy_list.append(np.nan)
                euclidean_list.append(np.nan)
        else:
            dx_list.append(np.nan)
            dy_list.append(np.nan)
            euclidean_list.append(np.nan)
    return dx_list, dy_list, euclidean_list

def compute_heading_direction(x, y, phase_range):
    """Compute heading direction (angle) for each phase (NaN-aware)"""
    directions = []
    for start, end in phase_range:
        if start < end <= len(x):
            x_seg = x[start:end]
            y_seg = y[start:end]
            
            # Find valid indices
            valid_x = np.where(~np.isnan(x_seg))[0]
            valid_y = np.where(~np.isnan(y_seg))[0]
            
            if len(valid_x) >= 2 and len(valid_y) >= 2:
                dx = x_seg[valid_x[-1]] - x_seg[valid_x[0]]
                dy = y_seg[valid_y[-1]] - y_seg[valid_y[0]]
                angle = compute_angle_from_vector(dx, dy)
                directions.append(angle)
            else:
                directions.append(np.nan)
        else:
            directions.append(np.nan)
    return directions

def compute_heading_concentration(heading_directions): 
    """Compute heading concentration (mean resultant length) from heading directions (NaN-aware)"""
    heading_directions = np.array(heading_directions, dtype=float)
    heading_directions = heading_directions[~np.isnan(heading_directions)]
    if len(heading_directions) == 0:
        return np.nan

    radians = np.radians(heading_directions)
    sin_sum = np.sum(np.sin(radians))
    cos_sum = np.sum(np.cos(radians))

    R = np.sqrt(sin_sum**2 + cos_sum**2) / len(radians)
    return R

def compute_phasewise_correlation(signal_a, signal_b, phase_ranges=None):
    """Compute overall and phasewise correlation between two signals (NaN-aware)"""
    # Overall correlation
    valid_idx = ~np.isnan(signal_a) & ~np.isnan(signal_b)
  
    if np.sum(valid_idx) >= 2:
        overall_corr = np.corrcoef(signal_a[valid_idx], signal_b[valid_idx])[0, 1]
    else:
        overall_corr = np.nan

    # Phasewise correlations
    phase_corrs = []
    if phase_ranges is not None:
        for s, e in phase_ranges:
            s_idx = max(0, s)
            e_idx = min(len(signal_a), e)
            seg_a = signal_a[s_idx:e_idx]
            seg_b = signal_b[s_idx:e_idx]
            valid_idx = ~np.isnan(seg_a) & ~np.isnan(seg_b)
            if np.sum(valid_idx) >= 2:
                phase_corrs.append(np.corrcoef(seg_a[valid_idx], seg_b[valid_idx])[0, 1])
            else:
                phase_corrs.append(np.nan)
    
    return {"overall_corr": overall_corr, "phase_corrs": phase_corrs}

# ===================================== #
# Head / Orientation Metrics
# ===================================== #

def compute_pitch(center, left, right):
    """Pitch angle in degrees (NaN-aware)"""
    side_vector = right - left
    center_vector = center - left
    
    side_norm_mag = np.linalg.norm(side_vector, axis=1, keepdims=True)
    center_norm_mag = np.linalg.norm(center_vector, axis=1, keepdims=True)
    
    # Avoid division by zero
    side_norm_mag = np.where(side_norm_mag == 0, np.nan, side_norm_mag)
    center_norm_mag = np.where(center_norm_mag == 0, np.nan, center_norm_mag)
    
    side_norm = side_vector / side_norm_mag
    center_norm = center_vector / center_norm_mag
    
    cross = side_norm[:,0] * center_norm[:,1] - side_norm[:,1] * center_norm[:,0]
    dot = np.einsum('ij,ij->i', side_norm, center_norm)
    return -compute_angle_from_vector(dot, cross) #Negative for interpretability since dlc markers are anatomical rather than video l/r

def compute_yaw(center, left, right):
    """Yaw: log-distance ratio center-sides (NaN-aware)"""
    d_left = np.linalg.norm(center - left, axis=1)
    d_right = np.linalg.norm(center - right, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        yaw = np.log(d_right / d_left)
        yaw[~np.isfinite(yaw)] = np.nan
    return yaw

def compute_roll(left, right):
    """Roll: tilt left/right in degrees (NaN-aware)"""
    dx = right[:,0] - left[:,0]
    dy = right[:,1] - left[:,1]
    return compute_angle_from_vector(dx, dy)

def compute_head_metrics_phasewise(head_metrics, phase_dict):
    """
    Compute mean and net change of yaw/pitch/roll for labeled phases (NaN-aware).
    """
    avg_metrics, change_metrics = {}, {}
    for label, ranges in phase_dict.items():
        avg_metrics[label], change_metrics[label] = {}, {}
        for metric in ['yaw', 'pitch', 'roll']:
            avg_list, change_list = [], []
            for s, e in ranges:
                segment = head_metrics[metric][s:e]
                valid_idx = np.where(~np.isnan(segment))[0]
                
                if len(valid_idx) >= 2:
                    avg_list.append(np.nanmean(segment))
                    change_list.append(segment[valid_idx[-1]] - segment[valid_idx[0]])
                else:
                    avg_list.append(np.nan)
                    change_list.append(np.nan)
            avg_metrics[label][metric] = avg_list
            change_metrics[label][metric] = change_list
    return avg_metrics, change_metrics

# ===================================== #
# Body / Limb Measurements
# ===================================== #

def compute_arm_extension_ratio(full_arm):
    """Arm extension as normalized shoulder->wrist length (NaN-aware)"""
    max_d = np.nanmax(full_arm)
    if np.isnan(max_d) or max_d == 0:
        return np.full_like(full_arm, np.nan)
    return full_arm / max_d

def compute_phasewise_extension_ratio(extension_ratio, ranges):
    """Compute per-phase mean, min, and max extension ratio (NaN-aware)"""
    phase_mean, phase_min, phase_max = [], [], []

    for s, e in ranges:
        if e <= s or e > len(extension_ratio):
            phase_mean.append(np.nan)
            phase_min.append(np.nan)
            phase_max.append(np.nan)
            continue
        segment = extension_ratio[s:e]
        if np.sum(~np.isnan(segment)) > 0:
            phase_mean.append(np.nanmean(segment))
            phase_min.append(np.nanmin(segment))
            phase_max.append(np.nanmax(segment))
        else:
            phase_mean.append(np.nan)
            phase_min.append(np.nan)
            phase_max.append(np.nan)

    return {'mean': phase_mean, 'min': phase_min, 'max': phase_max}

def compute_body_length(ear_l, ear_r, foot_l, foot_r):
    """Body length as distance from ear midpoint to foot midpoint (NaN-aware)"""
    ear_mid = compute_midpoint(ear_l, ear_r)
    foot_mid = compute_midpoint(foot_l, foot_r)
    return compute_euclidean_distance(ear_mid[:, 0], ear_mid[:, 1], foot_mid[:, 0], foot_mid[:, 1])

def compute_phasewise_bodylength(body_length, phase_ranges):
    """Compute mean and CV body length for given phase ranges (NaN-aware)"""
    phase_bodylengths = []
    phase_bodylength_cvs = []
    for start, end in phase_ranges:
        if start < end <= len(body_length):
            segment = body_length[start:end]
            if np.sum(~np.isnan(segment)) > 0:
                phase_bodylengths.append(np.nanmean(segment))
                phase_bodylength_cvs.append(coefficient_of_variation(segment))
            else:
                phase_bodylengths.append(np.nan)
                phase_bodylength_cvs.append(np.nan)
        else:
            phase_bodylengths.append(np.nan)
            phase_bodylength_cvs.append(np.nan)
    return phase_bodylengths, phase_bodylength_cvs

def compute_body_angle(body_vector):
    """Compute body angle relative to horizontal (NaN-aware)"""
    dx = body_vector[:, 0]
    dy = body_vector[:, 1]
    body_angle = compute_angle_from_vector(dx, dy)
    return body_angle

def compute_phasewise_body_angles(signal, phase_ranges):
    """Compute values and summary statistics of angle signal within phase range (NaN-aware)"""
    phase_values = []

    for (start, end) in phase_ranges:
        if end > start and start >= 0 and end <= len(signal):
            phase_segment = signal[start:end]
            if np.sum(~np.isnan(phase_segment)) > 0:
                phase_values.append(np.nanmean(phase_segment))

    if len(phase_values) > 0:
        mean_val = np.nanmean(phase_values)
        cv_val = coefficient_of_variation(phase_values)
    else:
        mean_val = np.nan
        cv_val = np.nan

    return {
        'angles': phase_values,
        'angles_mean': mean_val,
        'angles_cv': cv_val
    }

def compute_spine_curvature(spine_1, spine_2, spine_3):
    """Compute 2D curvature of the spine as deviation from straight line (NaN-aware)"""
    return 180 - compute_joint_angle(spine_1, spine_2, spine_3)

def compute_head_body_alignment(head_vector, body_vector):
    """Compute angle between head and body vectors (NaN-aware)"""
    head_norm = np.linalg.norm(head_vector, axis=1)
    body_norm = np.linalg.norm(body_vector, axis=1)
    
    # Avoid division by zero
    denominator = head_norm * body_norm
    denominator = np.where(denominator == 0, np.nan, denominator)
    
    cos_angle = np.einsum('ij,ij->i', head_vector, body_vector) / denominator
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

# ===================================== #
# Stroke, Cycle & Phase Detection
# ===================================== #

def detect_strokes(y, scale=None):
    """
    Detect beginning of withdraw phase (peaks) (NaN-aware).
    Requires continuous valid segments for peak detection.
    """
    base_prominence = 10
    min_stroke_interval = 10
    y = np.asarray(y)
    
    # Remove NaN values
    valid_mask = ~np.isnan(y)
    if np.sum(valid_mask) < min_stroke_interval * 2:
        return np.array([])

    prominence = base_prominence * scale if scale is not None else base_prominence
    
    # Find peaks only in valid segments
    # For now, we'll work with the full array and let find_peaks handle it
    # but ideally we'd find continuous segments
    try:
        peaks, _ = find_peaks(y, distance=min_stroke_interval, prominence=prominence)
        # Filter out peaks that occur at NaN positions
        peaks = peaks[~np.isnan(y[peaks])]
        return peaks
    except:
        return np.array([])

def compute_stroke_amplitudes(y, ranges):
    """Compute stroke amplitudes for given stroke ranges (NaN-aware)"""
    amplitudes = []
    for start, end in ranges:
        if start < end <= len(y):
            start_val = y[start] if not np.isnan(y[start]) else np.nan
            end_val = y[end-1] if not np.isnan(y[end-1]) else np.nan
            
            if not np.isnan(start_val) and not np.isnan(end_val):
                amplitude = start_val - end_val
                amplitudes.append(amplitude)
            else:
                amplitudes.append(np.nan)
        else:
            amplitudes.append(np.nan)
    return amplitudes

def detect_cycles_and_phases(y, fps, scale=None):
    """
    Detect cycles of withdraw and reach phases from vertical trajectory (NaN-aware).
    Only detects phases in continuous valid segments.
    """
    y = np.asarray(y)
    
    # Remove NaN values for peak detection
    valid_mask = ~np.isnan(y)
    if np.sum(valid_mask) < 20:  # Need minimum valid data
        return {
            'withdraw_ranges': [],
            'reach_ranges': [],
            'withdraw_durations': [],
            'reach_durations': [],
            'cycle_durations': [],
            'cycles': []
        }
    
    base_prominence = 10
    prominence = np.nanstd(y) * 0.5 if scale is None else base_prominence * scale

    try:
        peaks, _ = find_peaks(y, prominence=prominence)
        troughs, _ = find_peaks(-y, prominence=prominence)
        
        # Filter to only include peaks/troughs at valid positions
        peaks = peaks[valid_mask[peaks]]
        troughs = troughs[valid_mask[troughs]]
    except:
        return {
            'withdraw_ranges': [],
            'reach_ranges': [],
            'withdraw_durations': [],
            'reach_durations': [],
            'cycle_durations': [],
            'cycles': []
        }

    withdraw_phases = []
    reach_phases = []

    for i in range(len(peaks) - 1):
        start_peak = peaks[i]
        end_peak = peaks[i + 1]

        # Find troughs between current and next peak
        in_between_troughs = troughs[(troughs > start_peak) & (troughs < end_peak)]
        if in_between_troughs.size == 0:
            continue

        trough = in_between_troughs[np.argmin(y[in_between_troughs])]
        
        # Check that the segment between peak and trough is mostly valid
        segment_valid = np.sum(valid_mask[start_peak:trough+1]) / (trough - start_peak + 1)
        if segment_valid > 0.7:  # Require 70% valid data
            withdraw_phases.append((start_peak, trough))
        
        # Check reach segment
        segment_valid = np.sum(valid_mask[trough:end_peak+1]) / (end_peak - trough + 1)
        if segment_valid > 0.7:
            reach_phases.append((trough, end_peak))

    # Handle last peak
    if peaks.size > 0:
        last_peak = peaks[-1]
        trailing_troughs = troughs[troughs > last_peak]
        if trailing_troughs.size > 0:
            trough = trailing_troughs[0]
            segment_valid = np.sum(valid_mask[last_peak:trough+1]) / (trough - last_peak + 1)
            if segment_valid > 0.7:
                withdraw_phases.append((last_peak, trough))

    # Durations
    withdraw_durations = [(b - a) / fps for a, b in withdraw_phases]
    reach_durations = [(b - a) / fps for a, b in reach_phases]
    
    # Match withdraw and reach phases for cycle durations
    min_len = min(len(withdraw_durations), len(reach_durations))
    if withdraw_phases and reach_phases:
        first_withdraw = withdraw_phases[0][0]
        first_reach    = reach_phases[0][0]

        if first_withdraw <= first_reach:
            # withdraw → reach ordering
            cycles = list(zip(withdraw_phases[:min_len], reach_phases[:min_len]))
            cycle_durations = [w + r for w, r in zip(withdraw_durations[:min_len], reach_durations[:min_len])]
        else:
            # reach → withdraw ordering
            cycles = list(zip(reach_phases[:min_len], withdraw_phases[:min_len]))
            cycle_durations = [r + w for r, w in zip(reach_durations[:min_len], withdraw_durations[:min_len])]
    else:
        cycles = []
        cycle_durations = []

    
    cycles = list(zip(withdraw_phases[:min_len], reach_phases[:min_len]))

    return {
        'withdraw_ranges': withdraw_phases,
        'reach_ranges': reach_phases,
        'withdraw_durations': withdraw_durations,
        'reach_durations': reach_durations,
        'cycle_durations': cycle_durations,
        'cycles': cycles
    }

# ===================================== #
# Symmetry / Coordination
# ===================================== #

def compute_symmetry(left_y, right_y):
    """Compute symmetry as vertical distance between hands over time (NaN-aware)"""
    symmetry = left_y - right_y
    return symmetry

def compute_symmetry_index(left_y, right_y):
    """Compute symmetry index between left and right y-trajectories (NaN-aware)"""
    valid_mask = ~np.isnan(left_y) & ~np.isnan(right_y)
    if np.sum(valid_mask) == 0:
        return np.nan
    
    abs_diff = np.abs(left_y - right_y)
    sum_vals = (np.nanmean(left_y[valid_mask]) + np.nanmean(right_y[valid_mask]))
    if sum_vals == 0:
        return np.nan
    return np.nanmean(abs_diff[valid_mask] / sum_vals)

def compute_lag(left_displacement, right_displacement, fps):
    """Temporal lag (seconds) and max correlation between two signals (NaN-aware)"""
    # Remove NaN values
    valid_mask = ~np.isnan(left_displacement) & ~np.isnan(right_displacement)
    if np.sum(valid_mask) < 10:
        return np.nan, np.nan, np.array([]), np.array([])
    
    left_valid = left_displacement[valid_mask]
    right_valid = right_displacement[valid_mask]
    
    # Normalize
    left_norm = (left_valid - np.mean(left_valid)) / np.std(left_valid)
    right_norm = (right_valid - np.mean(right_valid)) / np.std(right_valid)
    
    corr = correlate(left_norm, right_norm, mode='full') / len(left_norm)
    lags = np.arange(-len(left_norm) + 1, len(right_norm))
    max_corr = np.max(corr)
    best_lag = lags[np.argmax(corr)] / fps

    return best_lag, max_corr, lags, corr

def compute_bimanual_coordination(left_displacement, right_displacement):
    """Compute correlation coefficient between left and right displacements (NaN-aware)"""
    valid_mask = ~np.isnan(left_displacement) & ~np.isnan(right_displacement)
    if np.sum(valid_mask) < 2:
        return np.nan, np.nan, np.nan, np.nan

    left_valid = left_displacement[valid_mask]
    right_valid = right_displacement[valid_mask]
    
    if len(left_valid) < 2:
        return np.nan, np.nan, np.nan, np.nan

    r_val, p_val = pearsonr(left_valid, right_valid)
    slope, intercept = np.polyfit(left_valid, right_valid, 1)

    return r_val, p_val, slope, intercept

def compute_cyclewise_bimanual_coordination(left_cycles, right_cycles, left_displacement, right_displacement):
    """
    Compute per-cycle Pearson correlation between left and right hand displacement.

    The leading hand (earliest cycle start) defines the cycle boundaries.
    The other hand's displacement is extracted over the same frame window
    and correlated directly.
    """

    if not left_cycles and not right_cycles:
        return {
            'correlations':     [],
            'mean_correlation': np.nan,
            'std_correlation':  np.nan,
            'n_cycles':         0,
            'leading_hand':     None
        }

    # Determine leading hand by earliest cycle start frame
    left_first  = left_cycles[0][0][0]  if left_cycles  else np.inf
    right_first = right_cycles[0][0][0] if right_cycles else np.inf

    if left_first <= right_first:
        leading_hand = 'left'
        ref_cycles   = left_cycles
        ref_trace    = left_displacement
        other_trace  = right_displacement
    else:
        leading_hand = 'right'
        ref_cycles   = right_cycles
        ref_trace    = right_displacement
        other_trace  = left_displacement

    correlations = []

    for (w_start, w_end), (r_start, r_end) in ref_cycles:
        start = w_start
        end   = r_end

        # Skip if cycle is too short or out of bounds
        if end - start < 5:
            continue
        if end >= min(len(ref_trace), len(other_trace)):
            continue

        ref_seg   = ref_trace[start:end + 1]
        other_seg = other_trace[start:end + 1]

            # Only keep frames where both hands are visible
        valid = ~np.isnan(ref_seg) & ~np.isnan(other_seg)
        if np.sum(valid) < 0.7 * len(ref_seg):
            continue

        r, _ = pearsonr(ref_seg[valid], other_seg[valid])
        if not np.isnan(r):
            correlations.append(r)

    if not correlations:
        return {
            'correlations':     [],
            'mean_correlation': np.nan,
            'std_correlation':  np.nan,
            'n_cycles':         0,
            'leading_hand':     leading_hand
        }

    return {
        'correlations':     correlations,
        'mean_correlation': float(np.mean(correlations)),
        'std_correlation':  float(np.std(correlations)),
        'n_cycles':         len(correlations),
        'leading_hand':     leading_hand
    }

def compute_nose_string_tracking_correlation(nose, string):
    """Compute correlation between nose and string positions (NaN-aware)"""
    # Check for valid rows (both x and y are valid)
    nose_valid = ~np.isnan(nose).any(axis=1)
    string_valid = ~np.isnan(string).any(axis=1)
    
    # Find frames where both are valid AND consecutive frames are valid
    valid_pairs = nose_valid[:-1] & nose_valid[1:] & string_valid[:-1] & string_valid[1:]
    
    if np.sum(valid_pairs) < 2:
        return np.array([]), np.array([]), np.nan, np.nan
    
    # Compute displacements only for valid consecutive pairs
    nose_disp_full = np.linalg.norm(np.diff(nose, axis=0), axis=1)
    string_disp_full = np.linalg.norm(np.diff(string, axis=0), axis=1)
    
    nose_displacement = nose_disp_full[valid_pairs]
    string_displacement = string_disp_full[valid_pairs]

    if len(nose_displacement) < 2:
        return nose_displacement, string_displacement, np.nan, np.nan

    r, p = pearsonr(nose_displacement, string_displacement)
    r_squared = r**2

    return nose_displacement, string_displacement, r_squared, p

# ===================================== #
# Body Compensation
# ===================================== #

def compute_body_recruitment(body_displacement, hand_displacement, phase_ranges, norm_factor):
    """
    Measure how much body moves relative to hand during pull phase (NaN-aware).
    High ratio indicates high torso motion during pull.
    """
    ratios = []
    for start, end in phase_ranges:
        if start < end <= min(len(body_displacement), len(hand_displacement)):
            body_seg = body_displacement[start:end]
            hand_seg = hand_displacement[start:end]
            
            # Compute movement only for valid consecutive pairs
            body_valid = ~np.isnan(body_seg[:-1]) & ~np.isnan(body_seg[1:])
            hand_valid = ~np.isnan(hand_seg[:-1]) & ~np.isnan(hand_seg[1:])
            
            if np.sum(body_valid) > 0 and np.sum(hand_valid) > 0:
                body_move = np.nansum(np.abs(np.diff(body_seg)[body_valid]))
                hand_move = np.nansum(np.abs(np.diff(hand_seg)[hand_valid]))
            
                if norm_factor and not np.isnan(norm_factor):
                    body_move /= norm_factor
                    hand_move /= norm_factor
                
                if hand_move > 0:
                    ratios.append(body_move / hand_move)
                else:
                    ratios.append(np.nan)
            else:
                ratios.append(np.nan)
        else:
            ratios.append(np.nan)
    
    if len(ratios) == 0 or all(np.isnan(ratios)):
        return {
            'phase_ratios': ratios,
            'mean_ratio': np.nan,
            'cv_ratio': np.nan
        }
    
    mean_ratio = np.nanmean(ratios)
    cv_ratio = np.nanstd(ratios) / mean_ratio if mean_ratio > 0 else np.nan
    
    return {
        'phase_ratios': ratios,
        'mean_ratio': mean_ratio,
        'cv_ratio': cv_ratio
    }