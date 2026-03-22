import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
import os
import re
#from compute_metrics import get_for_plotting
from stringpullkit.analysis.compute_metrics import get_for_plotting

LEFT_COLOR = "#FF10F0" # neon pink, for general left and reach
RIGHT_COLOR = "#1F51FF" # neon blue, for general right and reach
LEFT_COLOR_2 = "#9000A2" # darker pink, for left withdraw
RIGHT_COLOR_2 = "#00008B" # darker blue, for right withdraw
REACH_COLOR = '#EFBF04'
WITHDRAW_COLOR = '#008080'

CYCLE_COLORS = {"Left Hand": LEFT_COLOR, "Right Hand": RIGHT_COLOR}
PHASE_COLORS = {"Left Reach": LEFT_COLOR, "Right Reach": RIGHT_COLOR, "Left Withdraw": LEFT_COLOR_2, "Right Withdraw": RIGHT_COLOR_2}
PHASE_COLORS_NO_HANDS = {"Reach": REACH_COLOR, 'Withdraw': WITHDRAW_COLOR}

# ============================ #
# Helper Functions
# ============================ #

def save_fig(session, save_name, show_plot):
    if getattr(session, 'save_dir', None):
        save_path = os.path.join(session.save_dir, save_name)
        plt.savefig(save_path)
        
    if show_plot:
        plt.show()
    else:
        plt.close() 

def collect_session_variables(session):
    fps = session.fps
    scale = session.scale_factor
    label_units = "mm" if scale is not None else "px"

    return fps, label_units

# =================================================================================== #
# Plotting Functions #
# =================================================================================== #

# + + + + + + + + + + + + + + #
# Group 1: Head Metrics
# + + + + + + + + + + + + + + #

def plot_head_torso_metrics(session, show_plot=True):
    """Plot yaw, pitch and roll (head and torso metrics) over time."""
    print("Plotting head / torso metrics...")

    yaw = get_for_plotting(session, 'head_yaw')
    pitch = get_for_plotting(session, 'head_pitch')
    roll = get_for_plotting(session, 'head_roll')

    body_yaw = get_for_plotting(session, 'upper_torso_yaw')
    body_pitch = get_for_plotting(session, 'upper_torso_pitch')
    body_roll = get_for_plotting(session, 'upper_torso_roll')

    frames = np.arange(len(yaw))
    _, label_units = collect_session_variables(session)
    
    fig = plt.figure(figsize=(16, 10))

    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1])
    ax1_left = fig.add_subplot(gs[0, 0])  
    ax1_right = fig.add_subplot(gs[0, 1])  
    ax2_left = fig.add_subplot(gs[1, 0])  
    ax2_right = fig.add_subplot(gs[1, 1])  
    ax3_left = fig.add_subplot(gs[2, 0])
    ax3_right = fig.add_subplot(gs[2, 1])

    axs = [ax1_left, ax1_right, ax2_left, ax2_right, ax3_left, ax3_right]

    metrics = [yaw, body_yaw, pitch, body_pitch, roll, body_roll]
    titles = ["Head Yaw", "Upper Torso Yaw", " Head Pitch", "Upper Torso Pitch", "Head Roll", "Upper Torso Roll"]
    colors = ["#B388FF", "#B388FF", "#9C27B0", "#9C27B0", "#6A1B9A", "#6A1B9A"] # shades of purple

    for ax, metric, title, color in zip(axs, metrics, titles, colors):
        ax.plot(frames, metric, color=color, label=title)
        ax.axhline(0, color="black", linestyle='--', alpha=0.7)
        ax.set_ylabel("Log Ratio" if "yaw" in title.lower() else "Angle (degrees)")
        ax.set_title(f"{title}")
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axs[-1].set_xlabel("Frame")
    fig.suptitle("Head Yaw, Pitch, and Roll")
    plt.tight_layout()

    save_fig(session, f"head_torso_metrics.png", show_plot)

# + + + + + + + + + + + + + + #
# Group 2: Bimanual Coordination
# + + + + + + + + + + + + + + #

def plot_bimanual_coordination(session, show_plot=True):
    print("Plotting bimanual coordination metrics...")

    fps, label_units = collect_session_variables(session)
    fig = plt.figure(figsize=(10,12))

    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 1])
    ax1_left = fig.add_subplot(gs[0, 0])  
    ax1_right = fig.add_subplot(gs[0, 1])  
    ax2 = fig.add_subplot(gs[1, :])  
    ax3 = fig.add_subplot(gs[2, :])  
    ax4_left = fig.add_subplot(gs[3, 0])
    ax4_right = fig.add_subplot(gs[3, 1])

    axs = [ax1_left, ax1_right, ax2, ax3, ax4_left, ax4_right]

    # 1_1. Inter-Hand Distance
    ihd = get_for_plotting(session, 'interhand_distance')
    frames = np.arange(len(ihd))
    ax1_left.plot(frames, ihd, color='purple', label="Inter-Hand Distance")
    ax1_left.set_xlabel("Frame")
    ax1_left.set_ylabel(f"Euclidean Distance ({label_units})")
    ax1_left.set_title("Inter-Hand Distance")

    # 1_2. Inter-Hand Distance By Phase
    withdraw_ihd = (session.metrics['interhand_distance_withdraw'])
    reach_ihd = (session.metrics['interhand_distance_reach'])

    data = np.concatenate([reach_ihd, withdraw_ihd])
    labels = (["Reach"] * len(reach_ihd)) + (["Withdraw"] * len(withdraw_ihd))
    colors = {"Reach": REACH_COLOR, "Withdraw": WITHDRAW_COLOR}
    
    sns.boxplot(x=labels, y=data, ax=ax1_right, color='white', linecolor='black', linewidth=2, width=0.3)
    sns.stripplot(x=labels, y=data, ax=ax1_right, hue=labels, palette=colors, size=10, jitter=0.15, alpha=0.7)

    ax1_right.set_xlabel("")
    ax1_right.set_ylabel(f"Euclidean Distance ({label_units})")
    ax1_right.set_title("Inter-Hand Distance By Phase")

    # 2. Symmetry
    symmetry = get_for_plotting(session, 'symmetry')
    frames = np.arange(len(symmetry))

    ax2.plot(frames, symmetry, color='purple', label="Left Y Position - Right Y Position")
    ax2.axhline(0, color='black', linestyle='--', alpha=0.7)
    ax2.set_xlabel("Frame")
    ax2.set_ylabel(f"Symmetry ({label_units})")
    ax2.set_title("Hand Vertical Position Symmetry")
    ax2.legend()
    
    # 3. Bimanual Displacement
    left_displacement = get_for_plotting(session, 'hand_l_displacement')
    right_displacement = get_for_plotting(session, 'hand_r_displacement')
    frames = np.arange(len(left_displacement))

    ax3.plot(frames, left_displacement, label='Left Hand Vertical Displacement', color=LEFT_COLOR)
    ax3.plot(frames, right_displacement, label='Right Hand Vertical Displacement', color=RIGHT_COLOR)
    ax3.axhline(0, color='lightgray', linestyle='--', alpha=0.8)
    ax3.set_xlabel('Frames')
    ax3.set_ylabel(f'Displacement ({label_units})')
    ax3.set_title('Bimanual Displacement')
    ax3.legend()

    # 4_1. Bimanual Coordination
    r_value = session.metrics['bimanual_coordination']['r_value']
    p_value = session.metrics['bimanual_coordination']['p_value']
    slope = session.metrics['bimanual_coordination']['slope']
    intercept =session.metrics['bimanual_coordination']['intercept']

    clean_p =  "p < 0.001" if p_value < 0.001 else f"p = {p_value:.4f}"
 
    x_fit = np.linspace(np.nanmin(left_displacement), np.nanmax(left_displacement), 100)
    y_fit = slope * x_fit + intercept

    ax4_left.scatter(left_displacement, right_displacement, alpha=0.7, color='purple', s=10, label='Displacement Data')
    ax4_left.plot(x_fit, y_fit, color='black', linestyle='-', label='Regression Line')
    ax4_left.set_ylabel(f'Left Hand Vertical Displacement ({label_units})')
    ax4_left.set_xlabel(f'Right Hand Vertical Displacement ({label_units})')
    ax4_left.set_title(f'Bimanual Coordination\nr = {r_value:.2f}, {clean_p}')
    ax4_left.legend()

    # 4_2. Cross Correlation Lag
    cross_correlation = session.metrics['cross_correlation']
    best_lag = cross_correlation['best_lag']
    max_correlation = cross_correlation['max_correlation']
    lags = cross_correlation['lags']
    correlation = cross_correlation['correlation']
    
    ax4_right.plot(lags, correlation, label='Hand Position Cross Correlation')
    ax4_right.axvline(x=0, color='black', alpha=0.8, linestyle='--', label='Zero Lag')
    ax4_right.axvline(x=best_lag * fps, color='purple', linestyle='-', label=f"Peak Lag = {best_lag:.2f} s")
    ax4_right.set_xlabel("Lag (Frames)")
    ax4_right.set_ylabel("Cross Correlation")
    ax4_right.set_title("Left-Right Lag Analysis")
    ax4_right.legend()

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
    
    fig.suptitle("Bimanual Coordination")
    plt.tight_layout()

    save_fig(session, 'bimanual_coordination.png', show_plot)

# + + + + + + + + + + + + + + #
# Group 3: Hand Kinematics
# + + + + + + + + + + + + + + #

def plot_hand_kinematics(session, show_plot=True):
    """Plot velocity, acceleration, jerk, and mean/peak speed by phase. """
    print("Plotting hand kinematics...")

    fps, label_units = collect_session_variables(session)
    fig = plt.figure(figsize=(10,12))

    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 1])
    ax1 = fig.add_subplot(gs[0, :])   
    ax2 = fig.add_subplot(gs[1, :]) 
    ax3 = fig.add_subplot(gs[2, :]) 
    ax4_left = fig.add_subplot(gs[3, 0])  
    ax4_right = fig.add_subplot(gs[3, 1])
 
    axs = [ax1, ax2, ax3, ax4_left, ax4_right]
    
    # Preparing Kinematics
    left_y = session.metrics['hand_l_y_trajectory'] # Trajectories stored in session.metrics are smoothed
    right_y = session.metrics['hand_r_y_trajectory']
    l_vel = get_for_plotting(session, 'hand_l_velocity')
    l_acc = get_for_plotting(session, 'hand_l_acceleration')
    l_jerk = get_for_plotting(session, 'hand_l_jerk')
    r_vel = get_for_plotting(session, 'hand_r_velocity')
    r_acc = get_for_plotting(session, 'hand_r_acceleration')
    r_jerk = get_for_plotting(session, 'hand_r_jerk')

    l_mean_reach_speed = session.phase_metrics['left_reach']['mean_speed']
    r_mean_reach_speed = session.phase_metrics['right_reach']['mean_speed']
    l_mean_withdraw_speed = session.phase_metrics['left_withdraw']['mean_speed']
    r_mean_withdraw_speed = session.phase_metrics['right_withdraw']['mean_speed']

    mean_speed_data = np.concatenate([l_mean_reach_speed, r_mean_reach_speed, l_mean_withdraw_speed, r_mean_withdraw_speed])
    mean_speed_labels = ((["Left Reach"] * len(l_mean_reach_speed)) + (["Right Reach"] *len(r_mean_reach_speed)) + 
    (["Left Withdraw"] * len(l_mean_withdraw_speed)) + (['Right Withdraw'] * len(r_mean_withdraw_speed)))

    l_peak_reach_speed = session.phase_metrics['left_reach']['peak_speed']
    r_peak_reach_speed = session.phase_metrics['right_reach']['peak_speed']
    l_peak_withdraw_speed = session.phase_metrics['left_withdraw']['peak_speed']
    r_peak_withdraw_speed = session.phase_metrics['right_withdraw']['peak_speed']

    peak_speed_data = np.concatenate([l_peak_reach_speed, r_peak_reach_speed, l_peak_withdraw_speed, r_peak_withdraw_speed])
    peak_speed_labels = ((["Left Reach"] * len(l_peak_reach_speed)) + (["Right Reach"] *len(r_peak_reach_speed)) + 
    (["Left Withdraw"] * len(l_peak_withdraw_speed)) + (['Right Withdraw'] * len(r_peak_withdraw_speed)))

    colors = PHASE_COLORS

    # 1. Velocity
    frames = np.arange(len(l_vel))
    ax1.plot(frames, l_vel, color=LEFT_COLOR, label=f"Left Velocity ({label_units}/s)")
    ax1.plot(frames, r_vel, color=RIGHT_COLOR, label=f"Right Velocity ({label_units}/s)")
    ax1.set_ylabel(f"Velocity ({label_units}/s)")
    ax1.legend()

    # 2. Acceleration
    ax2.plot(frames, l_acc, color=LEFT_COLOR, label=f"Left Acceleration ({label_units}/s²)")
    ax2.plot(frames, r_acc, color=RIGHT_COLOR, label=f"Right Acceleration ({label_units}/s²)")
    ax2.set_ylabel(f"Acceleration ({label_units}/s²)")
    ax2.legend()
    
    # 3. Jerk 
    ax3.plot(frames, l_jerk, color=LEFT_COLOR, label=f"Left Jerk ({label_units}/s³)")
    ax3.plot(frames, r_jerk, color=RIGHT_COLOR, label=f"Right Jerk ({label_units}/s³)")
    ax3.set_ylabel(f"Jerk ({label_units}/s³)")
    ax3.set_xlabel("Frame")
    ax3.legend()

    # 4_1. Mean Speed By Phase
    sns.boxplot(x=mean_speed_labels, y=mean_speed_data, ax=ax4_left, color='white', linecolor='black', linewidth=2, width=0.3)
    sns.stripplot(x=mean_speed_labels, y=mean_speed_data, ax=ax4_left, hue=mean_speed_labels, palette=colors, size=10, jitter=0.15, alpha=0.8)

    ax4_left.set_xlabel("")
    ax4_left.set_ylabel(f"Speed ({label_units}/s)")
    ax4_left.set_title("Mean Hand Speed By Phase")

    # 4_2. Peak Speed By Phase
    sns.boxplot(x=peak_speed_labels, y=peak_speed_data, ax=ax4_right, color='white', linecolor='black', linewidth=2, width=0.3)
    sns.stripplot(x=peak_speed_labels, y=peak_speed_data, ax=ax4_right, hue=peak_speed_labels, palette=colors, size=10, jitter=0.15, alpha=0.8)

    ax4_right.set_xlabel("")
    ax4_right.set_ylabel(f"Speed ({label_units}/s)")
    ax4_right.set_title("Peak Hand Speed By Phase")

    for ax in axs:
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig.suptitle("Hand Kinematics")
    plt.tight_layout()

    save_fig(session, "hand_kinematics.png", show_plot)

# + + + + + + + + + + + + + + #
# Group 4: Cycle Metrics
# + + + + + + + + + + + + + + #

def plot_cycle_phase_metrics(session, show_plot=True):
    """ Plot cycle duration and phase component durations and displacement"""
    print("Plotting cycle / phase metrics...")
    _, label_units = collect_session_variables(session)

    fps, label_units = collect_session_variables(session)
    fig = plt.figure(figsize=(12,12))

    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1])
    ax1_left = fig.add_subplot(gs[0, 0])   
    ax1_right = fig.add_subplot(gs[0, 1]) 
    ax2_left = fig.add_subplot(gs[1, 0]) 
    ax2_right = fig.add_subplot(gs[1, 1])  
   
    
    axs = [ax1_left, ax1_right, ax2_left, ax2_right]

    # 1. Cycle Duration
    left_cd = session.metrics['hand_l_cycle_duration']
    right_cd = session.metrics['hand_r_cycle_duration']
    data = left_cd + right_cd
    labels = (["Left Hand"] * len(left_cd) + ["Right Hand"] * len(right_cd))

    sns.boxplot(x=labels, y=data, ax=ax1_left, color='white', linecolor='black', linewidth=2, width=0.3)
    sns.stripplot(x=labels, y=data, ax=ax1_left, hue=labels, palette=CYCLE_COLORS, size=10, jitter=0.15, alpha=0.6)
    
    ax1_left.set_xlabel("")
    ax1_left.set_ylabel(f"Duration (s)")
    ax1_left.set_title("Cycle Duration")

    # 2. Phase Component Durations
    left_reach_duration = session.phase_metrics['left_reach']['duration']
    right_reach_duration = session.phase_metrics['right_reach']['duration']
    left_withdraw_duration = session.phase_metrics['left_withdraw']['duration']
    right_withdraw_duration = session.phase_metrics['right_withdraw']['duration']

    data = left_reach_duration + right_reach_duration + left_withdraw_duration + right_withdraw_duration
    labels_2 = (["Left Reach"] * len(left_reach_duration) + ["Right Reach"] * len(right_reach_duration) +
    ["Left Withdraw"] * len(left_withdraw_duration) + ["Right Withdraw"] * len(right_withdraw_duration))
    
    sns.boxplot(x=labels_2, y=data, ax=ax1_right, color='white', linecolor='black', linewidth=2, width=0.3)
    sns.stripplot(x=labels_2, y=data, ax=ax1_right, hue=labels_2, palette=PHASE_COLORS, size=10, jitter=0.15, alpha=0.6)

    ax1_right.set_xlabel("")
    ax1_right.set_ylabel(f"Duration (s)")
    ax1_right.set_title("Phase Component Duration")

    # Hand Displacement Phases
    withdraw_dx = session.metrics['hand_withdraw_displacement_dx']
    withdraw_dy = session.metrics['hand_withdraw_displacement_dy']
    withdraw_euclidean = session.metrics['hand_withdraw_displacement_euclidean']

    reach_dx = session.metrics['hand_reach_displacement_dx']
    reach_dy = session.metrics['hand_reach_displacement_dy']
    reach_euclidean = session.metrics['hand_reach_displacement_euclidean']

    disp_data = {
        "Displacement Type": (["ΔX"] * (len(reach_dx) + len(withdraw_dx)) +
                              ["ΔY"] * (len(reach_dy) + len(withdraw_dy)) +
                              ["Euclidean"] * (len(reach_euclidean) + len(withdraw_euclidean))),
        "Value": np.concatenate([
            reach_dx, withdraw_dx,
            reach_dy, withdraw_dy,
            reach_euclidean, withdraw_euclidean
        ]),
        "Phase": (["Reach"] * len(reach_dx) +
                  ["Withdraw"] * len(withdraw_dx) +
                  ["Reach"] * len(reach_dy) +
                  ["Withdraw"] * len(withdraw_dy) +
                  ["Reach"] * len(reach_euclidean) +
                  ["Withdraw"] * len(withdraw_euclidean))
    }

    df_disp = pd.DataFrame(disp_data)

    box_palette = {"Reach": 'white', "Withdraw": 'white'}
    sns.boxplot(
        data=df_disp, x="Displacement Type", y="Value", hue="Phase",
        ax=ax2_left, palette=box_palette, linecolor='black', linewidth=2, width=0.7, legend=False
    )
    sns.stripplot(
        data=df_disp, x="Displacement Type", y="Value", hue="Phase",
        ax=ax2_left, palette=PHASE_COLORS_NO_HANDS, size=8, alpha=0.6, jitter=0.15, dodge=True, 
    )

    ax2_left.legend()

    ax2_left.set_title("Hand Displacement by Phase")
    ax2_left.set_ylabel(f"Displacement ({label_units})")
    ax2_left.set_xlabel("")

    # Nose Displacement Phases
    nose_withdraw_dx = session.metrics[f'nose_withdraw_displacement_dx']
    nose_withdraw_dy = session.metrics[f'nose_withdraw_displacement_dy']
    nose_withdraw_euclidean = session.metrics[f'nose_withdraw_displacement_euclidean']

    nose_reach_dx = session.metrics[f'nose_reach_displacement_dx']
    nose_reach_dy = session.metrics[f'nose_reach_displacement_dy']
    nose_reach_euclidean = session.metrics[f'nose_reach_displacement_euclidean']
    nose_disp_data = {
        "Displacement Type": (["ΔX"] * (len(nose_reach_dx) + len(nose_withdraw_dx)) +
                              ["ΔY"] * (len(nose_reach_dy) + len(nose_withdraw_dy)) +
                              ["Euclidean"] * (len(nose_reach_euclidean) + len(nose_withdraw_euclidean))),
        "Value": np.concatenate([
            nose_reach_dx, nose_withdraw_dx,
            nose_reach_dy, nose_withdraw_dy,
            nose_reach_euclidean, nose_withdraw_euclidean
        ]),
        "Phase": (["Reach"] * len(nose_reach_dx) +
                  ["Withdraw"] * len(nose_withdraw_dx) +
                  ["Reach"] * len(nose_reach_dy) +
                  ["Withdraw"] * len(nose_withdraw_dy) +
                  ["Reach"] * len(nose_reach_euclidean) +
                  ["Withdraw"] * len(nose_withdraw_euclidean))
    }

    df_nose_disp = pd.DataFrame(nose_disp_data)

    # Grouped box + strip plot
    sns.boxplot(
        data=df_nose_disp, x="Displacement Type", y="Value", hue="Phase",
        ax=ax2_right, palette=box_palette, linecolor='black', linewidth=2, width=0.7, legend=False
    )
    sns.stripplot(
        data=df_nose_disp, x="Displacement Type", y="Value", hue="Phase",
        ax=ax2_right, palette=PHASE_COLORS_NO_HANDS, size=8, alpha=0.6, jitter=0.15, dodge=True, 
    )

    ax2_right.legend()

    ax2_right.set_title("Nose Displacement by Phase")
    ax2_right.set_ylabel(f"Displacement {label_units}")
    ax2_right.set_xlabel("")

    for ax in axs:
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig.suptitle("Cycle Metrics")
    plt.tight_layout()

    save_fig(session, "cycle_phase_metrics.png", show_plot)

# + + + + + + + + + + + + + + #
# Group 5: Hand Trajectories
# + + + + + + + + + + + + + + #

def plot_hand_trajectories(session, show_plot=True):
    """Plot hand x, y trajectories, and phase trajectories"""
    print("Plotting hand trajectories...")
    _, label_units = collect_session_variables(session)

    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=False)

    # 1–2. Hand Trajectories (Horizontal and Vertical)
    trajectories = {
        "Horizontal": ("hand_l_x_trajectory", "hand_r_x_trajectory", "Horizontal Position"),
        "Vertical": ("hand_l_y_trajectory", "hand_r_y_trajectory", "Vertical Position"),
    }

    for i, (name, (left_key, right_key, ylabel)) in enumerate(trajectories.items(), start=1):
        ax = axs[i-1]  # assumes ax1 = axs[0], ax2 = axs[1]
        left = session.metrics[left_key]
        right = session.metrics[right_key]
        frames = np.arange(len(left))

        ax.plot(frames, left, color=LEFT_COLOR, label="Left Hand")
        ax.plot(frames, right, color=RIGHT_COLOR, label="Right Hand")
        ax.set_ylabel(f"{ylabel} ({label_units})")
        ax.set_xlabel("Frame")
        ax.set_title(f"{name} Trajectory")

    # 3-4. Phase Trajectories (Left and Right)
    hands = ["left", "right"]

    for i, hand in enumerate(hands, start=3):
        ax = axs[i-1]
        trajectory = session.metrics[f"hand_{hand[0]}_y_trajectory"]
        reach_phases = session.phase_metrics[f"{hand}_reach"]["ranges"]
        withdraw_phases = session.phase_metrics[f"{hand}_withdraw"]["ranges"]
        strokes = session.metrics[f'hand_{hand[0]}_strokes']
        frames = np.arange(len(trajectory))

        ax.plot(frames, trajectory, color='lightgray', alpha=0.8, label="Vertical Trajectory")
        for stroke in strokes:
            ax.axvline(x=stroke, color='magenta', linestyle='--', alpha=0.7)

        # Withdraw Phases
        for j, (start, end) in enumerate(withdraw_phases):
            idx_range = np.arange(start, end)
            ax.plot(idx_range, trajectory[idx_range],
             color=PHASE_COLORS[f"{hand.title()} Withdraw"], label="Withdraw" if j == 0 else "")
        
        for j, (start, end) in enumerate(reach_phases):
            idx_range = np.arange(start, end)
            ax.plot(idx_range, trajectory[idx_range],
             color=PHASE_COLORS[f"{hand.title()} Reach"], label="Reach" if j == 0 else "")
            
        ax.set_ylabel(f"Y Position ({label_units})")
        ax.set_xlabel("Frame")
        ax.set_title(f"{hand.title()} Hand Phase Trajectory")

    for ax in axs:
        ax.legend()
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig.suptitle("Hand Trajectories")
    plt.tight_layout()

    save_fig(session, "hand_trajectories.png", show_plot)

# + + + + + + + + + + + + + + #
# Group 6: Path Descriptives
# + + + + + + + + + + + + + + #

def plot_path_descriptives(session, show_plot):
    """Heading Directions, Plot Directional Concentration, Stroke Amplitude, 
    Movement Scaling Correlation, Path Circuity, and Topographic Representation"""
    print("Plotting path descriptives...")
    _, label_units = collect_session_variables(session)

    fig = plt.figure(figsize=(12, 12))
    colors = [PHASE_COLORS["Left Reach"], PHASE_COLORS["Right Reach"], 
            PHASE_COLORS["Left Withdraw"], PHASE_COLORS["Right Withdraw"]]


    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1])
    ax1_left = fig.add_subplot(gs[0, 0], projection='polar') 
    ax1_right = fig.add_subplot(gs[0, 1])  
    ax2_left = fig.add_subplot(gs[1, 0]) 
    ax2_right = fig.add_subplot(gs[1, 1])
    ax3_left = fig.add_subplot(gs[2, 0]) 
    ax3_right = fig.add_subplot(gs[2, 1])

    axs = [ax1_left, ax1_right, ax2_left, ax2_right, ax3_left, ax3_right]

    # 1_1. Heading Directions
    bins = 18
    for phase, key in zip(
        ["Left Reach", "Right Reach", "Left Withdraw", "Right Withdraw"],
        ["left_reach", "right_reach", "left_withdraw", "right_withdraw"]):

        angles = np.radians(session.phase_metrics[key]['heading_direction'])
        ax1_left.hist(angles, bins=bins, alpha=0.8, color=PHASE_COLORS[phase], label=phase)

    ax1_left.set_theta_zero_location("E")
    ax1_left.set_theta_direction(-1)
    ax1_left.set_title("Heading Directions", pad=30)
    
    # 1_2. Directional Concentration
    concentrations = [
        session.phase_metrics['left_reach']['heading_concentration'],
        session.phase_metrics['right_reach']['heading_concentration'],
        session.phase_metrics['left_withdraw']['heading_concentration'],
        session.phase_metrics['right_withdraw']['heading_concentration']
    ]
    labels = ["Left Reach", "Right Reach", "Left Withdraw", "Right Withdraw"]
    ax1_left.set_yticks([])  # Removes radial tick labels

    ax1_right.scatter(labels, concentrations, color=colors, s=120)
    ax1_right.set_title("Directional Concentration")
    ax1_right.set_ylabel("Heading Concentration (R̄)")
    for x, y in zip(labels, concentrations):
        ax1_right.text(x, y + 0.05 * np.sign(y), f"{y:.2f}", ha='center', va='bottom', fontsize=10)
    ax1_right.set_ylim(0, 1.075)

    # 2_1. Stroke Amplitude
    left_sa = session.metrics['hand_l_stroke_amplitude']
    right_sa = session.metrics['hand_r_stroke_amplitude']
    data = left_sa + right_sa
    sa_labels = ["Left Hand"] * len(left_sa) + ["Right Hand"] * len(right_sa)
    sns.boxplot(x=sa_labels, y=data, ax=ax2_left, color='white', linewidth=2, width=0.3)
    sns.stripplot(x=sa_labels, y=data, ax=ax2_left, hue=sa_labels,
                  palette=CYCLE_COLORS, size=10, jitter=0.15, alpha=0.6, dodge=False)
    ax2_left.set_xlabel("")
    ax2_left.set_ylabel(f"Amplitude ({label_units})")
    ax2_left.set_title("Stroke Amplitude")

    # 2_2. Movement Scaling Concentration
    msc = [
        session.phase_metrics['left_reach']['movement_scaling_correlation'],
        session.phase_metrics['right_reach']['movement_scaling_correlation'],
        session.phase_metrics['left_withdraw']['movement_scaling_correlation'],
        session.phase_metrics['right_withdraw']['movement_scaling_correlation']
    ]
    ax2_right.scatter(labels, msc, color=colors, s=120)
    ax2_right.set_ylim(-1, 1.09)
    for x, y in zip(labels, msc):
        ax2_right.text(x, y + 0.05, f"{y:.2f}", ha='center', va='bottom', fontsize=10)
    ax2_right.set_ylabel("Correlation (r)")
    ax2_right.set_title("Movement Scaling Correlation (Euclidean Dist. vs Peak Speed)")
    ax2_right.axhline(0, color='lightgray', alpha=0.8, linestyle='--')
    
    # 3_1. Path Circuity
    left_reach_circuity = session.phase_metrics['left_reach']['circuity']
    right_reach_circuity = session.phase_metrics['right_reach']['circuity']
    left_withdraw_circuity = session.phase_metrics['left_withdraw']['circuity']
    right_withdraw_circuity = session.phase_metrics['right_withdraw']['circuity']

    circuity_data = np.concatenate([left_reach_circuity, right_reach_circuity, left_withdraw_circuity, right_withdraw_circuity])
    circuity_labels = (["Left Reach"] * len(left_reach_circuity) + ["Right Reach"] * len(right_reach_circuity)
             + ["Left Withdraw"] * len(left_withdraw_circuity) + ["Right Withdraw"] * len(right_withdraw_circuity))
    
    sns.boxplot(x=circuity_labels, y=circuity_data, ax=ax3_left, color='white', linecolor='black', linewidth=2, width=0.3)
    sns.stripplot(x=circuity_labels, y=circuity_data, ax=ax3_left, hue=circuity_labels, palette=PHASE_COLORS, size=10, jitter=0.15, alpha=0.6)

    ax3_left.set_title("Path Circuity")
    ax3_left.set_ylabel("Circuity (Path Length / Euclidean Distance)")
    ax3_left.set_xlabel("")

    # 3_2. Topographic Representation
    path_sets = [("Left Reach", "left_reach"), ("Right Reach", "right_reach"),
        ("Left Withdraw", "left_withdraw"), ("Right Withdraw", "right_withdraw")]

    for label, key in path_sets:
        x, y = session.metrics[f"hand_{label.split()[0][0].lower()}_x_trajectory"], session.metrics[f"hand_{label.split()[0][0].lower()}_y_trajectory"]
        for start, end in session.phase_metrics[key]['ranges']:
            x_seg = x[start:end + 1] - x[start]
            y_seg = y[start:end + 1] - y[start]
            ax3_right.plot(x_seg, y_seg, color=PHASE_COLORS[label], alpha=0.6)
            ax3_right.scatter(x_seg[-1], y_seg[-1], color=PHASE_COLORS[label], s=15)

    ax3_right.axhline(0, color='lightgray', linestyle='--')
    ax3_right.axvline(0, color='lightgray', linestyle='--')
    ax3_right.set_title("Topographic Representation")
    ax3_right.set_xlabel(f"ΔX ({label_units})")
    ax3_right.set_ylabel(f"ΔY ({label_units})")
    ax3_right.axis("equal")

    for ax in axs:
        ax.grid(False)
        if 'top' in ax.spines:
            ax.spines['top'].set_visible(False)
        if 'right' in ax.spines:
            ax.spines['right'].set_visible(False)

    fig.suptitle("Path Descriptives")
    plt.tight_layout()
    save_fig(session, "path_descriptives.png", show_plot)

# + + + + + + + + + + + + + + #
# Group 7: Nose Tracking
# + + + + + + + + + + + + + + #

def plot_nose_tracking(session, show_plot=True):
    """Plot Nose Trajectory, Hand-Nose Distance, Nose-String Distance, Nose-String Correlation"""
    print("Plotting nose tracking...")
    _, label_units = collect_session_variables(session)
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=False)

    # 1. Nose Vertical Trajectory
    nose_y = session.metrics['nose_y_trajectory']
    frames = np.arange(len(nose_y))

    axs[0].plot(frames, nose_y, color='orange', label="Vertical Position")
    axs[0].set_title("Nose Vertical Trajectory")
    axs[0].set_ylabel(f"Vertical Position ({label_units})")
    axs[0].set_xlabel("Frame")
    axs[0].legend()

    # 2. Hand-Nose Distance
    left_hnd = get_for_plotting(session, 'hand_l_nose_distance')
    right_hnd = get_for_plotting(session, 'hand_r_nose_distance')
    min_len = min(len(left_hnd), len(right_hnd))
    frames = np.arange(min_len)

    axs[1].plot(frames, left_hnd[:min_len], label="Left Hand-Nose Distance", color=LEFT_COLOR)
    axs[1].plot(frames, right_hnd[:min_len], label="Right Hand-Nose Distance", color=RIGHT_COLOR)
    axs[1].axhline(0, color='black', linestyle='--', alpha=0.8)

    axs[1].set_title("Hand-Nose Distance")
    axs[1].set_ylabel(f"Distance ({label_units})")
    axs[1].set_xlabel("Frame")
    axs[1].legend()

    # 3. Nose-String Distance
    nose_string_distance = get_for_plotting(session, 'nose_string_distance')
    frames = np.arange(len(nose_string_distance))

    axs[2].plot(frames, nose_string_distance, label="Nose-String Distance", color='teal')
    axs[2].set_title("Nose-String Distance")
    axs[2].set_ylabel(f"Distance ({label_units})")
    axs[2].set_xlabel("Frame")
    axs[2].legend()

    # 4. Nose-String Tracking Correlation
    nose_string_tc_nose_displacement = session.metrics['nose_string_tracking_correlation']['nose_displacement'] # nose position changes by frame
    nose_string_tc_string_displacement = session.metrics['nose_string_tracking_correlation']['string_displacement'] # string ""
    nose_string_tc_r_squared = session.metrics['nose_string_tracking_correlation']['r_squared']
    nose_string_tc_p = session.metrics['nose_string_tracking_correlation']['p_value']

    sns.regplot(x=nose_string_tc_nose_displacement, y=nose_string_tc_string_displacement, ax=axs[3],
                scatter_kws={'s': 20, 'alpha': 0.6, 'color':'teal'}, line_kws={'color': 'black'})

    axs[3].set_title("Nose–String Tracking Correlation")
    axs[3].set_xlabel("Nose ΔPosition (Frame-to-Frame)")
    axs[3].set_ylabel("String ΔPosition (Frame-to-Frame)")

    p_text = "p < 0.001" if nose_string_tc_p < 0.001 else f"p = {nose_string_tc_p:.3g}"
    axs[3].text(0.05, 0.95, f"$R^2$ = {nose_string_tc_r_squared:.2f}\n{p_text}", 
            transform=axs[3].transAxes, ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='grey'))
    axs[3].axline((0, 0), slope=1, color='grey', linestyle='--', alpha=0.6)

    for ax in axs:
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig.suptitle("Nose Tracking")
    plt.tight_layout()

    save_fig(session, "nose_tracking.png", show_plot)


# + + + + + + + + + + + + + + #
# Group 8: Arms & Posture 
# + + + + + + + + + + + + + + #
def plot_postural_metrics(session, show_plot=True):
    """Plot body angle, spine curvature, body length, interfoot distance by phase, upper/lower torso roll and head-body alignment"""

    _, label_units = collect_session_variables(session)
    fig = plt.figure(figsize=(16, 10))

    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1])
    ax1_left = fig.add_subplot(gs[0, 0])
    ax1_right = fig.add_subplot(gs[0, 1])  
    ax2_left = fig.add_subplot(gs[1, 0]) 
    ax2_right = fig.add_subplot(gs[1, 1])
    ax3_left = fig.add_subplot(gs[2, 0]) 
    ax3_right = fig.add_subplot(gs[2, 1]) 

    axs = [ax1_left, ax1_right, ax2_left, ax2_right, ax3_left, ax3_right]

    # 1. Body Angle
    body_angle = get_for_plotting(session, 'body_angle')
    ax1_left.plot(body_angle, color='indigo', label='Body Angle')
    ax1_left.set_title("Body Angle")
    ax1_left.set_ylabel("Angle (degrees)")
    ax1_left.set_xlabel("Frame")
    ax1_left.legend()

    # 2. Spine Curvature
    spine_curvature = get_for_plotting(session, 'spine_curvature')
    ax1_right.plot(spine_curvature, color='green', label='Spine Curvature')
    ax1_right.set_title("Spine Curvature")
    ax1_right.set_ylabel("Curvature (degrees)")
    ax1_right.set_xlabel("Frame")
    ax1_right.legend()

    # 3. Body Length
    body_length = get_for_plotting(session, 'body_length')
    ax2_left.plot(body_length, color='brown', label="Body Length")
    ax2_left.set_title("Body Length")
    ax2_left.set_ylabel(f"Length ({label_units})")
    ax2_left.set_xlabel("Frame")
    ax2_left.legend()

    # 4. Interfoot Distance by Phase
    left_reach_interfoot = session.phase_metrics['left_reach']['interfoot_distance']
    right_reach_interfoot = session.phase_metrics['right_reach']['interfoot_distance']
    left_withdraw_interfoot = session.phase_metrics['left_withdraw']['interfoot_distance']
    right_withdraw_interfoot = session.phase_metrics['right_withdraw']['interfoot_distance']

    interfoot_data = np.concatenate([left_reach_interfoot, right_reach_interfoot, left_withdraw_interfoot, right_withdraw_interfoot])
    interfoot_labels = (["Left Reach"] * len(left_reach_interfoot) + ["Right Reach"] * len(right_reach_interfoot)
             + ["Left Withdraw"] * len(left_withdraw_interfoot) + ["Right Withdraw"] * len(right_withdraw_interfoot))
    
    sns.boxplot(x=interfoot_labels, y=interfoot_data, ax=ax2_right, color='white', linecolor='black', linewidth=2, width=0.3)
    sns.stripplot(x=interfoot_labels, y=interfoot_data, ax=ax2_right, hue=interfoot_labels, palette=PHASE_COLORS, size=10, jitter=0.15, alpha=0.6)

    ax2_right.set_title("Interfoot Distance by Phase")
    ax2_right.set_ylabel(f'Distance ({label_units})')
    ax2_right.set_xlabel("")

    # 5. Upper/Lower Torso Roll Overlay
    upper_torso_roll = get_for_plotting(session, 'upper_torso_roll')
    lower_torso_roll = get_for_plotting(session, 'lower_torso_roll')

    ax3_left.plot(upper_torso_roll, color='orangered', label="Upper Torso Roll", alpha=0.8)
    ax3_left.plot(lower_torso_roll, color='darkslategray', label="Lower Torso Roll", alpha=0.8)
    ax3_left.axhline(0, color='lightgray', linestyle="--")
    ax3_left.set_title("Upper vs Lower Torso Roll")
    ax3_left.set_ylabel("Roll (degrees)")
    ax3_left.set_xlabel("Frame")
    ax3_left.legend()

    # 6. Head-Body Alignment
    hba = get_for_plotting(session, 'head_body_alignment')
    ax3_right.plot(hba, color='maroon', label="Head-Body Alignment")
    ax3_right.set_title("Head-Body Alignment")
    ax3_right.set_ylabel("Angle (degrees)")
    ax3_right.set_xlabel("Frame")
    ax3_right.legend()
    
    for ax in axs:
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle("Postural Metrics")
    plt.tight_layout()

    save_fig(session, 'postural_metrics.png', show_plot)

def plot_arm_metrics(session, show_plot=True):
    """Plot arm position and kinematics"""
    fps, label_units = collect_session_variables(session)

    fig = plt.figure(figsize=(16,10))
    colors = PHASE_COLORS
    
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1])
    ax1_left = fig.add_subplot(gs[0, 0])
    ax1_right = fig.add_subplot(gs[0, 1])  
    ax2_left = fig.add_subplot(gs[1, 0]) 
    ax2_right = fig.add_subplot(gs[1, 1])
    ax3_left = fig.add_subplot(gs[2, 0]) 
    ax3_right = fig.add_subplot(gs[2, 1]) 

    axs = [ax1_left, ax1_right, ax2_left, ax2_right, ax3_left, ax3_right]

    # 1. Elbow Angle
    arm_l_elbow_angle = get_for_plotting(session, 'arm_l_elbow_angle')
    arm_r_elbow_angle = get_for_plotting(session, 'arm_r_elbow_angle')
    ax1_left.plot(arm_l_elbow_angle, color=LEFT_COLOR, label="Left Arm Elbow Angle")
    ax1_left.plot(arm_r_elbow_angle, color=RIGHT_COLOR, label="Right Arm Elbow Angle")

    ax1_left.set_title("Elbow Angle")
    ax1_left.set_ylabel("Angle (degrees)")
    ax1_left.set_xlabel("Frame")
    ax1_left.legend()

    # Elbow Angle Mean Speed
    l_mean_reach_speed = session.phase_metrics['left_reach']['elbow_mean_speed']
    r_mean_reach_speed = session.phase_metrics['right_reach']['elbow_mean_speed']
    l_mean_withdraw_speed = session.phase_metrics['left_withdraw']['elbow_mean_speed']
    r_mean_withdraw_speed = session.phase_metrics['right_withdraw']['elbow_mean_speed']

    mean_speed_data = np.concatenate([l_mean_reach_speed, r_mean_reach_speed, l_mean_withdraw_speed, r_mean_withdraw_speed])
    mean_speed_labels = ((["Left Reach"] * len(l_mean_reach_speed)) + (["Right Reach"] *len(r_mean_reach_speed)) + 
    (["Left Withdraw"] * len(l_mean_withdraw_speed)) + (['Right Withdraw'] * len(r_mean_withdraw_speed)))

    sns.boxplot(x=mean_speed_labels, y=mean_speed_data, ax=ax1_right, color='white', linecolor='black', linewidth=2, width=0.3)
    sns.stripplot(x=mean_speed_labels, y=mean_speed_data, ax=ax1_right, hue=mean_speed_labels, palette=colors, size=10, jitter=0.15, alpha=0.8)

    ax1_right.set_xlabel("")
    ax1_right.set_ylabel(f"Speed ({label_units}/s)")
    ax1_right.set_title("Mean Elbow Angular Speed By Phase")
         
    # 2. Shoulder Angle
    arm_l_shoulder_angle = get_for_plotting(session, 'arm_l_shoulder_angle')
    arm_r_shoulder_angle = get_for_plotting(session, 'arm_r_shoulder_angle')
    ax2_left.plot(arm_l_shoulder_angle, color=LEFT_COLOR, label="Left Arm Shoulder Angle")
    ax2_left.plot(arm_r_shoulder_angle, color=RIGHT_COLOR, label="Right Arm Shoulder Angle")

    ax2_left.set_title("Shoulder Angle")
    ax2_left.set_ylabel("Angle (degrees)")
    ax2_left.set_xlabel("Frame")
    ax2_left.legend()

    # Shoulder Angle Mean Speed 
    l_mean_reach_speed = session.phase_metrics['left_reach']['shoulder_mean_speed']
    r_mean_reach_speed = session.phase_metrics['right_reach']['shoulder_mean_speed']
    l_mean_withdraw_speed = session.phase_metrics['left_withdraw']['shoulder_mean_speed']
    r_mean_withdraw_speed = session.phase_metrics['right_withdraw']['shoulder_mean_speed']

    mean_speed_data = np.concatenate([l_mean_reach_speed, r_mean_reach_speed, l_mean_withdraw_speed, r_mean_withdraw_speed])
    mean_speed_labels = ((["Left Reach"] * len(l_mean_reach_speed)) + (["Right Reach"] *len(r_mean_reach_speed)) + 
    (["Left Withdraw"] * len(l_mean_withdraw_speed)) + (['Right Withdraw'] * len(r_mean_withdraw_speed)))

    sns.boxplot(x=mean_speed_labels, y=mean_speed_data, ax=ax2_right, color='white', linecolor='black', linewidth=2, width=0.3)
    sns.stripplot(x=mean_speed_labels, y=mean_speed_data, ax=ax2_right, hue=mean_speed_labels, palette=colors, size=10, jitter=0.15, alpha=0.8)

    ax2_right.set_xlabel("")
    ax2_right.set_ylabel(f"Speed ({label_units}/s)")
    ax2_right.set_title("Mean Shoulder Angular Speed By Phase")

    # 3. Arm Length
    arm_l_full_arm_length = get_for_plotting(session, 'arm_l_full_arm_length')
    arm_r_full_arm_length = get_for_plotting(session, 'arm_r_full_arm_length')
    ax3_left.plot(arm_l_full_arm_length, color=LEFT_COLOR, label="Left Arm Length")
    ax3_left.plot(arm_r_full_arm_length, color=RIGHT_COLOR, label="Right Arm Length")

    ax3_left.set_title("Arm Length")
    ax3_left.set_ylabel(f"Length ({label_units})")
    ax3_left.set_xlabel("Frame")
    ax3_left.legend()

    # Arm Extension Ratio
    left_ext_ratios = get_for_plotting(session, 'arm_l_extension_ratio')
    right_ext_ratios = get_for_plotting(session, 'arm_r_extension_ratio')

    ax3_right.plot(left_ext_ratios, color=LEFT_COLOR, label="Left Arm Extension Ratio")
    ax3_right.plot(right_ext_ratios, color=RIGHT_COLOR, label="Right Arm Extension Ratio")

    ax3_right.set_title("Arm Extension Ratio")
    ax3_right.set_ylabel("Extension Relative to Max")
    ax3_right.set_xlabel("Frame")
    ax3_right.legend()

    for ax in axs:
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle("Arm Metrics")
    plt.tight_layout()

    save_fig(session, 'arm_metrics.png', show_plot)


def plot_all_metrics(session, show_plot=True):
    """Plot all metrics in one function call."""
    show_plot = session.show_plot if hasattr(session, 'show_plot') else show_plot   

    plot_head_torso_metrics(session, show_plot)
    plot_bimanual_coordination(session, show_plot)
    plot_hand_kinematics(session, show_plot)
    plot_cycle_phase_metrics(session, show_plot)
    plot_hand_trajectories(session, show_plot)
    plot_path_descriptives(session, show_plot)
    plot_nose_tracking(session, show_plot)
    plot_postural_metrics(session, show_plot)
    plot_arm_metrics(session, show_plot)

    print("Session plotting complete.")
