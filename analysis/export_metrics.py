import os
import json
import h5py
import numpy as np
import pandas as pd
import datetime
#import utils
from stringpullkit.analysis import utils

def save_session_to_h5(session):
    """Save full session to HDF5 file (hierarchical structured format)."""
    if session.scale_factor:
        save_dir = os.path.join(session.save_dir, 'analysis_output_units')
    else: 
        save_dir = os.path.join(session.save_dir, 'analysis_output_pixels')
    session.save_dir = save_dir
    os.makedirs(session.save_dir, exist_ok=True)
    save_path = os.path.join(session.save_dir, f'{session.session_id}_session_data.h5')
    print(f"Saving session data to {save_path}...")

    with h5py.File(save_path, 'w') as f:

        # --- Metadata ---
        metadata_group = f.create_group("metadata")
        metadata = {
            'session_id': getattr(session, 'session_id', None),
            'export_time': datetime.datetime.now().isoformat(),
            'video_path': getattr(session, "video_path", None),
            'fps': getattr(session, "fps", None),
            'scale_factor': getattr(session, "scale_factor", None),
            'dlc_paths': json.dumps(session.dlc_paths)
        }

        for key, value in metadata.items():
            if value is not None:
                metadata_group.attrs[key] = str(value)

        metadata_group.attrs["dlc_paths"] = json.dumps(session.dlc_paths)

        # --- Metrics ---
        metrics_group = f.create_group("metrics")
        for key, value in session.metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                metrics_group.attrs[key] = value
            elif isinstance(value, (list, np.ndarray)):
                metrics_group.create_dataset(key, data=np.array(value))
            elif isinstance(value, dict):
                sub_group = metrics_group.create_group(key)
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float, np.integer, np.floating)):
                        sub_group.attrs[sub_key] = sub_value
                    elif isinstance(sub_value, (list, np.ndarray)):
                        sub_group.create_dataset(sub_key, data=np.array(sub_value))
            else:
                metrics_group.attrs[key] = str(value)

        # --- Phase Metrics ---
        phase_metrics_group = f.create_group("phase_metrics")
        for phase, phase_dict in session.phase_metrics.items():
            phase_group = phase_metrics_group.create_group(phase)
            for key, value in phase_dict.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    phase_group.attrs[key] = value
                elif isinstance(value, (list, np.ndarray)):
                    phase_group.create_dataset(key, data=np.array(value))
                elif isinstance(value, dict):
                    sub_group = phase_group.create_group(key)
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float, np.integer, np.floating)):
                            sub_group.attrs[sub_key] = sub_value
                        elif isinstance(sub_value, (list, np.ndarray)):
                            sub_group.create_dataset(sub_key, data=np.array(sub_value))
                else:
                    phase_group.attrs[key] = str(value)

    print(f"Session data saved successfully to {save_path}")


def save_session_to_xlsx(session):
    """Save session summary and vectors to Excel (multi-sheet)."""
    save_path = os.path.join(session.save_dir, f'{session.session_id}_session_data.xlsx')
    print(f"Saving session metrics to {save_path}...")

    # --- Metadata ---
    metadata = {
        'session_id': getattr(session, 'session_id', None),
        'export_time': datetime.datetime.now().isoformat(),
        'video_path': getattr(session, "video_path", None),
        'fps': getattr(session, "fps", None),
        'scale_factor': getattr(session, "scale_factor", None),
        'dlc_paths': json.dumps(session.dlc_paths)
    }
    metadata_df = pd.DataFrame(list(metadata.items()), columns=['Parameter', 'Value'])

    # --- Metrics (scalars) ---
    metrics_rows = []
    spillover = {} # To store lists from collapsed dictionaries.
    for key, value in session.metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating, str, np.generic)):
            metrics_rows.append({'Metric': key, 'Value': value})
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float, np.integer, np.floating, str, np.generic)):
                    metrics_rows.append({'Metric': f"{key}_{sub_key}", 'Value': sub_value})
                elif isinstance(sub_value, (list, np.ndarray)):
                    spillover[f'{key}_{sub_key}'] = sub_value

    metrics_df = pd.DataFrame(metrics_rows)

    # --- Vectors (time series) ---
    vector_data = {}
    for key, value in session.metrics.items():
        if isinstance(value, (list, np.ndarray)):
            arr = np.array(value)
            if arr.ndim == 1:
                vector_data[key] = arr
            # silently skip anything 2D+ — it belongs in H5 only
    for key, value in spillover.items():
        arr = np.array(value)
        if arr.ndim == 1 and len(arr) > 0:
            vector_data[key] = arr
    if vector_data:
        max_len = max(len(v) for v in vector_data.values())
        df_vectors = pd.DataFrame({k: np.pad(np.array(v, dtype=float), (0, max_len - len(v)), constant_values=np.nan)
                                   for k, v in vector_data.items()})
        df_vectors.insert(0, "Frame", np.arange(max_len))
    else:
        df_vectors = pd.DataFrame()

    # --- Phases (summary) ---
    phase_rows = []
    for phase_name, phase_dict in session.phase_metrics.items():
        summary = {"Phase": phase_name}
        for key, value in phase_dict.items():
            if isinstance(value, (int, float, np.integer, np.floating, str, np.generic)):
                summary[key] = value
            elif isinstance(value, (list, np.ndarray)):
                try:
                    if key != 'ranges':
                        summary[f'{key}_mean'] = np.nanmean(value)
                        summary[f'{key}_cv'] = utils.coefficient_of_variation(value)
                except:
                    pass
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (list, np.ndarray)):
                        try:
                            summary[f'{key}_{sub_key}_mean'] = np.nanmean(sub_value)
                            summary[f'{key}_{sub_key}_cv'] = utils.coefficient_of_variation(sub_value)
                        except:
                            pass
                    else:
                        summary[f"{key}_{sub_key}"] = sub_value
        phase_rows.append(summary)
    df_phases = pd.DataFrame(phase_rows)

    # --- Phase Vectors ---
    phase_vectors = {}

    for phase_name, phase_dict in session.phase_metrics.items():
        for key, value in phase_dict.items():

            # Case 1: nested dicts
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    arr = np.array(sub_value, dtype=float)
                    if arr.ndim > 1:
                        arr = arr.flatten()
                    phase_vectors[f"{phase_name}_{key}_{sub_key}"] = arr

            # Case 2: list or ndarray
            elif isinstance(value, (list, np.ndarray)):

                # Detect if it's a list of tuples/lists (e.g., frame ranges)
                if (len(value) > 0 and isinstance(value[0], (tuple, list)) and len(value[0]) == 2):
                    stringified = [f"({int(a)}, {int(b)})" for a, b in value]
                    phase_vectors[f"{phase_name}_{key}"] = np.array(stringified, dtype=object)

                else:
                    arr = np.array(value, dtype=float)
                    if arr.ndim > 1:
                        arr = arr.flatten()
                    phase_vectors[f"{phase_name}_{key}"] = arr

    if phase_vectors:
        max_len = max(len(v) for v in phase_vectors.values())
        df_phase_vectors = pd.DataFrame({
            k: np.pad(v, (0, max_len - len(v)), constant_values=np.nan)
            for k, v in phase_vectors.items()
        })
        df_phase_vectors.insert(0, "Index", np.arange(max_len))
    else:
        df_phase_vectors = pd.DataFrame()

    # --- Save to Excel ---
    with pd.ExcelWriter(save_path, engine="xlsxwriter") as writer:
        metadata_df.to_excel(writer, sheet_name="Metadata", index=False)
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
        if not df_vectors.empty:
            df_vectors.to_excel(writer, sheet_name="Vectors", index=False)
        if not df_phases.empty:
            df_phases.to_excel(writer, sheet_name="Phases", index=False)
        if not df_phase_vectors.empty:
            df_phase_vectors.to_excel(writer, sheet_name="Phase_Vectors", index=False)

    print(f"Session data saved successfully to {save_path}")


def save_all_metrics(session):

    save_session_to_h5(session)
    save_session_to_xlsx(session)
    print("Session metrics successfully exported.")
