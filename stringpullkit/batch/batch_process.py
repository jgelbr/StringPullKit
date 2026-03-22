import os
import json
from pathlib import Path
#import analysis  
from stringpullkit.analysis import analysis


def find_session_folders(base_dir):
    """
    Traverse directory structure to find all session folders containing dlc_output.
    
    Args:
        base_dir: Root directory to search (e.g., String_Pull)
    
    Returns:
        List of tuples: (session_folder_path, session_id)
    """
    session_folders = []
    base_path = Path(base_dir)
    
    # Walk through all directories
    for root, dirs, files in os.walk(base_path):
        # Skip Archive folders
        if 'Archive' in Path(root).parts:
            continue
            
        # Check if this directory contains dlc_output
        if 'dlc_output' in dirs:
            session_folder = Path(root)
            session_id = session_folder.name
            session_folders.append((str(session_folder), session_id))
    
    return session_folders


def get_dlc_csv_paths(session_folder):
    """
    Get all DLC CSV file paths from the dlc_output folder.
    
    Args:
        session_folder: Path to the session folder
    
    Returns:
        Dictionary mapping body part names to CSV file paths
    """
    dlc_output_dir = Path(session_folder) / 'dlc_output'
    
    if not dlc_output_dir.exists():
        return None
    
    # Define expected body parts
    body_parts = ['Arms', 'Ears2', 'Feet', 'Hands2', 'String3', 'Torso']
    dlc_paths = {}
    
    # Find CSV files for each body part
    for csv_file in dlc_output_dir.glob('*.csv'):
        filename = csv_file.name
        for part in body_parts:
            if part in filename:
                dlc_paths[part] = str(csv_file)
                break
    
    return dlc_paths if dlc_paths else None


def load_session_parameters(parameter_file):
    """
    Load scale factors and video heights from a JSON or CSV file.
    
    Expected format (JSON):
    {
        "session_id_1": {"scale_factor": 0.1, "height": 1416},
        "session_id_2": {"scale_factor": 0.15, "height": 1080},
        ...
    }
    
    Or CSV format:
    session_id,scale_factor,height
    session_id_1,0.1,1416
    session_id_2,0.15,1080
    
    Args:
        parameter_file: Path to the parameter file
    
    Returns:
        Dictionary mapping session_id to dict with scale_factor and height
    """
    parameters = {}
    file_path = Path(parameter_file)
    
    if file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            parameters = json.load(f)
    elif file_path.suffix == '.csv':
        import csv
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                parameters[row['session_id']] = {
                    'scale_factor': float((row['scale_factor'])),
                    'height': int(row['height'])
                }
    
    return parameters


def batch_process_sessions(base_dirs, parameter_file, fps=120, 
                           likelihood_threshold=0.6, smoothing_window=25, 
                           smoothing_poly=2, show_plot=False, dry_run=False):
    """
    Batch process all sessions in the given base directories.
    
    Args:
        base_dirs: List of base directory paths to search
        parameter_file: Path to file containing scale factors and heights for each session
        fps: Frame rate (default: 120)
        likelihood_threshold: DLC likelihood threshold (default: 0.6)
        smoothing_window: Smoothing window size (default: 25)
        smoothing_poly: Smoothing polynomial order (default: 2)
        show_plot: Whether to show plots (default: False)
        dry_run: If True, only print what would be processed without running analysis
    """
    # Load session parameters
    print(f"Loading session parameters from: {parameter_file}")
    session_params = load_session_parameters(parameter_file)
    print(f"Loaded parameters for {len(session_params)} sessions\n")
    
    # Find all session folders
    all_session_folders = []
    for base_dir in base_dirs:
        print(f"Searching for sessions in: {base_dir}")
        session_folders = find_session_folders(base_dir)
        all_session_folders.extend(session_folders)
        print(f"Found {len(session_folders)} sessions\n")
    
    print(f"Total sessions found: {len(all_session_folders)}\n")
    print("="*80)
    
    # Process each session
    processed = 0
    skipped = 0
    errors = []
    
    for session_folder, session_id in all_session_folders:
        print(f"\nProcessing: {session_id}")
        print(f"Location: {session_folder}")
        
        # Get DLC CSV paths
        dlc_paths = get_dlc_csv_paths(session_folder)
        if not dlc_paths:
            print(f"  ⚠️  No DLC CSV files found in dlc_output folder. Skipping.")
            skipped += 1
            continue
        
        print(f"  Found {len(dlc_paths)} DLC CSV files: {', '.join(dlc_paths.keys())}")
        
        # Check for session parameters
        if session_id not in session_params:
            print(f"  ⚠️  No parameters found for {session_id}. Skipping.")
            skipped += 1
            continue
        
        scale_factor = 1 / session_params[session_id]['scale_factor']
        height = session_params[session_id]['height']
        print(f"  Scale factor: {scale_factor}")
        print(f"  Video height: {height}")
        
        # Set save directory (replace analysis_output_units with new output)
        save_dir = session_folder
        print(f"  Output directory: {save_dir}")
        
        if dry_run:
            print(f"  [DRY RUN] Would process this session")
            processed += 1
            continue
        
        # Run analysis
        try:
            analysis.run_analysis(
                video_path=None,
                dlc_paths=dlc_paths,
                save_dir=save_dir,
                fps=fps,
                session_id=session_id,
                total_frames=0,
                likelihood_threshold=likelihood_threshold,
                smoothing_window=smoothing_window,
                smoothing_poly=smoothing_poly,
                scale_factor=scale_factor,
                height=height,
                show_plot=show_plot
            )
            print(f"  ✓ Analysis complete")
            processed += 1
            
        except Exception as e:
            error_msg = f"Error processing {session_id}: {str(e)}"
            print(f"  ✗ {error_msg}")
            errors.append(error_msg)
    
    # Summary
    print("\n" + "="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80)
    print(f"Total sessions found: {len(all_session_folders)}")
    print(f"Successfully processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {len(errors)}")
    
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"  - {error}")


if __name__ == "__main__":
    # Configuration
    BASE_DIRS = [
        r"\\m40-gci.sci.mcgill.ca\Watt-Projects\Rana-Juliana\String_Pull\Flx_Cohort_1",
        r"\\m40-gci.sci.mcgill.ca\Watt-Projects\Rana-Juliana\String_Pull\Flx_Cohort_2",
        r"\\m40-gci.sci.mcgill.ca\Watt-Projects\Rana-Juliana\String_Pull\Ctl_Early_Disease_Progression"
    ]
    
    # Path to session parameters file 
    PARAMETER_FILE = r"\\m40-gci.sci.mcgill.ca\Watt-Projects\Rana-Juliana\Code\Modules\session_parameters.json"
    
    # Analysis parameters
    FPS = 120
    LIKELIHOOD_THRESHOLD = 0.6
    SMOOTHING_WINDOW = 25
    SMOOTHING_POLY = 2
    SHOW_PLOT = False
    
    # Run batch processing
    # Set dry_run=True to test without actually running analysis
    batch_process_sessions(
        base_dirs=BASE_DIRS,
        parameter_file=PARAMETER_FILE,
        fps=FPS,
        likelihood_threshold=LIKELIHOOD_THRESHOLD,
        smoothing_window=SMOOTHING_WINDOW,
        smoothing_poly=SMOOTHING_POLY,
        show_plot=SHOW_PLOT,
        dry_run=False  # Set to False to actually run the analysis
    )