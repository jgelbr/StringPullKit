import os
import json
import pandas as pd
from pathlib import Path


def get_video_height_from_file(session_folder):
    """
    Extract video height from video files in the 'videos' subdirectory.
    Uses the first video file found.
    
    Args:
        session_folder: Path to the session folder
    
    Returns:
        Video height as integer, or None if not found
    """
    session_path = Path(session_folder)
    videos_dir = session_path / 'videos'
    
    if not videos_dir.exists():
        return None
    
    # Common video extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
    
    # Find first video file
    video_file = None
    for ext in video_extensions:
        video_files = list(videos_dir.glob(f'*{ext}'))
        if video_files:
            video_file = video_files[0]
            break
    
    if not video_file:
        return None
    
    # Method 1: Try opencv (cv2) - most common and reliable
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_file))
        if cap.isOpened():
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if height > 0:
                return height
    except ImportError:
        pass
    except Exception as e:
        print(f"    OpenCV method failed: {e}")
    
    # Method 2: Try moviepy
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(str(video_file))
        height = clip.size[1]
        clip.close()
        if height > 0:
            return height
    except ImportError:
        pass
    except Exception as e:
        print(f"    MoviePy method failed: {e}")
    
    # Method 3: Try ffprobe (if available on system)
    try:
        import subprocess
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=height', '-of', 'csv=p=0',
             str(video_file)],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            height = int(result.stdout.strip())
            if height > 0:
                return height
    except (ImportError, FileNotFoundError):
        pass
    except Exception as e:
        print(f"    ffprobe method failed: {e}")
    
    return None 


def find_session_folders(base_dir):
    """
    Traverse directory structure to find all session folders containing dlc_output.
    
    Args:
        base_dir: Root directory to search
    
    Returns:
        List of tuples: (session_folder_path, session_id)
    """
    session_folders = []
    base_path = Path(base_dir)
    
    for root, dirs, files in os.walk(base_path):
        if 'Archive' in Path(root).parts:
            continue
            
        if 'dlc_output' in dirs:
            session_folder = Path(root)
            session_id = session_folder.name
            session_folders.append((str(session_folder), session_id))
    
    return session_folders





def generate_parameters_template(base_dirs, output_file, default_scale_factor=None):
    """
    Generate a template JSON file with session IDs and video heights.
    Scale factors are left as None for manual entry.
    Heights are automatically extracted from video files when possible.
    
    Args:
        base_dirs: List of base directory paths to search
        output_file: Path to output JSON file
        default_scale_factor: Optional default scale factor to use for all sessions
    """
    print("Scanning directories for sessions...\n")
    print("NOTE: Attempting to extract video heights from video files...")
    print("This requires opencv-python (cv2) to be installed.")
    print("Install with: pip install opencv-python\n")
    
    all_sessions = {}
    total_found = 0
    heights_extracted = 0
    
    for base_dir in base_dirs:
        print(f"Searching: {base_dir}")
        session_folders = find_session_folders(base_dir)
        print(f"  Found {len(session_folders)} sessions")
        total_found += len(session_folders)
        
        for session_folder, session_id in session_folders:
            print(f"  Processing: {session_id}")
            
            # Try to get video height from video file
            height = get_video_height_from_file(session_folder)
            
            if height:
                print(f"    ✓ Extracted height: {height}")
                heights_extracted += 1
            else:
                print(f"    ✗ Could not extract height (no video file found or error)")
            
            all_sessions[session_id] = {
                "scale_factor": default_scale_factor,
                "height": height
            }
        
        print()
    
    # Sort sessions by ID for easier manual editing
    all_sessions = dict(sorted(all_sessions.items()))
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(all_sessions, f, indent=4)
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total sessions found: {total_found}")
    print(f"Video heights extracted: {heights_extracted}")
    print(f"Heights needing manual entry: {total_found - heights_extracted}")
    print(f"\nTemplate saved to: {output_file}")
    print("\nNext steps:")
    print("1. Open the JSON file")
    print("2. Fill in scale_factor values for each session")
    print("3. Fill in any missing height values (marked as null)")
    print("4. Save the file and use it with the batch processing script")


if __name__ == "__main__":
    # Configuration
    BASE_DIRS = [
        r"\\m40-gci.sci.mcgill.ca\Watt-Projects\Rana-Juliana\String_Pull\Flx_Cohort_1",
        r"\\m40-gci.sci.mcgill.ca\Watt-Projects\Rana-Juliana\String_Pull\Flx_Cohort_2",
        r"\\m40-gci.sci.mcgill.ca\Watt-Projects\Rana-Juliana\String_Pull\Ctl_Early_Disease_Progression"
    ]
    
    OUTPUT_FILE = r"\\m40-gci.sci.mcgill.ca\Watt-Projects\Rana-Juliana\DLC_Preprocessing_Juliana\Modules\session_parameters.json"
    
    # Optional: set a default scale factor (e.g., 0.1)
    # Leave as None to require manual entry for all sessions
    DEFAULT_SCALE_FACTOR = None  # or 0.1, 0.15, etc.
    
    generate_parameters_template(
        base_dirs=BASE_DIRS,
        output_file=OUTPUT_FILE,
        default_scale_factor=DEFAULT_SCALE_FACTOR
    )