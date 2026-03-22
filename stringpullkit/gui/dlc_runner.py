from tkinter import messagebox
import os
import glob
#import config_manager
import shutil

from stringpullkit.gui import config_manager


def run_dlc_analysis(self):
    import deeplabcut
    selected_parts = [part for part, var in self.dlc_vars.items() if var.get()]
    if not selected_parts:
        messagebox.showwarning("No Selection", "Please select at least one body part for analysis.")
        return

    cached_configs = config_manager.load_config_cache()
    missing_parts = [part for part in selected_parts if part not in cached_configs]

    if missing_parts:
        messagebox.showinfo("Config Needed", "Please select the config files for missing body parts.")
        new_configs = config_manager.update_config_interactively(missing_parts)
        cached_configs.update(new_configs)
        config_manager.save_config_cache(cached_configs)

    if not self.video_path:
        self.set_status("No video loaded.")
        return

    video_dir = os.path.dirname(self.video_path)
    base_name = os.path.splitext(os.path.basename(self.video_path))[0]

    main_dir = os.path.dirname(video_dir)
    dlc_dir = os.path.join(main_dir, 'dlc_output')
    os.makedirs(dlc_dir, exist_ok=True)

    self.dlc_csv_paths = {}

    for part in selected_parts:
        config_path = cached_configs.get(part)
        if not config_path or not os.path.exists(config_path):
            self.set_status(f"Config missing or invalid for {part}")
            continue

        try:
            self.set_status(f"Running DLC analysis for {part}...")
            deeplabcut.analyze_videos(config_path, [self.video_path], save_as_csv=True)

            self.set_status(f"Creating labeled video for {part}...")
            deeplabcut.create_labeled_video(config_path, [self.video_path])

            # Refined glob pattern: look for DLC-labeled video for this part
            labeled_pattern = os.path.join(video_dir, f"{base_name}DLC_*_{part}*labeled.mp4")
            matches = glob.glob(labeled_pattern)

            if len(matches) > 1:
                print(f"⚠️ Multiple labeled videos found for {part}, using most recent.")
            if matches:
                original_path = max(matches, key=os.path.getctime)
                new_name = f"{base_name}_{part}_labeled.mp4"
                new_path = os.path.join(video_dir, new_name)
                os.rename(original_path, new_path)
            else:
                print(f"❌ No labeled video found for {part}")

            # Find and store the path to the CSV file
        
            for file in os.listdir(video_dir):
                if file.endswith(('.h5', '.pickle', '.csv')):
                    src = os.path.join(video_dir, file)
                    dst = os.path.join(dlc_dir, file)
                    shutil.move(src, dst)

            csv_pattern = os.path.join(dlc_dir, f"{base_name}*{part}*csv")
            csv_matches = glob.glob(csv_pattern)
            if csv_matches:
                latest_csv = max(csv_matches, key=os.path.getctime)
                self.dlc_csv_paths[part] = latest_csv

            self.set_status(f"{part} analysis complete.")
        except Exception as e:
            self.set_status(f"Error analyzing {part}: {e}")
            print(f"Error analyzing {part}: {e}")

def show_labeled_videos(self):
    if not hasattr(self, "video_path") or not self.video_path:
        self.set_status("No video loaded.")
        return

    video_dir = os.path.dirname(self.video_path)
    base_name = os.path.splitext(os.path.basename(self.video_path))[0]

    pattern = os.path.join(video_dir, f"{base_name}*labeled.mp4")
    labeled_videos = glob.glob(pattern)

    if not labeled_videos:
        self.set_status("No labeled video found.")
        return

    return labeled_videos


def update_dlc_config_paths(self):
    updated = config_manager.update_config_interactively(["Arms", "Ears", "Feet", "Hands", "String", "Torso"])
    if updated:
        self.set_status("DLC config paths updated.")
    else:
        self.set_status("No DLC configs updated.")


