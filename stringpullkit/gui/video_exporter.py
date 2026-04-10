import ffmpeg
import os
from tkinter import filedialog, messagebox, simpledialog

def trim_and_export(self, session_id=None):
    if not self.video_path:
        messagebox.showerror("Error", "No video loaded")
        return

    if not self.clip_ranges:
        self.clip_ranges = [(0, self.total_frames - 1)]

    folder_dir = filedialog.askdirectory(title="Select folder to save session files")
    if not folder_dir:
        return

    videos_dir = os.path.join(folder_dir, 'videos')
    os.makedirs(videos_dir, exist_ok=True)

    if not session_id:
        save_name = simpledialog.askstring("Save As", "Enter a name for the exported video:")
        if not save_name:
            return
        save_path = os.path.join(videos_dir, f"{save_name}.mp4")
    else:
        save_path = os.path.join(videos_dir, f"{session_id}.mp4")

    crop = self.crop_rect or self.original_crop_rect

    try:
        clip_paths = []
        temp_dir = os.path.join(folder_dir, '_temp_clips')
        os.makedirs(temp_dir, exist_ok=True)

        total_clips = len(self.clip_ranges)
        self.progress["value"] = 0
        self.progress["maximum"] = total_clips
        self.progress.pack()
        self.root.update_idletasks()

        for i, (start, end) in enumerate(self.clip_ranges):
            if end < start:
                continue

            start_time = start / self.fps
            duration = (end - start + 1) / self.fps
            clip_path = os.path.join(temp_dir, f"clip_{i}.mp4")

            stream = ffmpeg.input(self.video_path, ss=start_time, t=duration)

            # Video filters
            video = stream.video

            if self.rotation_angle == 90:
                video = video.filter('transpose', 1)
            elif self.rotation_angle == 180:
                video = video.filter('transpose', 1).filter('transpose', 1)
            elif self.rotation_angle == 270:
                video = video.filter('transpose', 2)

            if crop:
                x1, y1, x2, y2 = crop
                video = video.filter('crop', x2 - x1, y2 - y1, x1, y1)

            # If no filters applied, use stream copy (fast), otherwise re-encode
            needs_encode = self.rotation_angle != 0 or crop
            if needs_encode:
                out = ffmpeg.output(video, clip_path, vcodec='libx264', an=None)
            else:
                out = ffmpeg.output(video, clip_path, vcodec='copy', an=None)

            ffmpeg.run(out, overwrite_output=True, quiet=True)
            clip_paths.append(clip_path)

            self.progress["value"] = i + 1
            self.root.update_idletasks()

        # Concatenate clips if more than one
        if len(clip_paths) == 1:
            os.rename(clip_paths[0], save_path)
        else:
            # Write concat list file
            concat_list_path = os.path.join(temp_dir, 'concat_list.txt')
            with open(concat_list_path, 'w') as f:
                for clip_path in clip_paths:
                    f.write(f"file '{clip_path}'\n")

            ffmpeg.input(concat_list_path, format='concat', safe=0).output(
                save_path, vcodec='copy'
            ).run(overwrite_output=True, quiet=True)

        # Cleanup temp files
        for clip_path in clip_paths:
            try:
                os.remove(clip_path)
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass

    except ffmpeg.Error as e:
        messagebox.showerror("Export Error", f"FFmpeg error:\n{e.stderr.decode()}")
        return

    self.progress.pack_forget()
    messagebox.showinfo("Export", f"Exported to:\n{save_path}")
    return folder_dir
