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
        input_stream = ffmpeg.input(self.video_path, noautorotate=None)

        segments = []
        for start, end in self.clip_ranges:
            if end < start:
                continue

            start_time = start / self.fps
            duration = (end - start + 1) / self.fps

            segment = ffmpeg.input(self.video_path, ss=start_time, t=duration, noautorotate=None).video

            # Rotation
            if self.rotation_angle == 90:
                segment = segment.filter('transpose', 1)
            elif self.rotation_angle == 180:
                segment = segment.filter('rotate', 'PI')
            elif self.rotation_angle == 270:
                segment = segment.filter('transpose', 2)

            # Crop
            if crop:
                x1, y1, x2, y2 = crop
                segment = segment.filter('crop', x2 - x1, y2 - y1, x1, y1)

            segments.append(segment)

        if len(segments) == 0:
            messagebox.showerror("Export Error", "No valid clip ranges.")
            return
        elif len(segments) == 1:
            video = segments[0]
        else:
            video = ffmpeg.concat(*segments, v=1, a=0)

        needs_encode = self.rotation_angle != 0 or crop or len(segments) > 1
        vcodec = 'libx264' if needs_encode else 'copy'

        out = ffmpeg.output(video, save_path, vcodec=vcodec, an=None)
        ffmpeg.run(out, overwrite_output=True, quiet=True)

    except ffmpeg.Error as e:
        messagebox.showerror("Export Error", f"FFmpeg error:\n{e.stderr.decode()}")
        return

    messagebox.showinfo("Export", f"Exported to:\n{save_path}")
    return folder_dir
