from tkinter import filedialog, messagebox, simpledialog
import cv2
import os

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
    
    cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_AUDIO_STREAM, -1  # disable audio stream
])
    
    # Determine output size from first frame (after rotation/crop)
    crop = self.crop_rect or self.original_crop_rect
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, test_frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Failed to read frame to determine output size.")
        cap.release()
        return

    # Apply rotation
    if self.rotation_angle == 90:
        test_frame = cv2.rotate(test_frame, cv2.ROTATE_90_CLOCKWISE)
    elif self.rotation_angle == 180:
        test_frame = cv2.rotate(test_frame, cv2.ROTATE_180)
    elif self.rotation_angle == 270:
        test_frame = cv2.rotate(test_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Apply crop (if any)
    if crop:
        x1, y1, x2, y2 = crop
        test_frame = test_frame[y1:y2, x1:x2]

    export_height, export_width = test_frame.shape[:2]
    
    cap.release()

    # Setup video writer - try codecs in order of reliability
    out = None
    codec_list = [
        ('mp4v', '.mp4'),
        ('XVID', '.avi'),
        ('MJPG', '.avi'),
        ('X264', '.mp4')
    ]
    
    for codec, ext in codec_list:
        # Adjust filename extension if needed
        if codec != 'mp4v' and save_path.endswith('.mp4'):
            test_path = save_path.replace('.mp4', ext)
        else:
            test_path = save_path
            
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(test_path, fourcc, self.fps, (export_width, export_height))
            if out.isOpened():
                save_path = test_path  # Use the working path
                print(f"Using codec: {codec}")
                break
            out.release()
        except:
            continue

    if not out or not out.isOpened():
        messagebox.showerror("Export Error", "Failed to create output video with any available codec.")
        return

    total_export_frames = sum(max(0, end - start + 1) for start, end in self.clip_ranges)
    self.progress["value"] = 0
    self.progress["maximum"] = total_export_frames
    self.progress.pack()
    self.root.update_idletasks()

    written_frames = 0

    for start, end in self.clip_ranges:
        if end < start:
            continue

        # Reopen video capture for each clip range for reliable seeking
        cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG, [
    cv2.CAP_PROP_AUDIO_STREAM, -1  # disable audio stream
])
        
        # Read and discard frames until we reach start position
        for i in range(start):
            cap.grab()
        
        # Now read and write the frames we want
        for frame_num in range(end - start + 1):
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame at position {start + frame_num}")
                break

            # Rotate
            if self.rotation_angle == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation_angle == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif self.rotation_angle == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Crop (if any)
            if crop:
                x1, y1, x2, y2 = crop
                frame = frame[y1:y2, x1:x2]

            if frame.shape[1] != export_width or frame.shape[0] != export_height:
                messagebox.showerror("Export Error",
                                     f"Frame size mismatch: got {frame.shape[1]}x{frame.shape[0]}, expected {export_width}x{export_height}")
                cap.release()
                out.release()
                self.progress.pack_forget()
                return

            out.write(frame)
            written_frames += 1
            self.progress["value"] = written_frames
            self.root.update_idletasks()
        
        cap.release()

    out.release()
    self.progress.pack_forget()
    
    print(f"Expected to write {total_export_frames} frames, actually wrote {written_frames} frames")
    messagebox.showinfo("Export", f"Exported {written_frames} frames to:\n{save_path}")
    return folder_dir
