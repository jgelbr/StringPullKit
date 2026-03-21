import cv2
import tkinter as tk
from tkinter import messagebox, filedialog
import json

def set_start(self):
    self.start_trim = self.frame_pos
    self.start_frame_var.set(str(self.start_trim))
    self.set_status(f"Start frame set to {self.start_trim}")

def set_end(self):
    self.end_trim = self.frame_pos
    self.end_frame_var.set(str(self.end_trim))
    self.set_status(f"End frame set to {self.end_trim}")

def add_clip(self):
    try:
        start = int(self.start_frame_var.get())
        end = int(self.end_frame_var.get())
        if start < end:
            self.clip_ranges.append((start, end))
            self.clip_listbox.insert(tk.END, f"{start} to {end}")
            self.set_status(f"Segment selected from {start} to {end}")
        else:
            messagebox.showerror("Invalid Range", "Start frame must be less than end frame.")
    except ValueError:
        messagebox.showerror("Invalid Input", "Start and End must be valid integers.")

def delete_clip(self):
    selected = self.clip_listbox.curselection()
    if selected:
        index = selected[0]
        self.clip_listbox.delete(index)
        del self.clip_ranges[index]
        self.set_status(f"Deleted segment from {self.start_trim} to {self.end_trim}")
    else:
        messagebox.showwarning("Delete Segment", "Please select a segment to delete.")

def clear_clips(self):
    self.clip_ranges.clear()
    self.clip_listbox.delete(0, tk.END)

def on_clip_select(self, event):
    if not self.cap:
        return
    selection = self.clip_listbox.curselection()
    if not selection:
        return
    index = selection[0]
    start_frame, _ = self.clip_ranges[index]

    #Jump to start frame
    self.frame_pos = start_frame
    self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    self.timeline.set(start_frame)
    self.start_frame_var.set(str(start_frame))
    self.show_frame()

def save_clips(self):
    if not self.clip_ranges:
        messagebox.showinfo("Save Segments", "No segments to save.")
        return

    save_path = filedialog.asksaveasfilename(defaultextension=".json",
                                             filetypes=[("JSON files", "*.json")])
    if not save_path:
        return

    try:
        with open(save_path, 'w') as f:
            json.dump(self.clip_ranges, f)
        messagebox.showinfo("Save Segments", f"Segments saved to {save_path}")
    except Exception as e:
        messagebox.showerror("Save Error", f"Could not save segments:\n{e}")

def load_clips(self):
    load_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
    if not load_path:
        return

    try:
        with open(load_path, 'r') as f:
            segments = json.load(f)

        if not isinstance(segments, list) or not all(isinstance(pair, list) and len(pair) == 2 for pair in segments):
            raise ValueError("Invalid format")

        self.clip_ranges = [tuple(pair) for pair in segments]
        self.clip_listbox.delete(0, tk.END)
        for start, end in self.clip_ranges:
            self.clip_listbox.insert(tk.END, f"{start} to {end}")
        messagebox.showinfo("Load Segments", f"Loaded {len(self.clip_ranges)} segments.")
    except Exception as e:
        messagebox.showerror("Load Error", f"Could not load segments:\n{e}")
