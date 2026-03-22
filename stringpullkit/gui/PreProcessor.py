import tkinter as tk
import math
from tkinter import filedialog, messagebox, simpledialog, ttk
import cv2
import os
from PIL import Image, ImageTk
#import video_exporter, themes, dlc_runner, segments, analysis

from stringpullkit.gui import video_exporter, themes, dlc_runner, segments
from stringpullkit.analysis import analysis


class PreProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("String Pull Preprocessor")

        #=== Theme ===
        self.style = ttk.Style()
        self.current_theme = "Sleek Modern"  # Default
        themes.apply_theme(self.style, self.current_theme)
        self.theme_colors = themes.get_theme_colors(self.current_theme)

        # === State ===
        self.video_path = None
        self.labeled_video = False
        self.current_video_path = None
        self.save_path = None
        self.session_id = None
        self.cap = None
        self.playing = False
        self.frame_pos = 0
        self.total_frames = 0
        self.fps = 120
        self.start_trim = 0
        self.end_trim = 0
        self.clip_ranges = []
        self.dlc_csv_paths = {}

        self.crop_mode = False
        self.crop_rect = None
        self.original_crop_rect = None
        self.drag_start = None
        self.rect_id = None

        self.calibration_mode = False
        self.calibration_line = None
        self.scale_factor = None

        self.rotation_angle = 0
        self.size_factor = 0.3
        self.original_width = 640
        self.original_height = 360
        self.display_width = int(self.original_width * self.size_factor)
        self.display_height = int(self.original_height * self.size_factor)

        # === Menu ===
        menubar = tk.Menu(self.root)

        # File menu
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Load Video", command=self.load_video)
        filemenu.add_command(label="Export", command=self.trim_and_export)
        filemenu.add_separator()
        filemenu.add_command(label="Update DLC Config Paths", command=self.update_dlc_config_paths)
        filemenu.add_command(label="Load DLC CSV", command=self.load_dlc_csvs)
        filemenu.add_command(label="Set Scale", command=self.set_scale_manually)
        filemenu.add_command(label="Set Session ID", command=self.set_session_id)
        filemenu.add_separator()
        filemenu.add_command(label="Quit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)

        #Segments submenu
        segmentmenu = tk.Menu(menubar, tearoff=0)
        segmentmenu.add_command(label="Save Segments", command=self.save_clips)
        segmentmenu.add_command(label="Load Segments", command=self.load_clips)
        menubar.add_cascade(label="Segments", menu=segmentmenu)

        #Themes Submenu
        thememenu = tk.Menu(menubar, tearoff=0)
        thememenu.add_command(label="Dark", command=lambda: self.set_theme("Sleek Modern"))
        thememenu.add_command(label="Pink", command=lambda: self.set_theme("Whimsical Pink"))
        thememenu.add_command(label="Brown", command=lambda: self.set_theme("Retro Brown"))
        thememenu.add_command(label="Green", command=lambda: self.set_theme("Green Forest"))
        thememenu.add_command(label="Blue", command=lambda: self.set_theme("Ocean"))
        menubar.add_cascade(label="Themes", menu=thememenu)

        # === Main layout: left canvas, right panel ===
        main_frame = ttk.Frame(root)
        main_frame.pack(fill='both', expand=True)

        # === Left: Video canvas ===
        self.canvas = tk.Canvas(main_frame, width=self.display_width, height=self.display_height)
        self.canvas.config(bg=self.theme_colors["bg"])
        self.canvas.grid(row=0, column=0, padx=5, pady=5)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=1, column=0, pady=5)

        self.prev_button = ttk.Button(controls_frame, text="<<", command=self.prev_frame)
        self.prev_button.pack(side='left', padx=2)

        self.play_button = ttk.Button(controls_frame, text="Play", command=self.toggle_play)
        self.play_button.pack(side='left', padx=2)

        self.next_button = ttk.Button(controls_frame, text=">>", command=self.next_frame)
        self.next_button.pack(side='left', padx=2)

        # === Right: All controls stacked vertically ===
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, rowspan=8, sticky="nw", padx=10)

        # Top row: Start/End Entry
        nav_frame = ttk.Frame(right_panel)
        nav_frame.pack(anchor="w", pady=(10, 2))

        entry_frame = ttk.Frame(nav_frame)
        entry_frame.pack(side='left')

        ttk.Label(entry_frame, text="Start Frame:").pack(side='left')
        self.start_frame_var = tk.StringVar()
        self.start_entry = tk.Entry(entry_frame, textvariable=self.start_frame_var, width=6)
        self.start_entry.pack(side='left')
        self.start_entry.config(bg=self.theme_colors["highlight"], fg=self.theme_colors["fg"], insertbackground=self.theme_colors["fg"])
        self.start_entry.bind("<Return>", self.jump_to_start_entry)

        ttk.Label(entry_frame, text="  End Frame:").pack(side='left')
        self.end_frame_var = tk.StringVar()
        self.end_entry = tk.Entry(entry_frame, textvariable=self.end_frame_var, width=6)
        self.end_entry.pack(side='left')
        self.end_entry.config(bg=self.theme_colors["highlight"], fg=self.theme_colors["fg"], insertbackground=self.theme_colors["fg"])
        self.end_entry.bind("<Return>", self.jump_to_end_entry)

        # Timeline
        self.timeline = tk.Scale(right_panel, from_=0, to=100, orient="horizontal", length=500, command=self.scrub)
        self.timeline.pack(anchor="w", pady=4)
        self.timeline.configure(bg=self.theme_colors["bg"],fg=self.theme_colors["fg"],
        troughcolor=self.theme_colors["highlight"],sliderrelief='flat',
        activebackground=self.theme_colors["btn_bg"], highlightthickness=0)

        self.time_label = ttk.Label(right_panel, text="00:00.000 / 00:00.000")
        self.time_label.pack(anchor="w")

        # Segment controls
        segment_frame = ttk.Frame(right_panel)
        segment_frame.pack(anchor="w", pady=(10, 0))

        self.set_start_button = ttk.Button(segment_frame, text="Set Start", command=self.set_start)
        self.set_start_button.pack(side='left', padx=2)

        self.set_end_button = ttk.Button(segment_frame, text="Set End", command=self.set_end)
        self.set_end_button.pack(side='left', padx=2)

        self.add_clip_button = ttk.Button(segment_frame, text="Add Segment", command=self.add_clip)
        self.add_clip_button.pack(side='left', padx=2)

        self.delete_clip_button = ttk.Button(segment_frame, text="Delete Segment", command=self.delete_clip)
        self.delete_clip_button.pack(side='left', padx=2)

        self.clear_clips_button = ttk.Button(segment_frame, text="Clear Segments", command=self.clear_clips)
        self.clear_clips_button.pack(side='left', padx=2)

        self.clip_listbox = tk.Listbox(right_panel, height=4)
        self.clip_listbox.pack(fill='x', padx=4, pady=(4, 0))
        self.clip_listbox.config(bg=self.theme_colors["highlight"], fg=self.theme_colors["fg"],
                                    selectbackground=self.theme_colors["btn_bg"])
        self.clip_listbox.bind("<<ListboxSelect>>", self.on_clip_select)
        self.clip_listbox.bind("<Delete>", lambda e: self.delete_clip())

        # Crop & Rotate & Scale
        transform_frame = ttk.Frame(right_panel)
        transform_frame.pack(anchor="w", pady=10)

        self.rotate_button = ttk.Button(transform_frame, text="Rotate", command=self.rotate_video)
        self.rotate_button.pack(side='left', padx=2)

        self.crop_button = ttk.Button(transform_frame, text="Crop Mode", command=self.toggle_crop_mode)
        self.crop_button.pack(side='left', padx=2)

        self.confirm_crop_button = ttk.Button(transform_frame, text="Confirm Crop", command=self.confirm_crop)
        self.confirm_crop_button.pack(side='left', padx=2)

        self.undo_crop_button = ttk.Button(transform_frame, text="Undo Crop", command=self.undo_crop)
        self.undo_crop_button.pack(side='left', padx=2)

        #Size and Scale Adjustment
        size_frame = ttk.Frame(right_panel)
        size_frame.pack(anchor="w", pady=(5, 0))

        self.size_var = tk.StringVar(value=str(int(self.size_factor * 100)))
        ttk.Label(size_frame, text="Size %:").pack(side='left')
        self.size_entry = tk.Entry(size_frame, textvariable=self.size_var, width=4)
        self.size_entry.pack(side='left', padx=2)
        self.size_entry.config(bg=self.theme_colors["highlight"], fg=self.theme_colors["fg"], insertbackground=self.theme_colors["fg"])
        self.size_entry.bind("<Return>", lambda e: self.apply_size_input())

        self.scale_button = ttk.Button(size_frame, text="Calibrate Scale", command=self.toggle_calibration_mode)
        self.scale_button.pack(side='left', padx=2)

        # --- DLC and Analysis ----

        dlc_video_frame = ttk.Frame(right_panel)
        dlc_video_frame.pack(anchor="w", pady=(5, 0), side="right")

        self.labeled_video_var = tk.StringVar()
        self.labeled_video_map = {}
        ttk.Label(dlc_video_frame, text="Select labeled video to view:").pack(anchor="w")
        self.labeled_video_dropdown = ttk.Combobox(
            dlc_video_frame,
            textvariable=self.labeled_video_var,
            state="readonly",
            width=40,
            style="Custom.TCombobox"
        )
        self.labeled_video_dropdown.pack(anchor="w", pady=(2, 5))
        self.labeled_video_dropdown.bind("<<ComboboxSelected>>", self.open_selected_labeled_video)

        analysis_frame = ttk.Frame(right_panel)
        analysis_frame.pack(anchor="w", pady=(5,0), side="left")

        self.dlc_vars = {}
        #ttk.Label(analysis_frame, text="Select DLC body parts:").pack(anchor="w")
       
        for part in ['Arms', 'Ears', 'Feet', 'Hands', 'String', 'Torso']:
            var = tk.BooleanVar(value=True)
            self.dlc_vars[part] = var
            # ttk.Checkbutton(
            #     analysis_frame,
            #     text=part,
            #     variable=var
            # ).pack(anchor="w")
    # Remove the checkboxes because the current program requires all parts (no checks in place yet). Could add back later. 

        self.run_dlc_button = ttk.Button(analysis_frame, text="Run DLC Tracking", command=self.run_dlc_analysis)
        self.run_dlc_button.pack(anchor="w", pady=(10, 5))

        self.analyze_hands_button = ttk.Button(analysis_frame, text="Run Kinematic Analysis", command=self.plot_results)
        self.analyze_hands_button.pack(anchor="w")

        self.show_plot = tk.BooleanVar(value=False)
        self.show_plot_checkbox = ttk.Checkbutton(analysis_frame, text="Show Plots", variable=self.show_plot)
        self.show_plot_checkbox.pack(anchor="w")

        # Status + Progress
        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(root, textvariable=self.status_var, anchor='w', relief='sunken')
        self.status_label.pack(fill='x', side='bottom')

        self.progress = ttk.Progressbar(root, orient="horizontal", length=self.display_width, mode="determinate")
        self.progress.pack()
        self.style.configure("TProgressbar", troughcolor=self.theme_colors["bg"], background=self.theme_colors["highlight"])
        self.progress.pack_forget()

        # Keyboard Bindings
        self.root.bind("<space>", lambda e: self.toggle_play())
        self.root.bind("<Left>", lambda e: self.prev_frame())
        self.root.bind("<Right>", lambda e: self.next_frame())
        self.root.bind("s", lambda e: self.set_start())
        self.root.bind("e", lambda e: self.set_end())
        self.root.bind("c", lambda e: self.toggle_crop_mode())
        self.root.bind("r", lambda e: self.rotate_video())
        self.root.bind("<Return>", lambda e: self.confirm_crop())
        self.root.bind("u", lambda e: self.undo_crop())
        self.root.bind("t", lambda e: self.trim_and_export())

        self.update_loop()

    def set_theme(self, theme_name):
        self.current_theme = theme_name
        themes.apply_theme(self.style, theme_name)
        self.theme_colors = themes.get_theme_colors(theme_name)
        self.apply_theme_colors()

    def apply_theme_colors(self):
        fg = self.theme_colors["fg"]
        bg = self.theme_colors["highlight"]

        entries = [self.start_entry, self.end_entry, self.size_entry]
        for entry in entries:
            entry.config(bg=bg, fg=fg, insertbackground=fg)

        self.clip_listbox.config(bg=bg, fg=fg, selectbackground=self.theme_colors["btn_bg"])
        self.timeline.config(bg=self.theme_colors["bg"], fg=fg, troughcolor=self.theme_colors["highlight"],
                             highlightthickness=0, sliderrelief='flat')
        self.canvas.config(bg=self.theme_colors["highlight"])

        self.style.configure("Custom.TCombobox",
                             foreground=self.theme_colors["fg"],
                             background=self.theme_colors["highlight"],
                             fieldbackground=self.theme_colors["highlight"],
                             selectforeground=self.theme_colors["fg"],
                             selectbackground=self.theme_colors["btn_bg"],
                             arrowcolor=self.theme_colors["fg"]
                             )
        self.style.map("Custom.TCombobox",
                       fieldbackground=[('readonly', self.theme_colors["highlight"])],
                       background=[('readonly', self.theme_colors["highlight"])],
                       foreground=[('readonly', self.theme_colors["fg"])],
                       arrowcolor=[('readonly', self.theme_colors["fg"])]
                       )

    def load_video(self, path=None, update_original=True):
        if path is None:
            path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
            if not path:
                return

        # Store path appropriately
        if update_original:
            self.video_path = path
            self.dlc_csv_paths = {}
            self.labeled_video_map.clear()

        else:
            self.labeled_video = path

        # Decide which video to actually load
        video_to_load = path if update_original else self.labeled_video

        # Load the video
        self.cap = cv2.VideoCapture(video_to_load, cv2.CAP_FFMPEG)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(self.total_frames)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # UI and state updates
        self.timeline.config(to=self.total_frames - 1)
        self.playing = False
        self.clip_ranges = []
        segments.clear_clips(self)
        self.crop_mode = False
        self.frame_pos = 0
        self.start_trim = 0
        self.end_trim = self.total_frames - 1
        self.crop_rect = None
        self.original_crop_rect = None
        self.rotation_angle = 0
        self.calibration_mode = False
        self.scale_factor = None

        self.resize_canvas()
        self.show_frame()
        self.set_status(f"Loaded {os.path.basename(path)}")

    def set_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()

    def trim_and_export(self):
        self.set_status("Exporting...")
        save_path = video_exporter.trim_and_export(self, session_id=self.session_id)
        return save_path

    def set_start(self):
        segments.set_start(self)

    def set_end(self):
        segments.set_end(self)

    def add_clip(self):
        segments.add_clip(self)

    def delete_clip(self):
        segments.delete_clip(self)

    def clear_clips(self):
        segments.clear_clips(self)

    def on_clip_select(self, event):
        segments.on_clip_select(self, event)

    def save_clips(self):
        segments.save_clips(self)

    def load_clips(self):
        segments.load_clips(self)

    def ask_body_part_choice(self, filename, part_options):
        dialog = tk.Toplevel()
        dialog.title("Select Body Part")

        # Set size
        width, height = 300, 120
        dialog.geometry(f"{width}x{height}")

        # Center on parent
        self.root.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (width // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (height // 2)
        dialog.geometry(f"+{x}+{y}")

        dialog.grab_set()  # Force user to answer  (mwhaahaha)

        tk.Label(dialog, text=f"File: {filename}").pack(pady=(10, 2))

        choice_var = tk.StringVar(value=part_options[0])
        dropdown = ttk.Combobox(dialog, textvariable=choice_var, values=part_options, state="readonly")
        dropdown.pack(pady=5)

        selected = []

        def on_ok():
            selected.append(choice_var.get())
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side="left", padx=5)

        dialog.wait_window()
        return selected[0] if selected else None

    def set_scale_manually(self):
        scale_input = simpledialog.askfloat("Manual Scale Selection", "Enter scale (px/mm):", parent=self.root)
        if scale_input is not None:
            self.scale_factor = 1 / scale_input
    
    def set_session_id(self):
        session_id = simpledialog.askstring("Set Session ID", "Enter a unique session ID:", parent=self.root)
        if session_id:
            self.session_id = session_id.strip()
            messagebox.showinfo("Session ID Set", f"Session ID set to: {self.session_id}")

    def load_dlc_csvs(self):
        # Open file dialog to select DLC CSVs for different body parts
        file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if not file_paths:
            return  # User canceled

        # Ask user for the body parts corresponding to the selected files
        part_labels = ['Arms', 'Ears', 'Feet', 'Hands', 'String', 'Torso']

        if len(self.dlc_csv_paths) == 6:
            self.dlc_csv_paths = {}

        for i, file_path in enumerate(file_paths):
            filename = os.path.basename(file_path)
            body_part = self.ask_body_part_choice(filename, part_labels)
            if body_part:
                self.dlc_csv_paths[body_part] = file_path

        self.set_status(f"Loaded {len(self.dlc_csv_paths)} DLC CSV files.")

    def on_mouse_press(self, event):
        if self.crop_mode:
            self.start_crop(event)
        elif self.calibration_mode:
            self.start_calibration_line(event)

    def on_mouse_drag(self, event):
        if self.crop_mode:
            self.draw_crop(event)
        elif self.calibration_mode:
            self.draw_calibration_line(event)


    def on_mouse_release(self, event):
        if self.crop_mode:
            self.end_crop(event)
        elif self.calibration_mode:
            self.end_calibration_line(event)

    def toggle_calibration_mode(self):
        self.calibration_mode= not self.calibration_mode
        if self.calibration_mode:
            self.crop_mode = False
        self.set_status("Calibration mode enabled. Draw a line over known distance." if self.calibration_mode
                        else "Exited calibration mode.")
        if self.calibration_line:
            self.canvas.delete(self.calibration_line)
            self.calibration_line = None

    def start_calibration_line(self, event):
        if not self.calibration_mode:
            return
        self.drag_start = (event.x, event.y)
        if self.calibration_line:
            self.canvas.delete(self.calibration_line)
            self.calibration_line = None

    def draw_calibration_line(self, event):
        if not self.calibration_mode or not self.drag_start:
            return
        x1, y1 = self.drag_start
        x2, y2 = event.x, event.y
        if self.calibration_line:
            self.canvas.delete(self.calibration_line)
        self.calibration_line = self.canvas.create_line(x1, y1, x2, y2, fill=self.theme_colors["highlight"], width=3)

    def end_calibration_line(self, event):
        if not self.calibration_mode or not self.drag_start:
            return

        x1, y1 = self.drag_start

        x2, y2 = event.x, event.y

        self.drag_start = None

        fx = 1 / self.size_factor
        fy = 1 / self.size_factor
        x1_real, y1_real = x1 * fx, y1 * fy
        x2_real, y2_real = x2 * fx, y2 * fy


        pixel_distance = math.sqrt((x2_real - x1_real)**2 + (y2_real - y1_real)**2)
        self.set_status(f"Pixel Distance: {pixel_distance}")
        real_length = simpledialog.askfloat("Calibration", "Enter real length (mm)")

        if real_length is None or real_length <= 0:
            self.set_status("Calibration cancelled.")
            if self.calibration_line:
                self.canvas.delete(self.calibration_line)
            return

        self.scale_factor = real_length / pixel_distance
        scale_inverted = pixel_distance / real_length
        self.set_status(f"Scale set: {scale_inverted: .4f} px/mm")
        print(scale_inverted)
        self.canvas.delete(self.calibration_line)
        self.calibration_mode = False

    def resize_canvas(self):
        if self.rotation_angle in [90, 270]:
            self.display_width = int(self.original_height * self.size_factor)
            self.display_height = int(self.original_width * self.size_factor)
        else:
            self.display_width = int(self.original_width * self.size_factor)
            self.display_height = int(self.original_height * self.size_factor)

        self.canvas.config(width=self.display_width, height=self.display_height)
        self.timeline.config(length=self.display_width)
       # self.progress.config(length=self.display_width)

    def apply_size_input(self):
        try:
            value = int(self.size_var.get())
            if 10 <= value <= 100:
                self.size_factor = value / 100.0
                self.display_width = int(self.original_width * self.size_factor)
                #####delete

                self.display_height = int(self.original_height * self.size_factor)

                self.canvas.config(width=self.display_width, height=self.display_height)
                self.timeline.config(length=self.display_width)
                self.progress.config(length=self.display_width)
                self.show_frame()
                self.set_status(f"Scale set to {self.size_factor*100}%")
            else:
                self.set_status("Scale must be between 10% and 100%")
        except ValueError:
            self.set_status("Invalid size value")

    def toggle_play(self):
        if not self.cap:
            return

        # Toggle between Play and Pause
        if not self.playing:
            self.playing = True
            self.play_button.config(text="Pause")
            self.set_status(message="Playing")
        else:
            self.playing = False
            self.play_button.config(text="Play")
            self.set_status(message="Paused")

    def update_loop(self):
        if self.playing and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                # Check if we are truly at the last frame
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if self.frame_pos >= total_frames - 1:
                    self.playing = False
                    self.play_button.config(text="Play")
                    self.set_status(message="Playback ended")

                    # Rewind to the start
                    self.frame_pos = 0
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:

                    self.root.after(int(1000 / self.fps), self.update_loop)
                    return

            self.frame_pos += 3
            self.show_frame(frame)

        self.root.after(int(1000 / self.fps), self.update_loop)


    def show_frame(self, frame=None):
        if not self.cap:
            return

        if frame is None:
            # Only do this when NOT in normal playback (e.g., scrubbing)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_pos)
            ret, frame = self.cap.read()
            if not ret:
                return

        # Rotate
        if self.rotation_angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation_angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Crop only if confirmed
        if self.original_crop_rect:
            x1, y1, x2, y2 = self.original_crop_rect
            frame = frame[y1:y2, x1:x2]

            # Dynamically update canvas and display dimensions
            h, w = frame.shape[:2]
            new_display_width = int(w * self.size_factor)
            new_display_height = int(h * self.size_factor)

            if new_display_width != self.display_width or new_display_height != self.display_height:
                self.display_width = new_display_width
                self.display_height = new_display_height
                self.canvas.config(width=self.display_width, height=self.display_height)

        # Resize for display
        frame = cv2.resize(frame, (self.display_width, self.display_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0, 0, anchor="nw", image=imgtk)

        # Draw crop rectangle if it exists and not confirmed
        if self.crop_rect and not self.original_crop_rect:
            sx = self.size_factor
            sy = self.size_factor
            x1, y1, x2, y2 = self.crop_rect
            x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
            if self.rect_id:
                self.canvas.delete(self.rect_id)
            self.rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=2)

        self.timeline.set(self.frame_pos)
        self.update_time_label()

    def scrub(self, value):
        if not self.cap:
            return
        self.frame_pos = int(value)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_pos)
        self.show_frame()

    def prev_frame(self):
        if self.cap and self.frame_pos > 0:
            self.frame_pos -= 120
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_pos)
            self.show_frame()

    def next_frame(self):
        if self.cap and self.frame_pos < self.total_frames - 1:
            self.frame_pos += 120
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_pos)
            self.show_frame()

    def toggle_crop_mode(self):
        self.crop_mode = not self.crop_mode
        if self.crop_mode:
            self.calibration_mode = False
        if not self.crop_mode:
            self.canvas.delete(self.rect_id)
        self.set_status("Crop mode" if self.crop_mode else "Exited crop mode")

    def start_crop(self, event):
        if not self.crop_mode:
            return
        self.drag_start = (event.x, event.y)
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

    def draw_crop(self, event):
        if not self.crop_mode or not self.drag_start:
            return
        x1, y1 = self.drag_start
        x2, y2 = event.x, event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline=self.theme_colors["highlight"], width=4)

    def end_crop(self, event):
        if not self.crop_mode or not self.drag_start:
            return

        x1, y1 = self.drag_start
        x2, y2 = event.x, event.y
        self.drag_start = None

        # Scale from display to real size
        fx = 1 / self.size_factor
        fy = 1 / self.size_factor
        x1, y1 = int(x1 * fx), int(y1 * fy)
        x2, y2 = int(x2 * fx), int(y2 * fy)

        self.crop_rect = (min(x1, x2),min(y1, y2),max(x1, x2),max(y1, y2))

        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

        self.show_frame()

    def confirm_crop(self):
        if self.crop_rect:
            self.original_crop_rect = self.crop_rect
            self.crop_rect = None
            if self.rect_id:
                self.canvas.delete(self.rect_id)
                self.rect_id = None

            # Adjust original dimensions
            x1, y1, x2, y2 = self.original_crop_rect

            self.original_width = x2 - x1
            self.original_height = y2 - y1

            self.resize_canvas()
            self.show_frame()
            self.set_status(f"Crop confirmed: {x2 - x1}px x {y2 - y1}px")

    def undo_crop(self):
        self.crop_rect = None
        self.original_crop_rect = None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_pos)

        # Restore full frame dimensions
        if self.cap:
            self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.resize_canvas()
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        self.set_status("Crop removed")
        self.show_frame()

    def rotate_video(self):
        self.rotation_angle = (self.rotation_angle + 90) % 360
        self.resize_canvas()
        self.show_frame()
        self.set_status(f"Video rotated {self.rotation_angle} degrees")

    def update_time_label(self):
        current_time = self.frame_pos / self.fps
        total_time = self.total_frames / self.fps
        self.time_label.config(text=f"{self.format_time(current_time)} / {self.format_time(total_time)}")

    def jump_to_start_entry(self, event=None):
        if not self.cap:
            return
        try:
            frame = int(self.start_frame_var.get())
            frame = max(0, min(frame, self.total_frames - 1))
            self.frame_pos = frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            self.timeline.set(frame)
            self.show_frame()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid integer frame number for Start.")

    def jump_to_end_entry(self, event=None):
        if not self.cap:
            return
        try:
            frame = int(self.end_frame_var.get())
            frame = max(0, min(frame, self.total_frames - 1))
            self.frame_pos = frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            self.timeline.set(frame)
            self.show_frame()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid integer frame number for End.")

    def run_dlc_analysis(self):
        if not self.dlc_csv_paths:
            messagebox.showinfo("DLC Export", "Please select a directory to save your video for analysis.")
            self.save_path = self.trim_and_export()
            
            videos_dir = os.path.join(self.save_path, 'videos')
            video_file = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
            video_path = os.path.join(videos_dir, video_file[0])

            self.load_video(video_path)
            dlc_runner.run_dlc_analysis(self)

            self.update_labeled_video_list()

    def update_labeled_video_list(self):
        labeled_videos = dlc_runner.show_labeled_videos(self)

        self.labeled_video_map.clear()
        display_names = []

        # Filter and map the labeled videos
        for path in labeled_videos:
            filename = os.path.basename(path)
            for part in ['Arms', 'Ears', 'Feet', 'Hands', 'String', 'Torso']:
                if part.lower() in filename.lower():
                    self.labeled_video_map[part] = path
                    display_names.append(part)
                    break

        if display_names:
            self.labeled_video_dropdown['values'] = display_names
            self.labeled_video_dropdown.current(0)  # Select first item by default
            self.set_status(f"Labeled videos found: {', '.join(display_names)}")
        else:
            self.set_status("No matching labeled videos.")

    def open_selected_labeled_video(self, event=None):
        selected_part = self.labeled_video_var.get()

        if selected_part not in self.labeled_video_map:
            self.set_status(f"No labeled video found for part: {selected_part}")
            return

        video_path = self.labeled_video_map[selected_part]

        if not os.path.exists(video_path):
            self.set_status(f"Video file does not exist: {video_path}")
            return

        # Load labeled video for preview only, without overwriting internal video_path
        self.load_video(video_path, update_original=False)

    def plot_results(self):
        if not self.dlc_csv_paths:
            messagebox.showerror("Error", "No DLC CSV found")
            return

        self.save_path = filedialog.askdirectory(title="Select Folder to Save Analysis Output")
        if not self.save_path: return

        self.set_status("Plotting results...")

        # if self.scale_factor None:
        #     save_path = f"{self.save_path}/analysis_output_units"
        # else:
        #     save_path = f"{self.save_path}/analysis_output_pixels"


        height = self.original_crop_rect if self.original_crop_rect is not None else self.original_height

        analysis.run_analysis(video_path=self.video_path, dlc_paths=self.dlc_csv_paths, save_dir=self.save_path, session_id=self.session_id, fps=self.fps, total_frames=self.total_frames,
                       scale_factor=self.scale_factor, height=height, show_plot=self.show_plot.get())


        self.set_status(f"Analysis saved to {self.save_path}")

    def update_dlc_config_paths(self):
        dlc_runner.update_dlc_config_paths(self)

    @staticmethod
    def format_time(seconds):
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        millis = round((seconds % 1) * 1000)

        if millis == 1000:
            secs += 1
            millis = 0
        if secs == 60:
            mins += 1
            secs = 0

        return f"{mins:02d}:{secs:02d}.{millis:03d}"

if __name__ == "__main__":
    root = tk.Tk()
    processor = PreProcessor(root)
    root.mainloop()


