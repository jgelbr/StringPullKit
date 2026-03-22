from tkinter import ttk

# Central dictionary of theme colors
THEME_COLORS = {
    "Whimsical Pink": {
        "bg": "#ffe6f0",
        "fg": "#880e4f",
        "btn_bg": "#f8bbd0",
        "btn_fg": "#4a148c",
        "highlight": "#f48fb1"
    },
    "Retro Brown": {
        "bg": "#493628",
        "fg": "#cba35c",
        "btn_bg": "#754e1a",
        "btn_fg": "#f8e1b7",
        "highlight": "#b6cbbd"
    },
    "Sleek Modern": {
        "bg": "#1e1e1e",
        "fg": "#ffffff",
        "btn_bg": "#3a3a3a",
        "btn_fg": "#ffffff",
        "highlight": "#5c5c5c"
    },
    "Green Forest": {
        "bg": "#6b8a7a",
        "fg": "#254336",
        "btn_bg": "#254336",
        "btn_fg": "#b7b597",
        "highlight": "#dad3be"
    },
    "Ocean": {
        "bg": "#153448",
        "fg": "#dfd0b8",
        "btn_bg": "#948979",
        "btn_fg": "#153448",
        "highlight": "#3c5b6f"
    }
}


def get_theme_colors(theme_name: str):
    return THEME_COLORS.get(theme_name, THEME_COLORS[theme_name])


def apply_theme(style: ttk.Style, theme_name: str):
    colors = get_theme_colors(theme_name)

    if theme_name == "Whimsical Pink":
        style.theme_use("clam")
        font = ("Comic Sans MS", 10, "bold")
    elif theme_name == "Retro Brown":
        style.theme_use("alt")
        font = ("Courier New", 10, "bold")
    elif theme_name == "Green Forest":
        style.theme_use("alt")
        font = ("Helvetica", 10, "bold")
    elif theme_name == "Ocean":
        style.theme_use("clam")
        font = ("Times New Roman", 10, "bold")
    else:  # Sleek Modern or fallback
        style.theme_use("clam")
        font = ("Segoe UI", 10)

    # General
    style.configure(".", background=colors["bg"], foreground=colors["fg"], font=font)
    style.configure("TLabel", background=colors["bg"], foreground=colors["fg"])
    style.configure("TFrame", background=colors["bg"])

    # Buttons
    style.configure("TButton", background=colors["btn_bg"], foreground=colors["btn_fg"], font=font)
    style.map("TButton", background=[("active", colors["highlight"])])

    # Progress bar (optional)
    style.configure("TProgressbar", troughcolor=colors["bg"], background=colors["highlight"])
