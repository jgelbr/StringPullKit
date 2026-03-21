import os
import json
from tkinter import filedialog, messagebox

CONFIG_CACHE_FILE = os.path.join(os.path.expanduser("~"), ".dlc_config_cache.json")

def load_config_cache():
    if os.path.exists(CONFIG_CACHE_FILE):
        try:
            with open(CONFIG_CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_config_cache(configs):
    with open(CONFIG_CACHE_FILE, "w") as f:
        json.dump(configs, f, indent=4)

def update_config_interactively(body_parts):
    configs = {}
    for part in body_parts:
        messagebox.showinfo("Config Selection", f"Select DLC config.yaml for {part}")
        path = filedialog.askopenfilename(
            title=f"Select config.yaml for {part}",
            filetypes=[("YAML files", "*.yaml")],
        )
        if path:
            configs[part] = path
        else:
            messagebox.showwarning("Config Skipped", f"No config selected for {part}")
    save_config_cache(configs)
    return configs
