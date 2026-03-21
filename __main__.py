import sys

def main():
    if "--batch" in sys.argv:
        from stringpullkit.batch.batch_process import batch_process_sessions
        print("StringPullKit — batch mode")
        # other arguments?
    else:
        from stringpullkit.gui.PreProcessor import PreProcessor
        import tkinter as tk
        root = tk.Tk()
        PreProcessor(root)
        root.mainloop()

if __name__ == "__main__":
    main()
