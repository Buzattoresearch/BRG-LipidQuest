import tkinter as tk
from GUI.view_start import MetaboscapeApp

def main():
    root = tk.Tk()
    app = MetaboscapeApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
