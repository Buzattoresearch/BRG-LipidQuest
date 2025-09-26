import tkinter as tk
from tkinter import filedialog, messagebox
import sys, os
from pathlib import Path
import threading
import traceback

# Import sanitize_file and search_local_database
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from load_file import sanitize_file
from search_local_database import search_local_database
from apply_filtering import run_pipeline

import sys
sys.stdout.flush()

class MetaboscapeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LipidQuest - Metaboscape")
        self.root.configure(bg="white")

        # --- Input file row ---
        input_frame = tk.Frame(root, bg="white")
        input_frame.pack(padx=25, pady=(20, 10), fill="x")

        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(input_frame, textvariable=self.input_var, width=80, state="readonly")
        self.input_entry.pack(side="left", expand=True, fill="x", padx=(0, 10))

        select_input_button = tk.Button(input_frame, text="Select Excel File", command=self.select_file)
        select_input_button.pack(side="left")

        # --- Output folder row ---
        output_frame = tk.Frame(root, bg="white")
        output_frame.pack(padx=25, pady=(0, 20), fill="x")

        self.output_var = tk.StringVar()
        self.output_entry = tk.Entry(output_frame, textvariable=self.output_var, width=80, state="readonly")
        self.output_entry.pack(side="left", expand=True, fill="x", padx=(0, 10))

        select_output_button = tk.Button(output_frame, text="Select Output Folder", command=self.select_output_folder)
        select_output_button.pack(side="left")

        # --- Status label ---
        self.status_var = tk.StringVar(value="")
        self.status_label = tk.Label(root, textvariable=self.status_var, bg="white", fg="blue")
        self.status_label.pack(pady=(0, 10))

        # --- Start/Stop/Quit row ---
        bottom_frame = tk.Frame(root, bg="white")
        bottom_frame.pack(pady=15)

        self.start_button = tk.Button(bottom_frame, text="Run MS search\n(local LipidMaps)", command=self.start_thread, width=18)
        self.start_button.pack(side="left", padx=15)

        self.process_button = tk.Button(bottom_frame, text="Process raw\nsearch results", command=self.start_process_thread, width=18)
        self.process_button.pack(side="left", padx=10)

        self.stop_button = tk.Button(bottom_frame, text="Stop", command=self.stop_processing, width=12, state="disabled")
        self.stop_button.pack(side="left", padx=10)

        quit_button = tk.Button(bottom_frame, text="Quit", command=root.destroy, width=12)
        quit_button.pack(side="left", padx=10)

        self.selected_file = None
        self.output_folder = None
        self.stop_flag = False
        self.worker_thread = None

    def select_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Excel or CSV File",
            filetypes=[("Excel/CSV files", "*.xlsx *.xls *.csv")]
        )
        if filepath:
            self.selected_file = filepath
            self.input_var.set(filepath)

    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder = folder
            self.output_var.set(folder)

    def start_thread(self):
        """Start the background thread for processing (sanitize + search + filter)."""
        if not self.selected_file:
            messagebox.showwarning("No File Selected", "Please select an Excel file first.")
            return
        if not self.output_folder:
            messagebox.showwarning("No Output Folder", "Please select an output folder first.")
            return

        self.stop_flag = False
        self.start_button.config(state="disabled")
        self.process_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.worker_thread = threading.Thread(target=self.run_local_search, daemon=True)
        self.worker_thread.start()

    def start_process_thread(self):
        """Start a background thread to just process existing raw search results."""
        if not self.selected_file:
            messagebox.showwarning("No File Selected", "Please select a raw search results CSV file first.")
            return
        if not self.output_folder:
            messagebox.showwarning("No Output Folder", "Please select an output folder first.")
            return

        self.stop_flag = False
        self.start_button.config(state="disabled")
        self.process_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.worker_thread = threading.Thread(target=self.run_process_only, daemon=True)
        self.worker_thread.start()

    def stop_processing(self):
        """Signal the worker thread to stop."""
        self.stop_flag = True
        self.status_var.set("Stopping...")
        self.root.update_idletasks()

    def run_local_search(self):
        try:
            # Step 1: Sanitize file
            self.status_var.set("Loading data...")
            self.root.update_idletasks()
            sanitized_path, _ = sanitize_file(self.selected_file, self.output_folder)

            if self.stop_flag:
                self.finish_processing("Processing stopped after sanitization.")
                return

            # Step 2: Run local database search
            self.status_var.set("Running MS search... (this may take some time)")
            self.root.update_idletasks()
            final_path, _ = search_local_database(
                sanitized_path,
                self.output_folder,
                stop_flag=lambda: self.stop_flag
            )

            if self.stop_flag:
                self.finish_processing("ERROR: Processing stopped during MS search.")
                return

            # Step 3: Run filtering pipeline
            self.status_var.set("Applying filtering...")
            self.root.update_idletasks()
            scored_path, filtered_path = run_pipeline(
                input_csv=final_path,
                output_folder=self.output_folder,
                min_score=70
            )

            if self.stop_flag:
                self.finish_processing("ERROR: Processing stopped during filtering.")
                return

            # Step 4: Done
            self.finish_processing(
                f"Sanitized file:\n{sanitized_path}\n\n"
                f"Raw search results:\n{final_path}\n\n"
                f"Scored results:\n{scored_path}\n\n"
                f"Filtered results:\n{filtered_path}"
            )

        except PermissionError:
            self.finish_processing("The file is currently open in Excel. Please close it and try again.", error=True)
        except Exception as e:
            tb = traceback.format_exc()  # full traceback with file and line numbers
            self.finish_processing(f"Failed to process file:\n{e}\n\nTraceback:\n{tb}", error=True)

    def run_process_only(self):
        """Run only the filtering step on an existing raw search results file."""
        try:
            self.status_var.set("Applying filtering to raw results...")
            self.root.update_idletasks()

            # Explicit raw results filename
            raw_file = Path(self.output_folder) / "raw_ms_search_results.csv"
            if not raw_file.exists():
                self.finish_processing(
                    f"Expected raw results file not found:\n{raw_file}\n\n"
                    "Make sure you ran a search first or copy the raw results CSV here.",
                    error=True
                )
                return

            print(f"[DEBUG] Using raw results file: {raw_file}", flush=True)

            scored_path, filtered_path = run_pipeline(
                input_csv=raw_file,
                output_folder=self.output_folder,
                min_score=70
            )

            if self.stop_flag:
                self.finish_processing("ERROR: Processing stopped during filtering.")
                return

            self.finish_processing(
                f"Processed raw search results:\n{raw_file}\n\n"
                f"Scored results:\n{scored_path}\n\n"
                f"Filtered results:\n{filtered_path}"
            )

        except Exception as e:
            tb = traceback.format_exc()
            self.finish_processing(f"Failed to process raw results:\n{e}\n\nTraceback:\n{tb}", error=True)


    def finish_processing(self, message, error=False):
        """Reset buttons and show final status."""
        self.start_button.config(state="normal")
        self.process_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_var.set("")

        if error:
            messagebox.showerror("Error", message)
        elif "stopped" in message:
            messagebox.showinfo("Stopped", message)
        else:
            messagebox.showinfo("Processing complete", message)


if __name__ == "__main__":
    root = tk.Tk()
    app = MetaboscapeApp(root)
    root.mainloop()
