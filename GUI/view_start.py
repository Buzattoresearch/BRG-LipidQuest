import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import sys, os, threading, traceback
from pathlib import Path
import pandas as pd

# Import pipeline functions
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from load_file import sanitize_file
from search_local_database import search_local_database
from apply_filtering import run_pipeline
from missing_values import impute_missing_values
from normalization import normalize_by_internal_standards
from median_normalization import median_normalization
from loess_normalization import loess_normalization
from generate_final_file import create_final_outputs
from GUI.view_statistics import StatisticsPage

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*is_sparse is deprecated.*",
    category=FutureWarning
)

# ==========================================================
# GROUP ASSIGNMENT WINDOW
# ==========================================================

class GroupAssignmentWindow(tk.Toplevel):
    def __init__(self, parent, sample_names, output_folder, callback_on_save):
        super().__init__(parent)
        self.title("Assign sample groups and injection order.")
        self.configure(bg="white")
        self.update_idletasks()
        w = 850
        h = 800
        x = (self.winfo_screenwidth() // 2) - (w // 2)
        y = (self.winfo_screenheight() // 2) - (h // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.sample_names = sample_names
        self.output_folder = Path(output_folder)
        self.group_vars = {}
        self.order_vars = {}
        self.group_options = ["QC"]
        self.callback_on_save = callback_on_save

        tk.Label(
            self,
            text="Assign a group and injection order to each sample.\nA CSV file can also be used for group assignment (output path/sample_groups.csv).",
            bg="white",
            font=("Segoe UI", 10, "bold")
        ).pack(pady=(10, 5))

        # Scrollable frame
        frame = tk.Frame(self, bg="white")
        frame.pack(padx=10, pady=(0, 10), fill="both", expand=True)

        canvas = tk.Canvas(frame, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        self.table = tk.Frame(canvas, bg="white")
        self.table.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.table, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Header row
        tk.Label(self.table, text="Sample Name", bg="white",
                 font=("Segoe UI", 10, "bold"), anchor="w").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        tk.Label(self.table, text="Group", bg="white",
                 font=("Segoe UI", 10, "bold")).grid(row=0, column=1, padx=5, pady=3)
        tk.Label(self.table, text="Injection Order", bg="white",
                 font=("Segoe UI", 10, "bold")).grid(row=0, column=2, padx=5, pady=3)

        # Load previous assignments (Group + Order)
        self.previous_mapping = {}
        groups_file = self.output_folder / "sample_groups.csv"
        if groups_file.exists():
            try:
                df_prev = pd.read_csv(groups_file)
                self.previous_mapping = {
                    s: (g, o)
                    for s, g, o in zip(
                        df_prev["Sample"],
                        df_prev["Group"],
                        df_prev.get("Order", [None] * len(df_prev))
                    )
                }
                for g in df_prev["Group"].dropna().unique():
                    if g not in self.group_options:
                        self.group_options.append(g)
            except Exception:
                pass

        # Populate table rows
        for i, s in enumerate(sample_names, start=1):
            tk.Label(self.table, text=s, bg="white", anchor="w").grid(row=i, column=0, sticky="w", padx=5, pady=2)

            # Group combobox
            prev_group = self.previous_mapping.get(s, ("", None))[0]
            group_var = tk.StringVar(value=prev_group)
            cb = ttk.Combobox(self.table, textvariable=group_var, values=self.group_options, width=20)
            cb.grid(row=i, column=1, padx=5, pady=2)
            self.group_vars[s] = group_var

            # Injection order entry
            prev_order = self.previous_mapping.get(s, ("", None))[1]
            order_var = tk.StringVar(value=str(prev_order) if prev_order is not None and not pd.isna(prev_order) else "")
            entry = tk.Entry(self.table, textvariable=order_var, width=10, justify="center")
            entry.grid(row=i, column=2, padx=5, pady=2)
            self.order_vars[s] = order_var

        # Add new group
        add_frame = tk.Frame(self, bg="white")
        add_frame.pack(pady=5)
        tk.Label(add_frame, text="Add new group:", bg="white").pack(side="left")
        self.new_group_var = tk.StringVar()
        tk.Entry(add_frame, textvariable=self.new_group_var, width=20).pack(side="left", padx=5)
        ttk.Button(add_frame, text="Add", command=self.add_new_group).pack(side="left")

        # Buttons
        btn_frame = tk.Frame(self, bg="white")
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Save", command=self.save_groups).pack(side="left", padx=10)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side="left", padx=10)

    def add_new_group(self):
        new_group = self.new_group_var.get().strip()
        if new_group and new_group not in self.group_options:
            self.group_options.append(new_group)
            for child in self.table.winfo_children():
                if isinstance(child, ttk.Combobox):
                    child.config(values=self.group_options)
            self.new_group_var.set("")

    def save_groups(self):
        # Gather all values
        rows = []
        for s in self.sample_names:
            group = self.group_vars[s].get().strip()
            order = self.order_vars[s].get().strip()
            if group == "":
                messagebox.showwarning("Incomplete", f"Please assign a group to sample: {s}")
                return
            # order can be empty, but if present, ensure numeric
            if order and not order.isdigit():
                messagebox.showwarning("Invalid Order", f"Injection order must be numeric for sample: {s}")
                return
            rows.append((s, group, int(order) if order else None))

        df = pd.DataFrame(rows, columns=["Sample", "Group", "Order"])
        out_path = self.output_folder / "sample_groups.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

        messagebox.showinfo("Saved", f"Group assignments saved to:\n{out_path}")
        self.callback_on_save()
        self.destroy()

# ==========================================================
# MAIN APP
# ==========================================================
class MetaboscapeApp:
    
    def update_status(self, message):
        """Safely update the blue status message from any thread."""
        self.root.after(0, lambda: self.status_var.set(message))
        self.root.update_idletasks()

    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder = folder
            self.output_var.set(folder)
            with open("last_output_path.txt", "w") as f:
                f.write(folder)
        self.check_group_status()

    def __init__(self, root):
        self.root = root
        self.root.title("LipidQuest - Metaboscape")
        self.root.configure(bg="white")

        # Default configuration
        self.config = {
            "data_cleansing": {
                "mz_tol_da": 0.003,
                "mz_tol_ppm": 3,
                "rsd_thresh": 0.05,
                "min_int": 1000,
                "rsd_qc_thresh": 30,             # QC RSD threshold (%)
                "min_detect_in_group": 80,       # Minimum % detected in at least one group
                "max_group_rsd_thresh": 50       # Max within-group RSD threshold (%)
            },
            "ms_search": {"mz_tol_da": 0.003, "mz_tol_ppm":3},
            "sample_type": "Mammalians"
        }

        # State
        self.selected_file = None
        self.output_folder = None
        self.stop_flag = False
        self.worker_thread = None

         # === Initialize StringVars early ===
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()

        # === Try restoring last output folder ===
        try:
            with open("last_output_path.txt") as f:
                last_path = f.read().strip()
                if last_path and os.path.isdir(last_path):
                    self.output_folder = last_path
                    self.output_var.set(last_path)
                else:
                    self.output_folder = None
        except FileNotFoundError:
            self.output_folder = None

        # === Input File ===
        input_frame = tk.Frame(root, bg="white")
        input_frame.pack(padx=25, pady=(20, 10), fill="x")
        tk.Entry(input_frame, textvariable=self.input_var, width=80, state="readonly").pack(side="left", expand=True, fill="x", padx=(0, 10))
        ttk.Button(input_frame, text="Select Excel File", command=self.select_file).pack(side="left")

        # === Output Folder ===
        output_frame = tk.Frame(root, bg="white")
        output_frame.pack(padx=25, pady=(0, 10), fill="x")
        tk.Entry(output_frame, textvariable=self.output_var, width=80, state="readonly").pack(side="left", expand=True, fill="x", padx=(0, 10))
        ttk.Button(output_frame, text="Select Output Folder", command=self.select_output_folder).pack(side="left")

        # === Assign Groups + Sample Type (same row) ===
        top_frame = tk.Frame(root, bg="white")
        top_frame.pack(pady=(20, 20)) #top and bottom padding for the group assingment section

        # Assign Sample Groups button
        ttk.Button(
            top_frame,
            text="Assign Sample Groups",
            command=self.assign_groups,
            width=22,
        ).pack(side="left", padx=5)

        # Group status label
        self.group_status_var = tk.StringVar(value="Groups not assigned ❌")
        tk.Label(
            top_frame,
            textvariable=self.group_status_var,
            bg="white",
            fg="red",
            font=("Segoe UI", 9, "italic")
        ).pack(side="left", padx=10)

        # Vertical separator for visual spacing
        ttk.Separator(top_frame, orient="vertical").pack(side="left", fill="y", padx=30)

        # Sample type label + dropdown (same line)
        tk.Label(top_frame, text="Sample Type:", bg="white").pack(side="left", padx=(0, 10))
        self.sample_type_var = tk.StringVar(value=self.config["sample_type"])
        tk.OptionMenu(top_frame, self.sample_type_var, "Mammalians", "Bacteria").pack(side="left")

        # === Section: Setup Buttons ===
        setup_label = tk.Label(root, text="🧩 Setup Section", bg="white",
                            font=("Segoe UI", 11, "bold"), anchor="w")
        setup_label.pack(fill="x", padx=25, pady=(15, 5))
        ttk.Separator(root, orient="horizontal").pack(fill="x", padx=25, pady=(0, 20))

        config_frame = tk.Frame(root, bg="white")
        config_frame.pack(pady=(0, 20))
        ttk.Button(config_frame, text="Set up Data Cleansing",
                command=self.open_data_cleansing_window, width=20).pack(side="left", padx=10)
        ttk.Button(config_frame, text="Set up MS Search",
                command=self.open_ms_search_window, width=20).pack(side="left", padx=10)

        # === Section: Processing Options ===
        options_label = tk.Label(root, text="⚙️ Processing Options", bg="white",
                                font=("Segoe UI", 11, "bold"), anchor="w")
        options_label.pack(fill="x", padx=25, pady=(10, 5))
        ttk.Separator(root, orient="horizontal").pack(fill="x", padx=25, pady=(0, 20))

        checkbox_frame = tk.Frame(root, bg="white")
        checkbox_frame.pack(fill="x", padx=25, pady=(0, 25))

        self.impute_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            checkbox_frame,
            text="Apply missing value substitution after filtering",
            variable=self.impute_var,
            anchor="w",
            justify="right"
        ).pack(fill="x", anchor="e", pady=(0, 10))

        self.normalize_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            checkbox_frame,
            text="Normalize intensities by class-matched internal standards",
            variable=self.normalize_var,
            anchor="w",
            justify="right"
        ).pack(fill="x", anchor="e", pady=(0, 10))

        self.median_norm_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            checkbox_frame,
            text="Apply within-class and global median normalization (experimental)",
            variable=self.median_norm_var,
            anchor="w",
            justify="right"
        ).pack(fill="x", anchor="e", pady=(0, 10))

        self.loess_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            checkbox_frame,
            text="Apply drift correction (requires injection order | linear for <=4 QCs, LOESS for >4 QCs)",
            variable=self.loess_var,
            anchor="w",
            justify="right"
        ).pack(fill="x", anchor="e", pady=(0, 10))

        # === Status + Run Section ===
        run_label = tk.Label(root, text="🚀 Run Section", bg="white",
                            font=("Segoe UI", 11, "bold"), anchor="w")
        run_label.pack(fill="x", padx=25, pady=(10, 5))
        ttk.Separator(root, orient="horizontal").pack(fill="x", padx=25, pady=(0, 20))

        # Status area (modernized)
        self.status_var = tk.StringVar(value="Ready")
        status_frame = tk.Frame(root, bg="white")
        status_frame.pack(fill="x", padx=25, pady=(5, 15))
        tk.Label(status_frame, textvariable=self.status_var, bg="white",
                fg="#0078D7", font=("Segoe UI", 9, "italic"), anchor="center").pack(fill="x")

        # === Control Buttons ===
        bottom_frame = tk.Frame(root, bg="white")
        bottom_frame.pack(pady=15)

        self.start_button = ttk.Button(bottom_frame, text="Run MS search\n(local LipidMaps)", command=self.start_thread, width=18, state="disabled")
        self.start_button.pack(side="left", padx=15)

        self.process_button = ttk.Button(bottom_frame, text="Process existing raw\nsearch results", command=self.start_process_thread, width=18, state="disabled")
        self.process_button.pack(side="left", padx=10)

        self.normalize_button = ttk.Button(
            bottom_frame,
            text="Run normalization\n(only)",
            command=self.start_normalize_thread,
            width=18,
            state="disabled"
        )
        self.normalize_button.pack(side="left", padx=10)


        self.stop_button = ttk.Button(bottom_frame, text="Stop", command=self.stop_processing, width=12, state="disabled")
        self.stop_button.pack(side="left", padx=10)

        ttk.Button(bottom_frame, text="Quit", command=self.quit_app, width=12).pack(side="left", padx=10)

        # === Navigation to Statistics ===
        nav_frame = tk.Frame(root, bg="white")
        nav_frame.pack(pady=(10, 20))
        self.stats_button = ttk.Button(
            nav_frame,
            text="Next: Statistics →",
            width=20,
            command=self.open_statistics_page,
            state="disabled"  # start disabled
        )
        self.stats_button.pack()

    def open_statistics_page(self):
        """Open the Statistics window (view_statistics.py)."""
        if not self.output_folder:
            messagebox.showwarning(
                "No Output Folder Selected",
                "Please select an output folder first before opening Statistics."
            )
            return

        try:
            from GUI.view_statistics import StatisticsPage
            self.root.withdraw()  # hide current window
            StatisticsPage(self.root, self.output_folder, sample_type=self.sample_type_var)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open statistics view:\n{e}")


    # ==========================================================
    # FILE / FOLDER SELECTION
    # ==========================================================
    def select_file(self):
        filepath = filedialog.askopenfilename(title="Select Excel or CSV File", filetypes=[("Excel/CSV files", "*.xlsx *.xls *.csv")])
        if filepath:
            self.selected_file = filepath
            self.input_var.set(filepath)
            with open("last_file_path.txt", "w") as f:
                f.write(filepath)
            self.check_group_status()

    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder = folder
            self.output_var.set(folder)
            with open("last_output_path.txt", "w") as f:
                f.write(folder)
            self.check_group_status()

    # ==========================================================
    # GROUP MANAGEMENT
    # ==========================================================
    def check_group_status(self):
        """Update button states based on available files in the output folder."""
        if self.output_folder:
            output_path = Path(self.output_folder)
            group_file = output_path / "sample_groups.csv"
            raw_search_file = output_path / "debug" / "MS_search_results_RAW.csv"

            # --- Check if sample groups are assigned ---
            if group_file.exists():
                self.group_status_var.set("Groups assigned ✅")
                self.group_status_var_label_color("green")
                # Enable "Run MS Search"
                self.start_button.config(state="normal")
                # Enable "Process existing raw search results" only if raw search file exists
                if raw_search_file.exists():
                    self.process_button.config(state="normal")
                else:
                    self.process_button.config(state="disabled")
            else:
                self.group_status_var.set("Groups not assigned ❌")
                self.group_status_var_label_color("red")
                self.start_button.config(state="disabled")
                self.process_button.config(state="disabled")

            # --- Enable normalization-only if imputed file exists ---
            imputed_file = output_path / "debug" / "4-Final_annotated_results_imputed_filtered.csv"
            if imputed_file.exists():
                self.normalize_button.config(state="normal")
            else:
                self.normalize_button.config(state="disabled")

            # --- Enable or disable "Next: Statistics" button ---
            final_file = output_path / "debug" / "4-Final_annotated_results_imputed_filtered.csv"
            alt_file = output_path / "Final_annotated_results.csv"
            if final_file.exists() or alt_file.exists():
                self.stats_button.config(state="normal")
            else:
                self.stats_button.config(state="disabled")
        else:
            # No output folder selected — disable everything
            self.start_button.config(state="disabled")
            self.process_button.config(state="disabled")
            self.normalize_button.config(state="disabled")
            self.stats_button.config(state="disabled")
            self.group_status_var.set("No output folder selected ⚠️")
            self.group_status_var_label_color("red")


    def group_status_var_label_color(self, color):
        for w in self.root.winfo_children():
            if isinstance(w, tk.Label) and w.cget("textvariable") == str(self.group_status_var):
                w.config(fg=color)

    def assign_groups(self):
        if not self.selected_file:
            messagebox.showwarning("No File Selected", "Please select an input file first.")
            return
        if not self.output_folder:
            messagebox.showwarning("No Output Folder", "Please select an output folder first.")
            return
        try:
            df = pd.read_excel(self.selected_file) if self.selected_file.endswith(('.xlsx', '.xls')) else pd.read_csv(self.selected_file)
            sample_cols = [c for c in df.columns if c.strip().startswith(("[POS", "[NEG"))]
            if not sample_cols:
                messagebox.showinfo("No Samples Detected", "No columns starting with [POS or [NEG] were found.")
                return
            
            # --- Write sample_groups.csv scaffold ---
            out_path = Path(self.output_folder) / "sample_groups.csv"

            # Preserve any existing assignments; add new samples with blanks
            prev = {}
            if out_path.exists():
                try:
                    df_prev = pd.read_csv(out_path)
                    prev = {
                        s: (g, o)
                        for s, g, o in zip(
                            df_prev["Sample"],
                            df_prev["Group"],
                            df_prev.get("Order", [None] * len(df_prev))
                        )
                    }
                except Exception:
                    prev = {}

            def _safe_int(x):
                try:
                    if x is None or (isinstance(x, float) and pd.isna(x)):
                        return None
                    xs = str(x).strip()
                    if xs == "":
                        return None
                    return int(float(xs))
                except Exception:
                    return None

            rows = []
            for s in sample_cols:
                g, o = prev.get(s, ("", None))
                o = _safe_int(o)
                rows.append((s, g, o))

            df = pd.DataFrame(rows, columns=["Sample", "Group", "Order"])
            df.to_csv(out_path, index=False, encoding="utf-8-sig")

            # reflect the new file in the UI immediately
            self.check_group_status()

            # then open the editor
            GroupAssignmentWindow(self.root, sample_cols, self.output_folder, callback_on_save=self.check_group_status)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")

    # ==========================================================
    # CONFIGURATION WINDOWS
    # ==========================================================
    def open_data_cleansing_window(self):
        win = tk.Toplevel(self.root)
        win.title("Data Cleansing Settings")
        win.configure(bg="white")
        win.geometry("420x500")

        # --- m/z tolerance (ppm) ---
        tk.Label(win, text="\nm/z tolerance (ppm):", bg="white").pack(pady=(10, 2))
        mz_ppm_var = tk.DoubleVar(value=self.config["data_cleansing"]["mz_tol_ppm"])
        tk.Entry(win, textvariable=mz_ppm_var).pack(pady=(2, 8))

        # --- RSD threshold to define flat peaks ---
        tk.Label(win, text="RSD threshold to define flat peaks:", bg="white").pack(pady=(10, 2))
        rsd_var = tk.DoubleVar(value=self.config["data_cleansing"]["rsd_thresh"])
        tk.Entry(win, textvariable=rsd_var).pack(pady=(2, 8))

        # --- Minimum intensity ---
        tk.Label(win, text="Minimum intensity:", bg="white").pack(pady=(10, 2))
        min_int_var = tk.DoubleVar(value=self.config["data_cleansing"]["min_int"])
        tk.Entry(win, textvariable=min_int_var).pack(pady=(2, 8))

        # --- NEW: QC RSD threshold (%) ---
        tk.Label(win, text="Maximum allowed RSD for QC features (%):", bg="white").pack(pady=(10, 2))
        rsd_qc_thresh_var = tk.DoubleVar(value=self.config["data_cleansing"].get("rsd_qc_thresh", 30))
        tk.Entry(win, textvariable=rsd_qc_thresh_var).pack(pady=(2, 8))

        # --- NEW: Minimum % detected in at least one group ---
        tk.Label(win, text="Minimum % detected in at least one group (%):", bg="white").pack(pady=(10, 2))
        min_detect_var = tk.DoubleVar(value=self.config["data_cleansing"].get("min_detect_in_group", 50))
        tk.Entry(win, textvariable=min_detect_var).pack(pady=(2, 8))

        # --- NEW: Maximum within-group RSD threshold (%) ---
        tk.Label(win, text="Maximum within-group RSD threshold (%):", bg="white").pack(pady=(10, 2))
        group_rsd_thresh_var = tk.DoubleVar(value=self.config["data_cleansing"].get("max_group_rsd_thresh", 40))
        tk.Entry(win, textvariable=group_rsd_thresh_var).pack(pady=(2, 8))

        # --- Save button ---
        def save():
            self.config["data_cleansing"]["mz_tol_ppm"] = mz_ppm_var.get()
            self.config["data_cleansing"]["rsd_thresh"] = rsd_var.get()
            self.config["data_cleansing"]["min_int"] = min_int_var.get()
            self.config["data_cleansing"]["rsd_qc_thresh"] = rsd_qc_thresh_var.get()
            self.config["data_cleansing"]["min_detect_in_group"] = min_detect_var.get()
            self.config["data_cleansing"]["max_group_rsd_thresh"] = group_rsd_thresh_var.get()
            win.destroy()

        ttk.Button(win, text="Save", command=save).pack(pady=15)

    def open_ms_search_window(self):
        win = tk.Toplevel(self.root)
        win.title("MS Search Settings")
        win.configure(bg="white")
        win.geometry("400x350")  # ← make window larger
        tk.Label(win, text="\nm/z tolerance (Da):", bg="white").pack(pady=(10,2)) # ← change pady to change the size of entry fields
        mz_da_var = tk.DoubleVar(value=self.config["ms_search"]["mz_tol_da"])
        tk.Entry(win, textvariable=mz_da_var).pack(pady=(10,2)) # ← change pady to change the size of entry fields
        tk.Label(win, text="\n\nm/z tolerance (ppm):", bg="white").pack(pady=(10,2)) # ← change pady to change the size of entry fields
        mz_ppm_var = tk.DoubleVar(value=self.config["ms_search"]["mz_tol_ppm"])
        tk.Entry(win, textvariable=mz_ppm_var).pack(pady=(10,2)) # ← change pady to change the size of entry fields
        def save():
            self.config["ms_search"]["mz_tol_da"] = mz_da_var.get()
            self.config["ms_search"]["mz_tol_ppm"] = mz_ppm_var.get()
            win.destroy()
        ttk.Button(win, text="Save", command=save).pack(pady=10)

    # ==========================================================
    # THREAD AND PROCESSING (unchanged)
    # ==========================================================
    def start_thread(self):
        if not self.selected_file or not self.output_folder:
            messagebox.showwarning("Missing Input", "Please select a file and output folder.")
            return
        self.stop_flag = False
        self.start_button.config(state="disabled")
        self.process_button.config(state="disabled")
        self.stop_button.config(state="normal")
        threading.Thread(target=self.run_local_search, daemon=True).start()

    def start_process_thread(self):
        if not self.selected_file or not self.output_folder:
            messagebox.showwarning("Missing Input", "Please select a file and output folder.")
            return
        self.stop_flag = False
        self.start_button.config(state="disabled")
        self.process_button.config(state="disabled")
        self.stop_button.config(state="normal")
        threading.Thread(target=self.run_process_only, daemon=True).start()

    def start_normalize_thread(self):
        """Start normalization-only step in a separate thread."""
        if not self.output_folder:
            messagebox.showwarning("Missing Output Folder", "Please select an output folder first.")
            return

        imputed_file = Path(self.output_folder) / "debug" /"4-Final_annotated_results_imputed_filtered.csv"
        if not imputed_file.exists():
            messagebox.showwarning("Missing Imputed File", "Cannot run normalization — Final_annotated_results_imputed_filtered.csv not found.")
            return

        self.stop_flag = False
        self.start_button.config(state="disabled")
        self.process_button.config(state="disabled")
        self.normalize_button.config(state="disabled")
        self.stop_button.config(state="normal")

        threading.Thread(target=self.run_normalization_only, daemon=True).start()

    def stop_processing(self):
        self.stop_flag = True
        self.update_status("Stopping...")
        

    def run_local_search(self):
        group_file = Path(self.output_folder) / "sample_groups.csv"
        try:
            self.update_status("Running data cleansing...")
            sanitized_path, clean_path, df_sanitized = sanitize_file(
                self.selected_file, self.output_folder,
                mz_tol_ppm=self.config["data_cleansing"]["mz_tol_ppm"],
                rsd_thresh=self.config["data_cleansing"]["rsd_thresh"],
                min_int=self.config["data_cleansing"]["min_int"],
                rsd_qc_thresh=self.config["data_cleansing"]["rsd_qc_thresh"],
                min_detect_in_group=self.config["data_cleansing"]["min_detect_in_group"],
                max_group_rsd_thresh=self.config["data_cleansing"]["max_group_rsd_thresh"]
            )

            self.update_status("Running MS search...")
            final_path, _ = search_local_database(
                clean_path, self.output_folder,
                mz_tolerance_Da=self.config["ms_search"]["mz_tol_da"],
                mz_tolerance_ppm=self.config["ms_search"]["mz_tol_ppm"],
                stop_flag=lambda: self.stop_flag)

            sample_type = self.sample_type_var.get()
            scored_path, filtered_path = run_pipeline(
                input_csv=final_path, output_folder=self.output_folder,
                min_score=70,
                scoring_module=f"scoring_{sample_type.lower()}",
                plausibility_module=f"plausability_filtering_{sample_type.lower()}")

            # --- Missing value imputation ---
            if self.impute_var.get():
                self.update_status("Applying missing value substitution...")
                
                try:
                    imputed_path = impute_missing_values(
                        filtered_path,
                        group_file,
                        output_folder=self.output_folder,
                        qc_rsd_threshold=self.config["data_cleansing"]["rsd_qc_thresh"]
                    )
                    message_extra = f"\n\nImputed results:\n{imputed_path}"
                except Exception as e:
                    message_extra = f"\n\nImputation skipped due to error: {e}"
            else:
                message_extra = ""

            import os
            print("\n=== PATH DEBUG INFO ===", flush = True)
            print("Current working directory:", os.getcwd(), flush = True)
            print("__file__:", __file__, flush = True)
            print("Resolved path:", Path(__file__).resolve(), flush = True)
            print("Parent:", Path(__file__).resolve().parent, flush = True)
            print("Parent.parent:", Path(__file__).resolve().parent.parent, flush = True)
            print("========================\n", flush = True)

            # === Class-matched internal standard normalization ===
            if self.normalize_var.get():
                self.update_status("Normalizing intensities by class-matched internal standards...")
                
                try:
                    is_file = Path(self.output_folder) / "Internal_standards.csv"
                    # Find the program root (the folder containing "Appendix")
                    # Find the Appendix file one level up from the GUI folder
                    classmap_file = Path(__file__).resolve().parent.parent / "Appendix" / "Class_to_internal_standards.csv"
                    if not classmap_file.exists():
                        raise FileNotFoundError(f"Could not locate Appendix/Class_to_internal_standards.csv at: {classmap_file}")
                    
                    norm_path = normalize_by_internal_standards(
                        features_csv=imputed_path,
                        internal_standards_csv=is_file,
                        class_to_is_csv=classmap_file,
                        output_folder=self.output_folder
                    )
                    message_extra += f"\n\nNormalized results:\n{norm_path}"
                except Exception as e:
                    message_extra += f"\n\nNormalization skipped due to error: {e}"
            
            # === Median normalization ===        
            if self.median_norm_var.get():
                self.update_status("Applying within-class + global median normalization...")
                
                try:
                    median_norm_path = median_normalization(
                        annotated_csv=norm_path,
                        unknowns_csv=Path(self.output_folder) / "debug" /"6-Final_unknowns.csv",
                        sample_groups_csv=group_file,
                        output_folder=self.output_folder
                    )
                    message_extra += f"\n\nMedian-normalized results:\n{median_norm_path}"
                except Exception as e:
                    message_extra += f"\n\nMedian normalization skipped due to error: {e}"
                    
            # === LOESS drift correction (optional) ===
            if self.loess_var.get():
                self.update_status("Applying LOESS drift correction (QC-based)...")
                
                try:
                    # median_normalization returns a tuple -> (annotated_path, unknowns_path)
                    if isinstance(median_norm_path, tuple):
                        median_annotated_path = median_norm_path[0]
                        median_unknowns_path = median_norm_path[1]
                    else:
                        median_annotated_path = median_norm_path
                        median_unknowns_path = Path(self.output_folder) / "debug" /"7-Final_unknowns_median_normalized.csv"

                    loess_norm_path = loess_normalization(
                        annotated_csv=median_annotated_path,
                        unknowns_csv=median_unknowns_path,
                        sample_groups_csv=group_file,
                        output_folder=self.output_folder
                    )

                    message_extra += f"\n\nLOESS-corrected results:\n{loess_norm_path}"
                except Exception as e:
                    message_extra += f"\n\nLOESS correction skipped due to error: {e}"
            
            # === Prepare final annotated and unknown files ===
            self.update_status("Creating final output files...")
            
            annotated_path, unknowns_path, method_used = create_final_outputs(self.output_folder)

            # --- Final summary popup ---
            self.finish_processing(
                f"Processing complete ({method_used}).\n\n"
                f"Sanitized file:\n{sanitized_path}\n\n"
                f"Raw search results:\n{final_path}\n\n"
                f"Scored results:\n{scored_path}\n\n"
                f"Filtered results:\n{filtered_path}\n\n"
                f"Final annotated:\n{annotated_path}\n\n"
                f"Final unknowns:\n{unknowns_path}"
            )

        except Exception as e:
            tb = traceback.format_exc()
            self.finish_processing(f"Error:\n{e}\n\n{tb}", error=True)

        finally:
            # Always restore UI even if finish_processing() failed
            self.start_button.config(state="normal")
            self.process_button.config(state="normal")
            self.normalize_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.stop_flag = False

    def run_process_only(self):
        group_file = Path(self.output_folder) / "sample_groups.csv"
        try:
            raw_file = Path(self.output_folder) / "debug" / "MS_search_results_RAW.csv"
            sample_type = self.sample_type_var.get()

            # let the user see what’s happening before the long step
            self.update_status("Scoring and filtering raw search results...")

            scored_path, filtered_path = run_pipeline(
                input_csv=raw_file, output_folder=self.output_folder,
                min_score=70,
                scoring_module=f"scoring_{sample_type.lower()}",
                plausibility_module=f"plausability_filtering_{sample_type.lower()}")

            # --- Missing value imputation ---
            if self.impute_var.get():
                self.update_status("Applying missing value substitution...")
                
                try:
                    imputed_path = impute_missing_values(
                        filtered_path,
                        group_file,
                        output_folder=self.output_folder,
                        qc_rsd_threshold=self.config["data_cleansing"]["rsd_qc_thresh"]
                    )
                    message_extra = f"\n\nImputed results:\n{imputed_path}"
                except Exception as e:
                    message_extra = f"\n\nImputation skipped due to error: {e}"
            else:
                message_extra = ""

            # === Class-matched internal standard normalization ===
            if self.normalize_var.get():
                self.update_status("Normalizing intensities by class-matched internal standards...")
                
                try:
                    is_file = Path(self.output_folder) / "Internal_standards.csv"
                    # Find the program root (the folder containing "Appendix")
                    # Find the Appendix file one level up from the GUI folder
                    classmap_file = Path(__file__).resolve().parent.parent / "Appendix" / "Class_to_internal_standards.csv"
                    if not classmap_file.exists():
                        raise FileNotFoundError(f"Could not locate Appendix/Class_to_internal_standards.csv at: {classmap_file}")
                    
                    norm_path = normalize_by_internal_standards(
                        features_csv=imputed_path,
                        internal_standards_csv=is_file,
                        class_to_is_csv=classmap_file,
                        output_folder=self.output_folder
                    )
                    message_extra += f"\n\nNormalized results:\n{norm_path}"
                except Exception as e:
                    message_extra += f"\n\nNormalization skipped due to error: {e}"

            # === Median normalization ===        
            if self.median_norm_var.get():
                self.update_status("Applying within-class + global median normalization...")
                
                try:
                    median_norm_path = median_normalization(
                        annotated_csv=norm_path,
                        unknowns_csv=Path(self.output_folder) / "debug" /"6-Final_unknowns.csv",
                        sample_groups_csv=group_file,
                        output_folder=self.output_folder
                    )
                    message_extra += f"\n\nMedian-normalized results:\n{median_norm_path}"
                except Exception as e:
                    message_extra += f"\n\nMedian normalization skipped due to error: {e}"
                    
            # === LOESS drift correction (optional) ===
            if self.loess_var.get():
                self.update_status("Applying LOESS drift correction (QC-based)...")
                
                try:
                    # median_normalization returns a tuple -> (annotated_path, unknowns_path)
                    if isinstance(median_norm_path, tuple):
                        median_annotated_path = median_norm_path[0]
                        median_unknowns_path = median_norm_path[1]
                    else:
                        median_annotated_path = median_norm_path
                        median_unknowns_path = Path(self.output_folder) / "debug" /"8-Final_unknowns_median_normalized.csv"

                    loess_norm_path = loess_normalization(
                        annotated_csv=median_annotated_path,
                        unknowns_csv=median_unknowns_path,
                        sample_groups_csv=group_file,
                        output_folder=self.output_folder
                    )

                    message_extra += f"\n\nLOESS-corrected results:\n{loess_norm_path}"
                except Exception as e:
                    message_extra += f"\n\nLOESS correction skipped due to error: {e}"
            
            # === Prepare final annotated and unknown files ===
            self.update_status("Creating final output files...")
            
            annotated_path, unknowns_path, method_used = create_final_outputs(self.output_folder)
        
            # --- Final summary popup ---
            self.finish_processing(
                f"Processing complete ({method_used}).\n\n"
                f"Scored results:\n{scored_path}\n\n"
                f"Filtered results:\n{filtered_path}\n\n"
                f"Final annotated:\n{annotated_path}\n\n"
                f"Final unknowns:\n{unknowns_path}"
                f"{message_extra}"
            )

        except Exception as e:
            tb = traceback.format_exc()
            self.finish_processing(f"Error:\n{e}\n\n{tb}", error=True)

        finally:
            # Always restore UI even if finish_processing() failed
            self.start_button.config(state="normal")
            self.process_button.config(state="normal")
            self.normalize_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.stop_flag = False

    def run_normalization_only(self):
        """Run class-matched IS normalization and optional median normalization."""
        try:
            imputed_file = Path(self.output_folder) / "debug" /"4-Final_annotated_results_imputed_filtered.csv"
            group_file = Path(self.output_folder) / "sample_groups.csv"
            is_file = Path(self.output_folder) / "Internal_standards.csv"

            # Locate Appendix/Class_to_internal_standards.csv
            p = Path(__file__).resolve().parent
            while p != p.parent:
                candidate = p / "Appendix" / "Class_to_internal_standards.csv"
                if candidate.exists():
                    classmap_file = candidate
                    break
                p = p.parent
            else:
                raise FileNotFoundError("Could not locate Appendix/Class_to_internal_standards.csv in parent directories.")

            message_extra = ""

            # === Step 1: Class-matched internal standard normalization ===
            if self.normalize_var.get():
                self.update_status("Normalizing intensities by class-matched internal standards...")
                
                norm_path = normalize_by_internal_standards(
                    features_csv=imputed_file,
                    internal_standards_csv=is_file,
                    class_to_is_csv=classmap_file,
                    output_folder=self.output_folder
                )
                message_extra += f"\n\nIS-normalized results:\n{norm_path}"
            else:
                norm_path = imputed_file  # skip to median normalization if user unchecked IS
                message_extra += "\n\nIS normalization skipped."

            # === Step 2: Median normalization (optional) ===
            if self.median_norm_var.get():
                self.update_status("Applying within-class + global median normalization...")
                
                median_norm_path = median_normalization(
                    annotated_csv=norm_path,
                    unknowns_csv=Path(self.output_folder) / "debug" /"6-Final_unknowns.csv",
                    sample_groups_csv=group_file,
                    output_folder=Path(self.output_folder)
                )
                message_extra += f"\n\nMedian-normalized results:\n{median_norm_path}"
                
            # === LOESS drift correction (optional) ===
            if self.loess_var.get():
                self.update_status("Applying LOESS drift correction (QC-based)...")
                
                try:
                    # median_normalization returns a tuple -> (annotated_path, unknowns_path)
                    if isinstance(median_norm_path, tuple):
                        median_annotated_path = median_norm_path[0]
                        median_unknowns_path = median_norm_path[1]
                    else:
                        median_annotated_path = median_norm_path
                        median_unknowns_path = Path(self.output_folder) / "debug" /"Final_unknowns_median_normalized.csv"

                    loess_norm_path = loess_normalization(
                        annotated_csv=median_annotated_path,
                        unknowns_csv=median_unknowns_path,
                        sample_groups_csv=group_file,
                        output_folder=self.output_folder
                    )
                    message_extra += f"\n\nLOESS-corrected results:\n{loess_norm_path}"
                except Exception as e:
                    message_extra += f"\n\nLOESS correction skipped due to error: {e}"
                        
            # === Prepare final annotated and unknown files ===
            self.update_status("Creating final output files...")
            
            annotated_path, unknowns_path, method_used = create_final_outputs(self.output_folder)
            
            self.finish_processing(
                f"Processing complete ({method_used}).\n\n"
                f"Final annotated:\n{annotated_path}\n\n"
                f"Final unknowns:\n{unknowns_path}"
            )
            
        except Exception as e:
            tb = traceback.format_exc()
            self.finish_processing(f"Normalization failed:\n{e}\n\n{tb}", error=True)

        finally:
            # Always restore UI even if finish_processing() failed
            self.start_button.config(state="normal")
            self.process_button.config(state="normal")
            self.normalize_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.stop_flag = False


    def finish_processing(self, message, error=False):
        """Safely finalize processing from a worker thread."""
        def _finish():
            # Re-enable all buttons and reset state
            self.start_button.config(state="normal")
            self.process_button.config(state="normal")
            self.normalize_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.stop_flag = False
            self.status_var.set("")

            # Show the messagebox safely in the main thread
            if error:
                messagebox.showerror("Error", message)
            else:
                messagebox.showinfo("Processing complete", message)

            self.root.update_idletasks()
            self.check_group_status()  # refresh buttons after process completes

        # Schedule it to run on the Tk main thread
        self.root.after(0, _finish)
    
    def quit_app(self):
        """Terminate the entire LipidQuest application cleanly."""
        try:
            self.root.destroy()
        except Exception:
            pass
        finally:
            import sys, os
            os._exit(0)

if __name__ == "__main__":
    root = tk.Tk()

    # === Global Style Configuration ===
    style = ttk.Style()
    root.tk.call("source", "azure.tcl") if Path("azure.tcl").exists() else None  # optional theme support

    # Use a clean base theme
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    # --- Colors ---
    primary_color = "#0078D7"     # Microsoft blue
    hover_color = "#005A9E"
    bg_color = "#FFFFFF"
    accent_bg = "#E0E0FF"
    text_color = "#000000"
    disabled_fg = "#888888"
    separator_color = "#CCCCCC"

    # --- General UI tweaks ---
    root.configure(bg=bg_color)
    style.configure(".", background=bg_color, foreground=text_color, font=("Segoe UI", 9))
    style.configure("TSeparator", background=separator_color)
    style.configure("TLabel", background=bg_color, foreground=text_color)
    style.configure("TCheckbutton", background=bg_color, foreground=text_color, font=("Segoe UI", 9))
    style.configure("TButton", font=("Segoe UI", 9), padding=4)

    # --- Accent button style (used for Run/Next) ---
    style.configure("Accent.TButton",
                    background=primary_color,
                    foreground="white",
                    font=("Segoe UI", 9, "bold"),
                    borderwidth=0,
                    focusthickness=3,
                    focuscolor=primary_color)
    style.map("Accent.TButton",
              background=[("active", hover_color)],
              foreground=[("disabled", disabled_fg)])

    # Apply accent style to key buttons after initialization
    app = MetaboscapeApp(root)
    app.stats_button.configure(style="Accent.TButton")
    app.start_button.configure(style="Accent.TButton")
    app.process_button.configure(style="Accent.TButton")

    root.mainloop()
