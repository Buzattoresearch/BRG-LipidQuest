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

# ==========================================================
# GROUP ASSIGNMENT WINDOW
# ==========================================================

class GroupAssignmentWindow(tk.Toplevel):
    def __init__(self, parent, sample_names, output_folder, callback_on_save):
        super().__init__(parent)
        self.title("Assign Sample Groups and Injection Order")
        self.configure(bg="white")
        self.update_idletasks()
        w = 800
        h = 600
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
            text="Assign a group and injection order to each sample",
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
                 font=("Segoe UI", 9, "bold"), anchor="w").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        tk.Label(self.table, text="Group", bg="white",
                 font=("Segoe UI", 9, "bold")).grid(row=0, column=1, padx=5, pady=3)
        tk.Label(self.table, text="Injection Order", bg="white",
                 font=("Segoe UI", 9, "bold")).grid(row=0, column=2, padx=5, pady=3)

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
        tk.Button(add_frame, text="Add", command=self.add_new_group).pack(side="left")

        # Buttons
        btn_frame = tk.Frame(self, bg="white")
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Save", command=self.save_groups).pack(side="left", padx=10)
        tk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side="left", padx=10)

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
    def __init__(self, root):
        self.root = root
        self.root.title("LipidQuest - Metaboscape")
        self.root.configure(bg="white")

        # Default configuration
        self.config = {
            "data_cleansing": {
                "mz_tol_da": 0.003,
                "mz_tol_ppm": 3,
                "rsd_thresh": 0.080,
                "min_int": 1000,
                "rsd_qc_thresh": 30,             # QC RSD threshold (%)
                "min_detect_in_group": 80,       # Minimum % detected in at least one group
                "max_group_rsd_thresh": 50       # Max within-group RSD threshold (%)
            },
            "ms_search": {"mz_tol_da": 0.003, "mz_tol_ppm": 5},
            "sample_type": "Mammalians"
        }

        # State
        self.selected_file = None
        self.output_folder = None
        self.stop_flag = False
        self.worker_thread = None

        # === Input File ===
        input_frame = tk.Frame(root, bg="white")
        input_frame.pack(padx=25, pady=(20, 10), fill="x")

        self.input_var = tk.StringVar()
        tk.Entry(input_frame, textvariable=self.input_var, width=80, state="readonly").pack(side="left", expand=True, fill="x", padx=(0, 10))
        tk.Button(input_frame, text="Select Excel File", command=self.select_file).pack(side="left")

        # === Output Folder ===
        output_frame = tk.Frame(root, bg="white")
        output_frame.pack(padx=25, pady=(0, 10), fill="x")

        self.output_var = tk.StringVar()
        tk.Entry(output_frame, textvariable=self.output_var, width=80, state="readonly").pack(side="left", expand=True, fill="x", padx=(0, 10))
        tk.Button(output_frame, text="Select Output Folder", command=self.select_output_folder).pack(side="left")

        # === Assign Groups button (moved up!) ===
        group_frame = tk.Frame(root, bg="white")
        group_frame.pack(pady=(0, 15))
        tk.Button(group_frame, text="Assign Sample Groups", command=self.assign_groups, width=22, bg="#f0f0f0").pack(side="left", padx=5)
        self.group_status_var = tk.StringVar(value="Groups not assigned ❌")
        tk.Label(group_frame, textvariable=self.group_status_var, bg="white", fg="red", font=("Segoe UI", 9, "italic")).pack(side="left", padx=10)

        # === Sample Type ===
        type_frame = tk.Frame(root, bg="white")
        type_frame.pack(pady=(0, 10))
        tk.Label(type_frame, text="Sample Type:", bg="white").pack(side="left", padx=(0, 10))
        self.sample_type_var = tk.StringVar(value=self.config["sample_type"])
        tk.OptionMenu(type_frame, self.sample_type_var, "Mammalians", "Bacteria").pack(side="left")

        # === Configuration Buttons (restored) ===
        config_frame = tk.Frame(root, bg="white")
        config_frame.pack(pady=(5, 10))
        tk.Button(config_frame, text="Set up Data Cleansing", command=self.open_data_cleansing_window, width=20).pack(side="left", padx=10)
        tk.Button(config_frame, text="Set up MS Search", command=self.open_ms_search_window, width=20).pack(side="left", padx=10)

        # === Missing Value Substitution Option ===
        self.impute_var = tk.BooleanVar(value=True) # default is checked
        impute_checkbox = tk.Checkbutton(
            root,
            text="Apply missing value substitution after filtering",
            variable=self.impute_var,
            bg="white"
        )
        impute_checkbox.pack(pady=(0, 10))

        # === Internal Standard Normalization Option ===
        self.normalize_var = tk.BooleanVar(value=True) # default is checked
        normalize_checkbox = tk.Checkbutton(
            root,
            text="Normalize intensities by class-matched internal standards",
            variable=self.normalize_var,
            bg="white"
        )
        normalize_checkbox.pack(pady=(0, 10))
        
        # === Median normalization option ===
        self.median_norm_var = tk.BooleanVar(value=True)
        median_checkbox = tk.Checkbutton(
            root,
            text="Apply within-class and global median normalization (experimental)",
            variable=self.median_norm_var,
            bg="white"
        )
        median_checkbox.pack(pady=(0, 10))

        # === LOESS drift correction option ===
        self.loess_var = tk.BooleanVar(value=True)
        loess_checkbox = tk.Checkbutton(
            root,
            text="Apply drift correction (requires injection order | linear for <= 4 QCs, LOESS for >4 QCs)",
            variable=self.loess_var,
            bg="white"
        )
        loess_checkbox.pack(pady=(0, 10))

        # === Status Label ===
        self.status_var = tk.StringVar(value="")
        tk.Label(root, textvariable=self.status_var, bg="white", fg="blue").pack(pady=(5, 10))

        # === Control Buttons ===
        bottom_frame = tk.Frame(root, bg="white")
        bottom_frame.pack(pady=15)

        self.start_button = tk.Button(bottom_frame, text="Run MS search\n(local LipidMaps)", command=self.start_thread, width=18, state="disabled")
        self.start_button.pack(side="left", padx=15)

        self.process_button = tk.Button(bottom_frame, text="Process existing raw\nsearch results", command=self.start_process_thread, width=18, state="disabled")
        self.process_button.pack(side="left", padx=10)

        self.normalize_button = tk.Button(
            bottom_frame,
            text="Run normalization\n(only)",
            command=self.start_normalize_thread,
            width=18,
            state="disabled"
        )
        self.normalize_button.pack(side="left", padx=10)


        self.stop_button = tk.Button(bottom_frame, text="Stop", command=self.stop_processing, width=12, state="disabled")
        self.stop_button.pack(side="left", padx=10)

        tk.Button(bottom_frame, text="Quit", command=root.destroy, width=12).pack(side="left", padx=10)

    
    # ==========================================================
    # FILE / FOLDER SELECTION
    # ==========================================================
    def select_file(self):
        filepath = filedialog.askopenfilename(title="Select Excel or CSV File", filetypes=[("Excel/CSV files", "*.xlsx *.xls *.csv")])
        if filepath:
            self.selected_file = filepath
            self.input_var.set(filepath)
            self.check_group_status()

    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder = folder
            self.output_var.set(folder)
            self.check_group_status()

    # ==========================================================
    # GROUP MANAGEMENT
    # ==========================================================
    def check_group_status(self):
        if self.output_folder:
            group_file = Path(self.output_folder) / "sample_groups.csv"
            if group_file.exists():
                self.group_status_var.set("Groups assigned ✅")
                self.group_status_var_label_color("green")
                self.start_button.config(state="normal")
                self.process_button.config(state="normal")
            else:
                self.group_status_var.set("Groups not assigned ❌")
                self.group_status_var_label_color("red")
                self.start_button.config(state="disabled")
                self.process_button.config(state="disabled")

        # Enable normalization-only if imputed file exists and output folder is defined
        if self.output_folder:
            imputed_file = Path(self.output_folder) / "3-Final_search_results_imputed_filtered.csv"
            if imputed_file.exists():
                self.normalize_button.config(state="normal")
            else:
                self.normalize_button.config(state="disabled")
        else:
            self.normalize_button.config(state="disabled")

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

        tk.Button(win, text="Save", command=save).pack(pady=15)

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
        tk.Button(win, text="Save", command=save).pack(pady=10)

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

        imputed_file = Path(self.output_folder) / "3-Final_search_results_imputed_filtered.csv"
        if not imputed_file.exists():
            messagebox.showwarning("Missing Imputed File", "Cannot run normalization — Final_search_results_imputed_filtered.csv not found.")
            return

        self.stop_flag = False
        self.start_button.config(state="disabled")
        self.process_button.config(state="disabled")
        self.normalize_button.config(state="disabled")
        self.stop_button.config(state="normal")

        threading.Thread(target=self.run_normalization_only, daemon=True).start()

    def stop_processing(self):
        self.stop_flag = True
        self.status_var.set("Stopping...")
        self.root.update_idletasks()

    def run_local_search(self):
        group_file = Path(self.output_folder) / "sample_groups.csv"
        try:
            self.status_var.set("Running data cleansing...")
            sanitized_path, clean_path, df_sanitized = sanitize_file(
                self.selected_file, self.output_folder,
                mz_tol_ppm=self.config["data_cleansing"]["mz_tol_ppm"],
                rsd_thresh=self.config["data_cleansing"]["rsd_thresh"],
                min_int=self.config["data_cleansing"]["min_int"],
                rsd_qc_thresh=self.config["data_cleansing"]["rsd_qc_thresh"],
                min_detect_in_group=self.config["data_cleansing"]["min_detect_in_group"],
                max_group_rsd_thresh=self.config["data_cleansing"]["max_group_rsd_thresh"]
            )

            self.status_var.set("Running MS search...")
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
                self.status_var.set("Applying missing value substitution...")
                self.root.update_idletasks()
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
                self.status_var.set("Normalizing intensities by class-matched internal standards...")
                self.root.update_idletasks()
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
                self.status_var.set("Applying within-class + global median normalization...")
                self.root.update_idletasks()
                try:
                    median_norm_path = median_normalization(
                        annotated_csv=norm_path,
                        unknowns_csv=Path(self.output_folder) / "Final_unknowns.csv",
                        sample_groups_csv=group_file,
                        output_folder=self.output_folder
                    )
                    message_extra += f"\n\nMedian-normalized results:\n{median_norm_path}"
                except Exception as e:
                    message_extra += f"\n\nMedian normalization skipped due to error: {e}"
                    
            # === LOESS drift correction (optional) ===
            if self.loess_var.get():
                self.status_var.set("Applying LOESS drift correction (QC-based)...")
                self.root.update_idletasks()
                try:
                    # median_normalization returns a tuple -> (annotated_path, unknowns_path)
                    if isinstance(median_norm_path, tuple):
                        median_annotated_path = median_norm_path[0]
                        median_unknowns_path = median_norm_path[1]
                    else:
                        median_annotated_path = median_norm_path
                        median_unknowns_path = Path(self.output_folder) / "Final_unknowns_median_normalized.csv"

                    loess_norm_path = loess_normalization(
                        annotated_csv=median_annotated_path,
                        unknowns_csv=median_unknowns_path,
                        sample_groups_csv=group_file,
                        output_folder=self.output_folder
                    )

                    message_extra += f"\n\nLOESS-corrected results:\n{loess_norm_path}"
                except Exception as e:
                    message_extra += f"\n\nLOESS correction skipped due to error: {e}"
            
            # --- Final summary popup ---
            self.finish_processing(
                f"Sanitized file:\n{self.selected_file}\n\n"
                f"Raw search results:\n{final_path}\n\n"
                f"Scored results:\n{scored_path}\n\n"
                f"Filtered results:\n{filtered_path}"
                f"{message_extra}"
            )

        except Exception as e:
            tb = traceback.format_exc()
            self.finish_processing(f"Error:\n{e}\n\n{tb}", error=True)

    def run_process_only(self):
        group_file = Path(self.output_folder) / "sample_groups.csv"
        try:
            raw_file = Path(self.output_folder) / "debug" / "MS_search_results_RAW.csv"
            sample_type = self.sample_type_var.get()
            scored_path, filtered_path = run_pipeline(
                input_csv=raw_file, output_folder=self.output_folder,
                min_score=70,
                scoring_module=f"scoring_{sample_type.lower()}",
                plausibility_module=f"plausability_filtering_{sample_type.lower()}")

            # --- Missing value imputation ---
            if self.impute_var.get():
                self.status_var.set("Applying missing value substitution...")
                self.root.update_idletasks()
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
                self.status_var.set("Normalizing intensities by class-matched internal standards...")
                self.root.update_idletasks()
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
                self.status_var.set("Applying within-class + global median normalization...")
                self.root.update_idletasks()
                try:
                    median_norm_path = median_normalization(
                        annotated_csv=norm_path,
                        unknowns_csv=Path(self.output_folder) / "Final_unknowns.csv",
                        sample_groups_csv=group_file,
                        output_folder=self.output_folder
                    )
                    message_extra += f"\n\nMedian-normalized results:\n{median_norm_path}"
                except Exception as e:
                    message_extra += f"\n\nMedian normalization skipped due to error: {e}"
                    
            # === LOESS drift correction (optional) ===
            if self.loess_var.get():
                self.status_var.set("Applying LOESS drift correction (QC-based)...")
                self.root.update_idletasks()
                try:
                    # median_normalization returns a tuple -> (annotated_path, unknowns_path)
                    if isinstance(median_norm_path, tuple):
                        median_annotated_path = median_norm_path[0]
                        median_unknowns_path = median_norm_path[1]
                    else:
                        median_annotated_path = median_norm_path
                        median_unknowns_path = Path(self.output_folder) / "Final_unknowns_median_normalized.csv"

                    loess_norm_path = loess_normalization(
                        annotated_csv=median_annotated_path,
                        unknowns_csv=median_unknowns_path,
                        sample_groups_csv=group_file,
                        output_folder=self.output_folder
                    )

                    message_extra += f"\n\nLOESS-corrected results:\n{loess_norm_path}"
                except Exception as e:
                    message_extra += f"\n\nLOESS correction skipped due to error: {e}"
                    
            # --- Final summary popup ---
            self.finish_processing(
                f"Processed raw results:\n{filtered_path}"
                f"{message_extra}"
            )

        except Exception as e:
            tb = traceback.format_exc()
            self.finish_processing(f"Error:\n{e}\n\n{tb}", error=True)

    def run_normalization_only(self):
        """Run class-matched IS normalization and optional median normalization."""
        try:
            imputed_file = Path(self.output_folder) / "3-Final_search_results_imputed_filtered.csv"
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
                self.status_var.set("Normalizing intensities by class-matched internal standards...")
                self.root.update_idletasks()
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
                self.status_var.set("Applying within-class + global median normalization...")
                self.root.update_idletasks()
                median_norm_path = median_normalization(
                    annotated_csv=norm_path,
                    unknowns_csv=Path(self.output_folder) / "Final_unknowns.csv",
                    sample_groups_csv=group_file,
                    output_folder=Path(self.output_folder)
                )
                message_extra += f"\n\nMedian-normalized results:\n{median_norm_path}"
                
            # === LOESS drift correction (optional) ===
            if self.loess_var.get():
                self.status_var.set("Applying LOESS drift correction (QC-based)...")
                self.root.update_idletasks()
                try:
                    # median_normalization returns a tuple -> (annotated_path, unknowns_path)
                    if isinstance(median_norm_path, tuple):
                        median_annotated_path = median_norm_path[0]
                        median_unknowns_path = median_norm_path[1]
                    else:
                        median_annotated_path = median_norm_path
                        median_unknowns_path = Path(self.output_folder) / "Final_unknowns_median_normalized.csv"

                    loess_norm_path = loess_normalization(
                        annotated_csv=median_annotated_path,
                        unknowns_csv=median_unknowns_path,
                        sample_groups_csv=group_file,
                        output_folder=self.output_folder
                    )
                    message_extra += f"\n\nLOESS-corrected results:\n{loess_norm_path}"
                except Exception as e:
                    message_extra += f"\n\nLOESS correction skipped due to error: {e}"

            self.finish_processing(
                f"Normalization complete.{message_extra}"
            )

        except Exception as e:
            tb = traceback.format_exc()
            self.finish_processing(f"Normalization failed:\n{e}\n\n{tb}", error=True)


    def finish_processing(self, message, error=False):
        self.start_button.config(state="normal")
        self.process_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_var.set("")
        if error:
            messagebox.showerror("Error", message)
        else:
            messagebox.showinfo("Processing complete", message)


if __name__ == "__main__":
    root = tk.Tk()
    app = MetaboscapeApp(root)
    root.mainloop()
