import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import sys, os, threading, traceback
from pathlib import Path
import pandas as pd

# Import pipeline functions
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from load_file import sanitize_file
from search_local_database import search_local_database
from apply_annotation_filtering import run_pipeline
from missing_values import impute_missing_values
from normalization import normalize_by_internal_standards
from median_normalization import median_normalization
from loess_normalization import loess_normalization
from generate_final_file import create_final_outputs
from generate_annotation_plots import plot_results, plot_kendrick_mass_vs_defect
from merging import merge_simple, merge_best_polarity, merge_pre_norm_simple, merge_pre_norm_best_polarity

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
        w = int(self.winfo_screenwidth() * 0.75)
        h = int(self.winfo_screenheight() * 0.85)
        x = (self.winfo_screenwidth() // 2) - (w // 2)
        y = (self.winfo_screenheight() // 2) - (h // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")
        # Force window to accept the size we request
        self.update() # force Tk to accept geometry
        self.minsize(int(w * 0.95), int(h * 0.95))  # allow a large minimum
        self.maxsize(w, h) # optional — prevent overshrink

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
        # ensure canvas expands
        frame.pack(fill="both", expand=True)

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

    def _warn_if_groups_weak(self):
        """
        Pop a single warning dialog if groups/QC setup is weak.
        Non-blocking: user can continue.
        """
        try:
            if not self.output_folder:
                return

            group_file = Path(self.output_folder) / "sample_groups.csv"
            if not group_file.exists():
                return

            df_groups = pd.read_csv(group_file)
            warn_list = self._compute_group_warnings(df_groups)
            if not warn_list:
                return

            msg = "Group assignment warnings:\n\n- " + "\n- ".join(warn_list)

            # Avoid repeating the same popup every time the status refreshes
            if msg == self._last_group_warning_text:
                return
            self._last_group_warning_text = msg

            messagebox.showwarning("Group assignment warning", msg)

        except Exception:
            # Never block the run due to warning computation
            return
        
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
                "min_int": 4000,
                "rsd_qc_thresh": 30,             # QC RSD threshold after normalization (%)
                "min_detect_in_group": 85,       # Minimum % detected in at least one group
                "max_group_rsd_thresh": 50       # Max within-group RSD threshold after normalization (%)
            },
            "ms_search": {"mz_tol_da": 0.003, "mz_tol_ppm":3},
            "sample_type": "Mammalians",
            "is_dilution_factor": 10.0,
            "is_mix_type": "Avanti Splash Lipidomix",     # default
            "is_mix_file": None                           # if user picks "Other"

        }


        # State
        self.selected_pos_file = None
        self.output_folder = None
        self.stop_flag = False
        self.worker_thread = None
        # Track last group warning to avoid repeated popups
        self._last_group_warning_text = None

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

        # === Input Files (POS and NEG) ===
        self.pos_file_var = tk.StringVar()
        self.neg_file_var = tk.StringVar()

        # Instruction label for sample naming requirement
        tk.Label(
            root,
            text="Copy the positive and negative feature tables from Metaboscape (after MS/MS annotations) and paste into two Excel files.\nSample column names in both files must start with P_ for positive-ion samples or N_ for negative-ion samples.",
            bg="white",
            fg="black",
            font=("Segoe UI", 9, "italic"),
            justify="left",
            anchor="w"
        ).pack(padx=25, pady=(10, 5), fill="x")

        pos_frame = tk.Frame(root, bg="white")
        pos_frame.pack(padx=25, pady=(20, 5), fill="x")
        tk.Label(pos_frame, text="Positive ionization file:", bg="white").pack(side="left", padx=(0, 10))
        tk.Entry(pos_frame, textvariable=self.pos_file_var, width=140, state="readonly").pack(side="left", expand=True, fill="x", padx=(0, 10))
        ttk.Button(pos_frame, text="Browse", command=self.select_pos_file).pack(side="left")

        neg_frame = tk.Frame(root, bg="white")
        neg_frame.pack(padx=25, pady=(0, 10), fill="x")
        tk.Label(neg_frame, text="Negative ionization file:", bg="white").pack(side="left", padx=(0, 10))
        tk.Entry(neg_frame, textvariable=self.neg_file_var, width=140, state="readonly").pack(side="left", expand=True, fill="x", padx=(0, 10))
        ttk.Button(neg_frame, text="Browse", command=self.select_neg_file).pack(side="left")

        # === Output Folder ===
        output_frame = tk.Frame(root, bg="white")
        output_frame.pack(padx=25, pady=(0, 10), fill="x")
        tk.Entry(output_frame, textvariable=self.output_var, width=100, state="readonly").pack(side="left", expand=True, fill="x", padx=(0, 10))
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

        # Explanation message under the Assign Sample Groups button
        group_explain = tk.Label(
            root,
            text="A sample_groups.csv file will be created in the output folder and can be filled using Excel. QC replicates must be assigned to a 'QC' group.\n"
                "Each sample must have a Group and an Injection Order. Groups define biological or experimental conditions. Injection Order enables drift correction.\n"
                "Recommend ≥3 samples per group and ≥3 QCs. With fewer, QC RSD, drift correction, and group statistics may be unstable or skipped.\n",
            bg="white",
            fg="#444444",
            font=("Segoe UI", 9, "italic"),
            justify="left",
            anchor="w"
        )
        group_explain.pack(padx=25, pady=(0, 5), fill="x")


        # Group status label
        self.group_status_var = tk.StringVar(value="Groups not assigned ❌")
        self.group_status_label = tk.Label(
            top_frame,
            textvariable=self.group_status_var,
            bg="white",
            fg="red",
            font=("Segoe UI", 9, "italic")
        )
        self.group_status_label.pack(side="left", padx=10)

        # Vertical separator for visual spacing
        ttk.Separator(top_frame, orient="vertical").pack(side="left", fill="y", padx=30)

        # Sample type label + dropdown (same line)
        tk.Label(top_frame, text="Sample Type:", bg="white").pack(side="left", padx=(0, 10))
        self.sample_type_var = tk.StringVar(value=self.config["sample_type"])
        tk.OptionMenu(top_frame, self.sample_type_var, "Mammalians", "Bacteria", "Fungi").pack(side="left")

        # ------------------------------------------------------------
        #   SETUP SECTION (Two-column layout: IS Left, Buttons Right)
        # ------------------------------------------------------------
        setup_label = tk.Label(
            root,
            text="🧩 Setup Section",
            bg="white",
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        )
        setup_label.pack(fill="x", padx=25, pady=(10, 5))
        ttk.Separator(root, orient="horizontal").pack(fill="x", padx=25, pady=(0, 20))

        # Outer frame as 2-column grid
        setup_outer = tk.Frame(root, bg="white")
        setup_outer.pack(fill="x", padx=25, pady=(0, 10))

        # ---------------- LEFT COLUMN ----------------
        left_col = tk.Frame(setup_outer, bg="white")
        left_col.grid(row=0, column=0, sticky="nw")

        tk.Label(
            left_col,
            text="Internal Standards Setup:",
            bg="white",
            font=("Segoe UI", 10, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))

        # IS mix selection
        tk.Label(left_col, text="IS Mix:", bg="white").grid(row=1, column=0, sticky="w")
        self.is_mix_var = tk.StringVar(value=self.config["is_mix_type"])
        self.is_mix_file_var = tk.StringVar(value="")

        mix_options = [
            "Avanti Splash Lipidomix",
            "BRG Internal Standard Mix",
            "Other (select file...)"
        ]

        is_mix_dropdown = ttk.Combobox(
            left_col,
            textvariable=self.is_mix_var,
            values=mix_options,
            width=35,
            state="readonly"
        )
        is_mix_dropdown.grid(row=1, column=1, sticky="w", padx=(10, 0))

        # # File path row
        # tk.Label(left_col, text="File (if 'other'):", bg="white").grid(row=2, column=0, sticky="w", pady=(8, 0))
        # self.is_mix_file_var = tk.StringVar(value="")
        # tk.Entry(left_col, textvariable=self.is_mix_file_var, width=48, state="readonly")\
        #     .grid(row=2, column=1, sticky="w", pady=(8, 0))

        # Dilution entry
        tk.Label(
            left_col,
            text="IS dilution factor (e.g., 20 for 1:20):",
            bg="white"
        ).grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.is_dilution_var = tk.DoubleVar(value=self.config["is_dilution_factor"])
        tk.Entry(left_col, textvariable=self.is_dilution_var, width=10)\
            .grid(row=3, column=1, sticky="w", padx=(10, 0), pady=(10, 0))

        # Dropdown handler
        def handle_is_mix_selection(event=None):
            selection = self.is_mix_var.get()
            prev_selection = self.config.get("is_mix_type", "Avanti Splash Lipidomix")

            if selection == "Other (select file...)":
                filepath = filedialog.askopenfilename(
                    title="Select Internal Standards File",
                    filetypes=[("Excel files", "*.xlsx *.xls")]
                )
                if filepath:
                    self.is_mix_file_var.set(filepath)
                    self.config["is_mix_file"] = filepath
                    self.config["is_mix_type"] = selection
                else:
                    # revert if cancelled
                    self.is_mix_var.set(prev_selection)
            else:
                self.is_mix_file_var.set("")
                self.config["is_mix_file"] = None
                self.config["is_mix_type"] = selection

        is_mix_dropdown.bind("<<ComboboxSelected>>", handle_is_mix_selection)

        # ---------------- RIGHT COLUMN ----------------
        right_col = tk.Frame(setup_outer, bg="white")
        right_col.grid(row=0, column=1, sticky="ne", padx=(50, 0))

        ttk.Button(
            right_col,
            text="Set up Data Cleansing",
            command=self.open_data_cleansing_window,
            width=22
        ).pack(pady=(0, 10))

        ttk.Button(
            right_col,
            text="Set up MS Search",
            command=self.open_ms_search_window,
            width=22
        ).pack()


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
            text=" Apply missing value substitution after filtering",
            variable=self.impute_var,
            anchor="w",
            justify="right"
        ).pack(fill="x", anchor="e", pady=(2, 15))

        self.normalize_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            checkbox_frame,
            text=" Normalize intensities by class-matched internal standards",
            variable=self.normalize_var,
            anchor="w",
            justify="right"
        ).pack(fill="x", anchor="e", pady=(2, 15))

        self.median_norm_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            checkbox_frame,
            text=" Apply within-class and global median normalization.\n WARNING: median normalization is not appropriate when sample matrices differ strongly or QCs are not pooled samples.",
            variable=self.median_norm_var,
            anchor="w",
            justify="left"
        ).pack(fill="x", anchor="e", pady=(2, 15))

        self.loess_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            checkbox_frame,
            text=" Apply drift correction (linear for <=4 QCs, LOESS for >4 QCs) - REQUIRES INJECTION ORDER IN 'ASSIGN SAMPLE GROUPS'.  \n WARNING: drift correction may degrade the dataset when QCs are not pooled samples.",
            variable=self.loess_var,
            anchor="w",
            justify="left"
        ).pack(fill="x", anchor="e", pady=(2, 10))

        # === Status + Run Section ===
        run_label = tk.Label(root, text="🚀 Run Section", bg="white",
                            font=("Segoe UI", 11, "bold"), anchor="w")
        run_label.pack(fill="x", padx=25, pady=(10, 5))
        ttk.Separator(root, orient="horizontal").pack(fill="x", padx=25, pady=(0, 15))

        # Status area (modernized)
        self.status_var = tk.StringVar(value="Ready")
        status_frame = tk.Frame(root, bg="white")
        status_frame.pack(fill="x", padx=25, pady=(5, 15))
        tk.Label(status_frame, textvariable=self.status_var, bg="white",
                fg="#0078D7", font=("Segoe UI", 9, "italic"), anchor="center").pack(fill="x")

        # === Control Buttons ===
        bottom_frame = tk.Frame(root, bg="white")
        bottom_frame.pack(pady=15)

        self.start_button = ttk.Button(bottom_frame, 
                                       text="Run MS search\n(local LipidMaps)", 
                                       command=self.start_thread, 
                                       width=18, 
                                       state="disabled")
        self.start_button.pack(side="left", padx=15)

        self.process_button = ttk.Button(bottom_frame, 
                                         text="Process existing raw\nsearch results", 
                                         command=self.start_process_thread, 
                                         width=18, 
                                         state="disabled")
        self.process_button.pack(side="left", padx=10)

        self.normalize_button = ttk.Button(
            bottom_frame,
            text="Run normalization\n(only)",
            command=self.start_normalize_thread,
            width=18,
            state="disabled"
        )
        self.normalize_button.pack(side="left", padx=10)

        self.merge_button = ttk.Button(
            bottom_frame,
            text="Merge polarities",
            width=18,
            command=self.merge_polarities,
            state="disabled"  # enabled once at least one final file exists
        )
        self.merge_button.pack(side="left", padx=10)
        
        self.stop_button = ttk.Button(
            bottom_frame, text="Stop", 
            command=self.stop_processing, 
            width=18, state="disabled")
        self.stop_button.pack(side="left", padx=10)

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
        self.stats_button.pack(side="left", padx=10)

        ttk.Button(nav_frame, 
                   text="Quit", 
                   command=self.quit_app, 
                   width=12).pack(side="left", padx=10)


    # ==========================================================
    # MERGING HELPERS
    # ========================================================== 
      
    def _merge_polarities_silent(self):
        """
        Internal helper: run all polarity-merging steps:
        1) normalized: simple + best
        2) pre-normalization: simple + best
        """
        if not self.output_folder:
            return
        root = Path(self.output_folder)
        try:
            print(f"\n\n[MERGE] Running simple polarity concatenation (FINAL + SEMI-QUANT) in {root}", flush=True)
            merge_simple(root)

            print(f"[MERGE] Running best-polarity selection (FINAL + SEMI-QUANT) in {root}", flush=True)
            merge_best_polarity(root)

            print(f"[MERGE] Running simple merging BEFORE normalization in {root}", flush=True)
            from merging import merge_pre_norm_simple, merge_pre_norm_best_polarity
            merge_pre_norm_simple(root)

            print(f"[MERGE] Running BEST-polarity BEFORE normalization in {root}", flush=True)
            merge_pre_norm_best_polarity(root)

            print("[MERGE] All merging completed.", flush=True)

        except Exception as e:
            print(f"[MERGE] Polarity merging failed: {e}", flush=True)

    def plot_merged_annotation_results(self):
        """
        Generate annotation plots for merged POS+NEG final files.
        """
                
        if not self.output_folder:
            return

        root = Path(self.output_folder)
        merged_plot_root = root / "debug_merging"
        merged_plot_root.mkdir(parents=True, exist_ok=True)
        
        print(f'\n[PLOT] Saving the merged annotation plots to: {merged_plot_root}\n', flush=True)
        
        plot_inputs = [
            ("Combined_", root / "Final_Annotated.csv"),
        ]

        for tag, csv_path in plot_inputs:
            print(f"[DEBUG] Checking merged CSV: {csv_path}", flush=True)
            print(f"[DEBUG] Exists? {csv_path.exists()}", flush=True)
            if not csv_path.exists():
                continue

            print(f"[INFO] Plotting merged annotation results from {csv_path}", flush=True)

            try:
                plot_results(
                    pol_tag="",
                    input_csv=csv_path,
                    output_folder=merged_plot_root,
                    suffix=f"_MERGED_{tag}"
                )
            except Exception as e:
                print(f"[WARNING] Failed merged annotation plots for {csv_path}: {e}", flush=True)

            try:
                plot_kendrick_mass_vs_defect(
                    input_csv=csv_path,
                    results_folder=merged_plot_root,
                    suffix=f"_MERGED_{tag}"
                )
            except Exception as e:
                print(f"[WARNING] Failed merged Kendrick plots for {csv_path}: {e}", flush=True)
                
    def merge_polarities(self):
        """
        Public GUI action: merge POS and NEG final files into combined outputs.
        Wraps _merge_polarities_silent() and shows a messagebox.
        """
        if not self.output_folder:
            messagebox.showwarning(
                "No Output Folder Selected",
                "Please select an output folder first."
            )
            return

        # Run the actual merge
        self._merge_polarities_silent()
        self.plot_merged_annotation_results()

        # Inform the user
        messagebox.showinfo(
            "Polarity merge complete",
            "Merged polarity files have been created in the output folder:\n\n"
            "debug_merging/\n"
            "  - Final_Annotated_simple_combination.csv\n"
            "  - Final_Annotated_semi_quant_simple_combination.csv\n"
            "  - Final_Unknowns_simple_combination.csv (if unknowns exist)\n\n"
            "root output folder:\n"
            "  - Final_Annotated.csv\n"
            "  - Final_Annotated_semi_quant.csv\n"
            "  - Final_Unknowns.csv (if unknowns exist)"
        )


    # ==========================================================
    # STATS PAGE HELPERS
    # ==========================================================

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
            StatisticsPage(self.root, self.output_folder, sample_type=self.sample_type_var.get())
        except Exception as e:
            messagebox.showerror("Error", f"Could not open statistics view:\n{e}")


    # ==========================================================
    # FILE / FOLDER SELECTION
    # ==========================================================
    def select_pos_file(self):
        filepath = filedialog.askopenfilename(
            title="Select POS Excel File",
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )
        if filepath:
            self.pos_file_var.set(filepath)
            self.selected_pos_file = filepath
            self.check_group_status()

    def select_neg_file(self):
        filepath = filedialog.askopenfilename(
            title="Select NEG Excel File",
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )
        if filepath:
            self.neg_file_var.set(filepath)
            self.selected_neg_file = filepath
            self.check_group_status()


    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder = folder
            self.output_var.set(folder)
            with open("last_output_path.txt", "w") as f:
                f.write(folder)
            self.check_group_status()
        
    def _compute_group_warnings(self, df_groups: pd.DataFrame) -> list[str]:
        """
        Return non-fatal warnings for group assignment quality:
          - no QC assigned
          - <3 samples in any non-QC group
        """
        warnings_list = []

        if df_groups is None or df_groups.empty:
            return ["sample_groups.csv is empty"]

        if "Sample" not in df_groups.columns or "Group" not in df_groups.columns:
            return ["sample_groups.csv missing required columns (Sample, Group)"]

        grp = df_groups["Group"].astype(str).str.strip()
        grp_upper = grp.str.upper()

        # QC count
        n_qc = int((grp_upper == "QC").sum())
        if n_qc == 0:
            warnings_list.append("No QCs assigned. QC-based filtering/scoring and drift correction will be unreliable or skipped.")
        elif n_qc < 3:
            warnings_list.append(f"Only {n_qc} QC sample(s) assigned. Recommend ≥3 QCs for stable QC RSD and drift correction.")

        # Group sizes (exclude QC)
        nonqc = grp[grp_upper != "QC"]
        if not nonqc.empty:
            counts = nonqc.value_counts()
            small = counts[counts < 3]
            if not small.empty:
                small_txt = ", ".join([f"{g} (n={int(n)})" for g, n in small.items()])
                warnings_list.append(
                    f"Group(s) with <3 samples: {small_txt}. "
                    "With n<3, within-group RSD filtering and statistics are not meaningful (n=1 gives undefined variance)."
                )
        return warnings_list
    
    # ==========================================================
    # GROUP MANAGEMENT
    # ==========================================================
    def check_group_status(self):
        """Update button states based on available files in the output folder."""
        if self.output_folder:
            output_path = Path(self.output_folder)
            group_file = output_path / "sample_groups.csv"
                

            # --- Check if sample groups are fully assigned ---
            groups_ok = False
            status_msg = ""
            status_color = "red"
            group_size_warning = False

            if group_file.exists():
                try:
                    df_groups = pd.read_csv(group_file)

                    # Check required columns
                    has_group_col = "Group" in df_groups.columns
                    has_inj_col = "Order" in df_groups.columns

                    if not has_group_col or not has_inj_col:
                        status_msg = "sample_groups.csv missing required columns"
                    else:
                        grp = df_groups["Group"].astype(str).str.strip()
                        inj = df_groups["Order"]

                        # 1. Missing group assignments
                        missing_grp = grp.isna() | (grp == "") | (grp.str.lower() == "nan")

                        # 2. Missing injection order
                        missing_inj = inj.isna() | (inj == "")

                        # 3. Injection order duplicates
                        dup_inj = inj[inj.duplicated()].unique()

                        # Hard failures
                        if missing_grp.any():
                            status_msg = "[WARNING] At least one sample is missing its group assignment. Data processing cannot proceed."
                        elif missing_inj.any():
                            status_msg = "[WARNING] At least one sample is missing its injection order. Data processing cannot proceed."
                        elif len(dup_inj) > 0:
                            status_msg = "[WARNING] Duplicate injection order numbers detected. Data processing cannot proceed."
                        else:
                            # Passed all hard checks
                            groups_ok = True
                            status_msg = "Groups and injection order assigned."
                            status_color = "green"

                            # Non-fatal warnings (QC + small groups)
                            warn_list = self._compute_group_warnings(df_groups)
                            if warn_list:
                                status_color = "orange"
                                # Show a compact warning in the status label
                                status_msg = "[WARNING] " + " | ".join(warn_list)

                except Exception:
                    status_msg = "Error reading sample_groups.csv"

            else:
                status_msg = "sample_groups.csv not found"

            # Update UI
            self.group_status_var.set(status_msg)
            self.group_status_var_label_color(status_color)
            self.start_button.config(state=("normal" if groups_ok else "disabled"))

                        # For now, disable process-existing and normalization-only because
            # the new per-mode layout does not use the old mixed debug files.
            self.process_button.config(state="disabled")
            self.normalize_button.config(state="disabled")

            # Check if POS or NEG has search results
            pos_final = (output_path / "POS" / "debug" / "Pos_1-Final_MS_results.csv")
            neg_final = (output_path / "NEG" / "debug" / "Neg_1-Final_MS_results.csv")
            if pos_final.exists() or neg_final.exists():
                self.process_button.config(state="enabled")
            else:
                self.stats_button.config(state="disabled")

            # Check if POS or NEG has imputed results
            pos_final = (output_path / "POS" / "debug" / "Pos_3-Final_annotated_results_imputed.csv")
            neg_final = (output_path / "NEG" / "debug" / "Neg_3-Final_annotated_results_imputed.csv")
            if pos_final.exists() or neg_final.exists():
                self.normalize_button.config(state="enabled")
            else:
                self.stats_button.config(state="disabled")


            # --- Enable or disable "Next: Statistics" and Merge buttons ---
            # Check if POS or NEG has final outputs
            pos_final = (output_path / "POS" / "Pos_Final_Annotated.csv")
            neg_final = (output_path / "NEG" / "Neg_Final_Annotated.csv")
            if pos_final.exists() or neg_final.exists():
                self.stats_button.config(state="normal")
                self.merge_button.config(state="normal")
            else:
                self.stats_button.config(state="disabled")
                self.merge_button.config(state="disabled")
        else:
            # No output folder selected — disable everything
            self.start_button.config(state="disabled")
            self.process_button.config(state="disabled")
            self.normalize_button.config(state="disabled")
            self.stats_button.config(state="disabled")
            self.merge_button.config(state="disabled")
            self.group_status_var.set("No output folder selected ⚠️")
            self.group_status_var_label_color("red")

    def group_status_var_label_color(self, color):
        try:
            self.group_status_label.config(fg=color)
        except Exception:
            pass

    def assign_groups(self):
        # Require at least one file + output folder
        pos_file = getattr(self, "selected_pos_file", None)
        neg_file = getattr(self, "selected_neg_file", None)

        if (not pos_file and not neg_file) or not self.output_folder:
            messagebox.showwarning(
                "Missing Input",
                "Please select at least one input file (POS and/or NEG) and an output folder first."
            )
            return

        try:
            sample_cols = []

            # --- Collect samples from POS file ---
            if pos_file:
                df_pos = pd.read_excel(pos_file)
                def _is_sample_col(c) -> bool:
                    # Excel headers can be int/float/NaN; normalize safely
                    if c is None or (isinstance(c, float) and pd.isna(c)):
                        return False
                    s = str(c).strip()
                    return s.startswith(("P_", "N_"))

                pos_cols = [str(c).strip() for c in df_pos.columns if _is_sample_col(c)]
                sample_cols.extend(pos_cols)

            # --- Collect samples from NEG file ---
            if neg_file:
                df_neg = pd.read_excel(neg_file)
                neg_cols = [str(c).strip() for c in df_neg.columns if _is_sample_col(c)]
                sample_cols.extend(neg_cols)

            # De-duplicate and sort
            sample_cols = sorted(dict.fromkeys(sample_cols))

            if not sample_cols:
                messagebox.showinfo(
                    "No Samples Detected",
                    "No columns starting with P_ or N_ were found in the selected files."
                )
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

            # then open the editor with POS + NEG samples
            self.group_window = GroupAssignmentWindow(
                self.root, sample_cols, self.output_folder,
                callback_on_save=self.check_group_status
            )

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
        tk.Label(win, text="Minimum average intensity across all samples and QCs:", bg="white").pack(pady=(10, 2))
        min_int_var = tk.DoubleVar(value=self.config["data_cleansing"]["min_int"])
        tk.Entry(win, textvariable=min_int_var).pack(pady=(2, 8))
        
        # --- NEW: Minimum % detected in at least one group ---
        tk.Label(win, text="Minimum % detected in at least one group (%):", bg="white").pack(pady=(10, 2))
        min_detect_var = tk.DoubleVar(value=self.config["data_cleansing"].get("min_detect_in_group", 50))
        tk.Entry(win, textvariable=min_detect_var).pack(pady=(2, 8))

        # --- NEW: QC RSD threshold (%) ---
        tk.Label(win, text="Maximum allowed RSD for QC features (%):\n(applied after normalization)", bg="white").pack(pady=(10, 2))
        rsd_qc_thresh_var = tk.DoubleVar(value=self.config["data_cleansing"].get("rsd_qc_thresh", 40))
        tk.Entry(win, textvariable=rsd_qc_thresh_var).pack(pady=(2, 8))
       

        # --- NEW: Maximum within-group RSD threshold (%) ---
        tk.Label(win, text="Maximum within-group RSD threshold (%):\n(applied after normalization)", bg="white").pack(pady=(10, 2))
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
        if (not self.selected_pos_file and not self.selected_neg_file) or not self.output_folder:
            messagebox.showwarning(
                "Missing Input",
                "Please select at least one input file (POS and/or NEG) and an output folder."
            )
            return

        self._warn_if_groups_weak()  # warning only

        self.stop_flag = False
        self.start_button.config(state="disabled")
        self.process_button.config(state="disabled")
        self.stop_button.config(state="normal")
        threading.Thread(target=self.run_local_search, daemon=True).start()


    def start_process_thread(self):
        if (not self.selected_pos_file and not self.selected_neg_file) or not self.output_folder:
            messagebox.showwarning(
                "Missing Input",
                "Please select at least one input file (POS and/or NEG) and an output folder."
            )
            return

        self._warn_if_groups_weak()  # warning only

        self.stop_flag = False
        self.start_button.config(state="disabled")
        self.process_button.config(state="disabled")
        self.stop_button.config(state="normal")
        threading.Thread(target=self.run_process_only, daemon=True).start()

    def start_normalize_thread(self):
        if not self.output_folder:
            messagebox.showwarning("Missing Output Folder", "Please select an output folder first.")
            return

        self._warn_if_groups_weak()

        self.stop_flag = False
        self.start_button.config(state="disabled")
        self.process_button.config(state="disabled")
        self.normalize_button.config(state="disabled")
        self.stop_button.config(state="normal")

        threading.Thread(target=self.run_normalization_only, daemon=True).start()


    def stop_processing(self):
        self.stop_flag = True
        self.update_status("Stopping...")

    def _get_is_dilution_factor(self) -> float:
        """Return a safe IS dilution factor (>0), defaulting to 12.0."""
        try:
            val = float(self.is_dilution_var.get())
            if val <= 0:
                return 12.0
            return val
        except Exception:
            return 12.0
        
    def _get_classmap_file(self) -> Path:
        """
        Choose the class-to-IS mapping file based on the IS mix selection.
        """
        mix = (self.config.get("is_mix_type") or "").strip()

        base = Path(__file__).resolve().parent.parent / "Appendix"

        if mix == "Avanti Splash Lipidomix":
            fname = "Class_to_internal_standards_Avanti_Splash_Lipidomix.csv"
        elif mix == "BRG Internal Standard Mix":
            fname = "Class_to_internal_standards_BRG_IS_mix.csv"
        else:
            raise ValueError(
                "IS Mix is set to 'Other (select file...)' but no class-to-IS mapping is defined for this option. "
                "Select Avanti or BRG, or implement a mapping-file picker for 'Other'."
            )

        classmap_file = base / fname
        if not classmap_file.exists():
            raise FileNotFoundError(
                f"Could not locate class-to-IS mapping file for mix '{mix}':\n{classmap_file}"
            )

        return classmap_file

    def _run_full_pipeline_for_mode(self, label, infile):
        """
        Run the complete pipeline (sanitization -> MS search -> filtering ->
        imputation -> IS normalization -> median -> LOESS -> final outputs)
        for a single mode (POS or NEG), into its own subfolder.
        """
        if not infile:
            return f"{label}: no input file provided, skipping.\n"

        # Mode-specific output folder: <output_folder>/POS or <output_folder>/NEG
        mode_folder = Path(self.output_folder) / label
        mode_folder.mkdir(parents=True, exist_ok=True)
        debug_folder = mode_folder / "debug"
        debug_folder.mkdir(parents=True, exist_ok=True)

        group_file = Path(self.output_folder) / "sample_groups.csv"

        # ---------------- Data cleansing ----------------
        self.update_status(f"Running data cleansing ({label})...")
        sanitized_path, clean_path, df_sanitized, pol_tag = sanitize_file(
            infile,
            mode_folder,
            mz_tol_ppm=self.config["data_cleansing"]["mz_tol_ppm"],
            rsd_thresh=self.config["data_cleansing"]["rsd_thresh"],
            min_int=self.config["data_cleansing"]["min_int"],
            rsd_qc_thresh=self.config["data_cleansing"]["rsd_qc_thresh"],
            min_detect_in_group=self.config["data_cleansing"]["min_detect_in_group"],
            max_group_rsd_thresh=self.config["data_cleansing"]["max_group_rsd_thresh"],
        )

        # ---------------- MS search ----------------
        self.update_status(f"Running MS search ({label})...")
        final_path_mode, _ = search_local_database(
            clean_path,
            mode_folder,
            pol_tag,
            mz_tolerance_Da=self.config["ms_search"]["mz_tol_da"],
            mz_tolerance_ppm=self.config["ms_search"]["mz_tol_ppm"],
            stop_flag=lambda: self.stop_flag,
        )

        # ---------------- Scoring / filtering ----------------
        self.update_status(f"Scoring and filtering MS search results ({label})...")
        sample_type = self.sample_type_var.get()
        scored_path, filtered_path = run_pipeline(
            input_csv=final_path_mode,
            output_folder=mode_folder,
            min_score=70,
            scoring_module=f"scoring_{sample_type.lower()}",
            plausibility_module=f"plausability_filtering_{sample_type.lower()}",
        )

        # ---------------- Missing value imputation ----------------
        message_extra = ""
        if self.impute_var.get():
            self.update_status(f"Applying missing value substitution ({label})...")
            try:
                imputed_path = impute_missing_values(
                    filtered_path,
                    group_file,
                    output_folder=mode_folder,
                    qc_rsd_threshold=self.config["data_cleansing"]["rsd_qc_thresh"],
                )
                message_extra += f"\n{label}: Imputed results:\n{imputed_path}"
            except Exception as e:
                imputed_path = filtered_path
                message_extra += f"\n{label}: Imputation skipped due to error: {e}"
        else:
            imputed_path = filtered_path

        # Ensure downstream steps always have an input
        norm_path = imputed_path

        # ---------------- IS normalization (gated by checkbox) ----------------
        if self.normalize_var.get():
            self.update_status(f"Normalizing by class-matched IS ({label})...")
            try:
                is_file = mode_folder / f"{label.capitalize()}_Internal_standards.csv"
                classmap_file = self._get_classmap_file()

                norm_path = normalize_by_internal_standards(
                    features_csv=imputed_path,
                    internal_standards_csv=is_file,
                    class_to_is_csv=classmap_file,
                    output_folder=mode_folder,
                    is_dilution_factor=self._get_is_dilution_factor(),
                    is_mix_file=self.config.get("is_mix_file"),
                    is_mix_type=self.config.get("is_mix_type"),
                )

                message_extra += f"\n{label}: IS-normalized results:\n{norm_path}"
            except Exception as e:
                # If anything goes wrong, fall back to the imputed data
                norm_path = imputed_path
                message_extra += f"\n{label}: IS normalization skipped due to error: {e}"
        else:
            # Checkbox off → skip normalization explicitly
            norm_path = imputed_path
            message_extra += f"\n{label}: IS normalization skipped (checkbox off)."

        # Default pass-through if median normalization is off
        median_norm_path = norm_path

        # ---------------- Median normalization ----------------
        if self.median_norm_var.get():
            self.update_status(f"Applying median normalization ({label})...")
            try:
                # Unknowns file: use normalized unknowns for this polarity
                unk_candidates = sorted(
                    debug_folder.glob(f"{pol_tag}5-Final_unknowns_normalized*.csv")
                )
                if not unk_candidates:
                    raise FileNotFoundError(
                        f"Could not locate {pol_tag}5-Final_unknowns_normalized*.csv in {debug_folder}"
                    )

                # Median-normalize main annotated file (4-)
                median_norm_path = median_normalization(
                    annotated_csv=norm_path,
                    unknowns_csv=unk_candidates[0],
                    sample_groups_csv=group_file,
                    output_folder=mode_folder,
                )

                # --- Apply median normalization to semi_quant file (if present) ---
                sq_file = sorted(
                    debug_folder.glob(f"{pol_tag}4b-Final_annotated_results_norm_semi_quant*.csv")
                )
                if sq_file:
                    median_sq = median_normalization(
                        annotated_csv=sq_file[0],
                        unknowns_csv=unk_candidates[0],
                        sample_groups_csv=group_file,
                        output_folder=mode_folder,
                        suffix="_semi_quant",
                    )

                message_extra += f"\n{label}: Median-normalized results:\n{median_norm_path}"
            except Exception as e:
                message_extra += f"\n{label}: Median normalization skipped due to error: {e}"


        # ---------------- LOESS / linear drift correction ----------------
        if self.loess_var.get():
            self.update_status(f"Applying LOESS drift correction ({label})...")
            try:
                # If median normalization was OFF, remove any stale median-normalized files for this mode
                if not self.median_norm_var.get():
                    for p in debug_folder.glob("8-Final_annotated_median_normalized*.csv"):
                        try:
                            p.unlink(missing_ok=True)
                        except Exception:
                            pass
                    for p in debug_folder.glob("8-Final_unknowns_median_normalized*.csv"):
                        try:
                            p.unlink(missing_ok=True)
                        except Exception:
                            pass

                if self.median_norm_var.get() and isinstance(median_norm_path, tuple):
                    # median_normalization returned explicit paths
                    median_annotated_path = median_norm_path[0]
                    median_unknowns_path = median_norm_path[1]
                else:
                    # Fall back to normalized (non-median) outputs in this mode's debug folder.
                    ann_candidates = sorted(
                        list(debug_folder.glob(pol_tag + "4-Final_annotated_results_normalized*.csv")) +
                        list(debug_folder.glob(pol_tag + "4b-Final_annotated_results_norm_semi_quant*.csv"))
                    )
                    unk_candidates = sorted(
                        debug_folder.glob(pol_tag + "5-Final_unknowns_normalized*.csv")
                    )
                    if not ann_candidates or not unk_candidates:
                        raise FileNotFoundError(
                            f"Could not locate normalized files for LOESS in {debug_folder}"
                        )
                    median_annotated_path = ann_candidates[0]
                    median_unknowns_path = unk_candidates[0]

                loess_norm_path = loess_normalization(
                    annotated_csv=median_annotated_path,
                    unknowns_csv=median_unknowns_path,
                    sample_groups_csv=group_file,
                    output_folder=mode_folder,
                )

                # --- LOESS-correct the semi_quant file (if present) ---
                sq_median = list(debug_folder.glob(pol_tag + "8b-Final_annotated_median_normalized_semi_quant*.csv"))
                if sq_median:
                    loess_sq = loess_normalization(
                        annotated_csv=sq_median[0],
                        unknowns_csv=median_unknowns_path,
                        sample_groups_csv=group_file,
                        output_folder=mode_folder,
                        suffix="_semi_quant"
                    )

                message_extra += f"\n{label}: LOESS-corrected results:\n{loess_norm_path}"
            except Exception as e:
                message_extra += f"\n{label}: LOESS correction skipped due to error: {e}"

        # ---------------- Final outputs ----------------
        self.update_status(f"Creating final output files ({label})...")
        annotated_path, annotated_path_semi, unknowns_path, method_used = create_final_outputs(
            mode_folder,
            rsd_qc_thresh=self.config["data_cleansing"]["rsd_qc_thresh"],
            max_group_rsd_thresh=self.config["data_cleansing"]["max_group_rsd_thresh"],
        )

        # ----------------------------------------------
        #      PLOT RESULTS (from generate_plots.py)
        # ----------------------------------------------
                    
        print("\n\n[INFO] Plotting annotation results after normalization and RSD filtering.", flush = True)
        try:
            plot_results(pol_tag, input_csv = annotated_path, output_folder= mode_folder, suffix=f"_{pol_tag}FINAL")
        except Exception as e:
            print(f"\n\n ======= Plot normalized and RSD filtered annotation results failed due to error {e}. ========\n\n", flush = True)
        try:
            plot_kendrick_mass_vs_defect(input_csv = annotated_path, results_folder = mode_folder, suffix=f"_{pol_tag}FINAL")
        except  Exception as e:
            print(f"\n\n ======= Plot kendrick mass defect for normalized and RSD filtered annotation results failed due to error {e}. ========\n\n", flush = True)

        summary = (
            f"{label} processing complete ({method_used}).\n\n"
            f"{label} sanitized file:\n{sanitized_path}\n\n"
            f"{label} RAW MS search results:\n{final_path_mode}\n\n"
            f"{label} scored results:\n{scored_path}\n\n"
            f"{label} filtered results:\n{filtered_path}\n\n"
            f"{label} final annotated:\n{annotated_path}\n\n"
            f"{label} final unknowns:\n{unknowns_path}"
            f"{message_extra}\n"
        )
        return summary

    def run_local_search(self):
        """Run the complete pipeline separately for POS and NEG (no mixing)."""
        try:
            if not self.output_folder:
                raise RuntimeError("No output folder selected.")

            summaries = []

            # POS
            summaries.append(self._run_full_pipeline_for_mode("POS", getattr(self, "selected_pos_file", None)))
            # NEG
            summaries.append(self._run_full_pipeline_for_mode("NEG", getattr(self, "selected_neg_file", None)))

            # Keep only non-empty summaries
            summaries = [s for s in summaries if s and "no input file provided" not in s]

            if not summaries:
                raise RuntimeError("No POS or NEG input file was provided.")

            final_message = ("\n" + ("-" * 60) + "\n").join(summaries)

            # After both polarities have been processed, run polarity merging silently.
            self._merge_polarities_silent()
            self.plot_merged_annotation_results() # plot merged annotation results

            self.finish_processing(final_message, error=False)


        except Exception as e:
            tb = traceback.format_exc()
            self.finish_processing(f"Error:\n{e}\n\n{tb}", error=True)

        finally:
            self.start_button.config(state="normal")
            self.process_button.config(state="normal")
            self.normalize_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.stop_flag = False


    def run_process_only(self):
        group_file = Path(self.output_folder) / "sample_groups.csv"
        print(f'\nStarted processing using the group assignment file in {group_file}.', flush = True)
        polarity = ""
        try:
            summaries = []
            for polarity in ("Pos", "Neg"):
                print(f' \n\n ------------------ {polarity} ----------------------\n\n', flush = True)

                filename = polarity + "_MS_search_results_RAW.csv"
                raw_file = Path(self.output_folder) / polarity / "debug" / filename
                sample_type = self.sample_type_var.get()
                output_path = Path(self.output_folder) / polarity
                output_debug_path = Path(self.output_folder) / polarity / "debug"
                print(f'Processing raw file {raw_file}. \nThe output folder is {output_path}. \nThe sample type is {sample_type}. \nThe sample groups are in {group_file}.', flush = True)

                # let the user see what’s happening before the long step
                self.update_status("Scoring and filtering raw search results...")

                scored_path, filtered_path = run_pipeline(
                    input_csv=raw_file, output_folder=output_path,
                    min_score=70,
                    scoring_module=f"scoring_{sample_type.lower()}",
                    plausibility_module=f"plausability_filtering_{sample_type.lower()}")

                # --- Missing value imputation ---
                if self.impute_var.get():
                    self.update_status("\nApplying missing value substitution...")
                    print(f'Applying missing value imputation using {filtered_path} and {group_file}. Saving results to {output_debug_path}.', flush = True)    
                    try:
                        imputed_path = impute_missing_values(
                            filtered_path,
                            group_file,
                            output_folder=output_path,
                            qc_rsd_threshold=self.config["data_cleansing"]["rsd_qc_thresh"]
                        )
                        message_extra = f"\n\nImputed results:\n{imputed_path}"
                    except Exception as e:
                        message_extra = f"\n\nImputation skipped due to error: {e}"
                else:
                    message_extra = ""

                # Ensure downstream steps have a valid input even if IS normalization is skipped
                print(f'Starting IS normalization with the file {imputed_path}', flush = True)
                norm_path = imputed_path
                
                # Default pass-through if median normalization is skipped
                median_norm_path = norm_path

                # === Class-matched internal standard normalization ===
                if self.normalize_var.get():
                    self.update_status("Normalizing intensities by class-matched internal standards...")
                    
                    try:
                        is_filename = polarity + "_Internal_standards.csv"
                        is_file = Path(self.output_folder) / polarity / is_filename
                        if not is_file.exists():
                            raise FileNotFoundError(
                                f"Could not locate internal standard file at: {is_file}.")
                        # Find the program root (the folder containing "Appendix")
                        classmap_file = self._get_classmap_file()
                        
                        norm_path = normalize_by_internal_standards(
                            features_csv=imputed_path,
                            internal_standards_csv=is_file,
                            class_to_is_csv=classmap_file,
                            output_folder=output_path,
                            is_dilution_factor=self._get_is_dilution_factor(),
                            is_mix_file=self.config.get("is_mix_file"),
                            is_mix_type=self.config.get("is_mix_type"),
                        )


                        message_extra += f"\n\nNormalized results:\n{norm_path}"
                    except Exception as e:
                        message_extra += f"\n\nNormalization skipped due to error: {e}"                


                # === Median normalization ===            
                if self.median_norm_var.get():
                    self.update_status("Applying within-class + global median normalization...")
                    try:
                        pol_tag = f"{polarity}_"
                        unk_filename = f"{polarity}_5-Final_unknowns_normalized.csv"
                        unk_filepath = output_debug_path / unk_filename

                        print(
                            f"Starting median normalization with the files {norm_path} and {unk_filepath}. "
                            f"Saving results to {output_debug_path}.",
                            flush=True,
                        )

                        # Median-normalize main annotated file (4-)
                        median_norm_path = median_normalization(
                            annotated_csv=norm_path,
                            unknowns_csv=unk_filepath,
                            sample_groups_csv=group_file,
                            output_folder=output_path,
                        )

                        # --- Apply median normalization to semi_quant file (if present) ---
                        sq_candidates = sorted(
                            output_debug_path.glob(f"{polarity}_4b-Final_annotated_results_norm_semi_quant*.csv")
                        )
                        if sq_candidates:
                            median_sq = median_normalization(
                                annotated_csv=sq_candidates[0],
                                unknowns_csv=unk_filepath,
                                sample_groups_csv=group_file,
                                output_folder=output_path,
                                suffix="_semi_quant",
                            )

                        message_extra += f"\n\nMedian-normalized results:\n{median_norm_path}"
                    except Exception as e:
                        message_extra += f"\n\nMedian normalization skipped due to error: {e}"
                        
                # === LOESS drift correction (optional) ===
                if self.loess_var.get():
                    self.update_status("Applying LOESS drift correction (QC-based)...")
                    try:
                        pol_tag = f"{polarity}_"

                        # If median normalization was OFF, remove any stale median-normalized outputs
                        final_filename = f"{polarity}_8-Final_annotated_median_normalized.csv"
                        unk_final_filename = f"{polarity}_8-Final_unknowns_median_normalized.csv"
                        if not self.median_norm_var.get():
                            for p in [
                                output_debug_path / final_filename,
                                output_debug_path / unk_final_filename,
                            ]:
                                try:
                                    p.unlink(missing_ok=True)
                                except Exception:
                                    pass

                        # Choose input for LOESS
                        if self.median_norm_var.get() and isinstance(median_norm_path, tuple):
                            median_annotated_path = median_norm_path[0]
                            median_unknowns_path = median_norm_path[1]
                        else:
                            median_annotated_path = output_debug_path / f"{polarity}_4-Final_annotated_results_normalized.csv"
                            median_unknowns_path = output_debug_path / f"{polarity}_5-Final_unknowns_normalized.csv"

                        print(
                            f"Starting drift correction with the files {median_annotated_path} and {median_unknowns_path}. "
                            f"Saving results to {output_debug_path}.",
                            flush=True,
                        )
                        loess_norm_path = loess_normalization(
                            annotated_csv=median_annotated_path,
                            unknowns_csv=median_unknowns_path,
                            sample_groups_csv=group_file,
                            output_folder=output_path,
                        )

                        # --- LOESS-correct the semi_quant file (if present) ---
                        sq_median = sorted(
                            output_debug_path.glob(f"{polarity}_8b-Final_annotated_median_normalized_semi_quant*.csv")
                        )
                        if sq_median:
                            loess_sq = loess_normalization(
                                annotated_csv=sq_median[0],
                                unknowns_csv=median_unknowns_path,
                                sample_groups_csv=group_file,
                                output_folder=output_path,
                                suffix="_semi_quant",
                            )

                        message_extra += f"\n\nLOESS-corrected results:\n{loess_norm_path}"
                    except Exception as e:
                        message_extra += f"\n\nLOESS correction skipped due to error: {e}"

                
                # === Prepare final annotated and unknown files ===
                self.update_status("Creating final output files...")
                
                annotated_path, annotated_path_semi, unknowns_path, method_used = create_final_outputs(
                    output_path,
                    rsd_qc_thresh=self.config["data_cleansing"]["rsd_qc_thresh"],
                    max_group_rsd_thresh=self.config["data_cleansing"]["max_group_rsd_thresh"]
                )

                # ----------------------------------------------
                #      PLOT RESULTS (from generate_plots.py)
                # ----------------------------------------------
                pol_tag = polarity + "_"  
                print("\n\n[INFO] Plotting annotation results after normalization and RSD filtering.", flush = True)
                try:
                    plot_results(pol_tag, input_csv = annotated_path, output_folder=output_path, suffix=f"_{pol_tag}FINAL")
                except Exception as e:
                    print(f"\n\n ======= Plot normalized and RSD filtered annotation results failed due to error {e}. ========\n\n", flush = True)
                try:
                    plot_kendrick_mass_vs_defect(input_csv = annotated_path, results_folder = output_path, suffix=f"_{pol_tag}FINAL")
                except  Exception as e:
                   print(f"\n\n ======= Plot kendrick mass defect for normalized and RSD filtered annotation results failed due to error {e}. ========\n\n", flush = True)


                summary = (
                    f"{polarity} normalization-only complete ({method_used}).\n\n"
                    f"{polarity} imputed file:\n{imputed_path}\n\n"
                    f"{polarity} final annotated:\n{annotated_path}\n\n"
                    f"{polarity} final unknowns:\n{unknowns_path}"
                    f"{message_extra}\n"
                )
                summaries.append(summary)

            if not summaries:
                raise RuntimeError(
                    "No per-mode imputed files found. Make sure POS/NEG runs have completed."
                )
            
            final_message = ("\n" + ("-" * 60) + "\n").join(summaries)

            # Run polarity merging after process-only mode finishes.
            self._merge_polarities_silent()
            self.plot_merged_annotation_results()    # plot merged annotation results
            
            self.finish_processing(final_message, error=False)


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
        """Run class-matched IS normalization and optional median/LOESS per mode (POS/NEG)."""
        try:
            output_root = Path(self.output_folder)
            group_file = output_root / "sample_groups.csv"
            print(f'Group file path: {group_file}')
            if not group_file.exists():
                raise FileNotFoundError(f"sample_groups.csv not found in {output_root}")

            # Locate Appendix/Class_to_internal_standards.csv once
            classmap_file = self._get_classmap_file()

            summaries = []

            for label in ("Pos", "Neg"):
                mode_folder = output_root / label
                debug_folder = mode_folder / "debug"
                pol_tag = f"{label}_"

                if not debug_folder.exists():
                    # Nothing processed for this mode
                    print(f'No debug folder for label ({debug_folder}).', flush = True)
                    continue

                filename = label + "_3-Final_annotated_results_imputed.csv"
                is_file = label + "_Internal_standards.csv"
                is_file_path = mode_folder / f"{label}_Internal_standards.csv"
                print(f'\n\n ---------------  Polarity: {label} ----------------- \n\n', flush = True)
                print(f'filename: {filename}', flush = True)
                print(f'internal standard filename: {is_file}', flush = True)
                                    
                # Imputed file now carries polarity tag and lives under the mode's debug folder.
                imputed_candidates = sorted(
                    debug_folder.glob(filename)
                )
                if not imputed_candidates:
                    # No imputed data for this mode; skip quietly
                    continue
                imputed_file = imputed_candidates[0]

                message_extra = ""
                norm_path = imputed_file

                # === Step 1: Class-matched internal standard normalization (per mode) ===
                if self.normalize_var.get():
                    self.update_status(
                        f"Normalizing intensities by class-matched internal standards ({label})..."
                    )
                    if not is_file_path.exists():
                        message_extra += (
                            f"\n{label}: IS normalization skipped "
                            f"(missing {is_file.name})."
                        )
                    else:
                        norm_path = normalize_by_internal_standards(
                            features_csv=imputed_file,
                            internal_standards_csv=is_file_path,
                            class_to_is_csv=classmap_file,
                            output_folder=mode_folder,
                            is_dilution_factor=self._get_is_dilution_factor(),
                            is_mix_file=self.config.get("is_mix_file"),
                            is_mix_type=self.config.get("is_mix_type"),
                        )

                        message_extra += f"\n{label}: IS-normalized results:\n{norm_path}"
                else:
                    message_extra += f"\n{label}: IS normalization skipped (checkbox off)."

                # === Step 2: Median normalization (optional) ===
                median_norm_path = norm_path
                if self.median_norm_var.get():
                    self.update_status(
                        f"Applying within-class + global median normalization ({label})..."
                    )
                    try:
                        unk_candidates = sorted(
                            debug_folder.glob(f"{pol_tag}5-Final_unknowns_normalized*.csv")
                        )
                        if not unk_candidates:
                            raise FileNotFoundError(
                                f"Could not locate {pol_tag}5-Final_unknowns_normalized*.csv in {debug_folder}"
                            )

                        # Median-normalize main annotated file (4-)
                        median_norm_path = median_normalization(
                            annotated_csv=norm_path,
                            unknowns_csv=unk_candidates[0],
                            sample_groups_csv=group_file,
                            output_folder=mode_folder,
                        )

                        # --- Apply median normalization to semi_quant file (if present) ---
                        sq_file = sorted(
                            debug_folder.glob(f"{pol_tag}4b-Final_annotated_results_norm_semi_quant*.csv")
                        )
                        if sq_file:
                            median_sq = median_normalization(
                                annotated_csv=sq_file[0],
                                unknowns_csv=unk_candidates[0],
                                sample_groups_csv=group_file,
                                output_folder=mode_folder,
                                suffix="_semi_quant",
                            )

                        message_extra += (
                            f"\n{label}: Median-normalized results:\n{median_norm_path}"
                        )
                    except Exception as e:
                        message_extra += (
                            f"\n{label}: Median normalization skipped due to error: {e}"
                        )

                # === Step 3: LOESS drift correction (optional, per mode) ===
                if self.loess_var.get():
                    self.update_status(
                        f"Applying LOESS drift correction (QC-based, {label})..."
                    )
                    try:
                        # If median normalization was OFF, remove stale median-normalized outputs for this mode
                        if not self.median_norm_var.get():
                            for p in debug_folder.glob(f"{pol_tag}8-Final_annotated_median_normalized*.csv"):
                                try:
                                    p.unlink(missing_ok=True)
                                except Exception:
                                    pass
                            for p in debug_folder.glob(f"{pol_tag}8-Final_unknowns_median_normalized*.csv"):
                                try:
                                    p.unlink(missing_ok=True)
                                except Exception:
                                    pass

                        # Choose inputs for LOESS
                        if self.median_norm_var.get() and isinstance(median_norm_path, tuple):
                            median_annotated_path = median_norm_path[0]
                            median_unknowns_path = median_norm_path[1]
                        else:
                            ann_candidates = sorted(
                                debug_folder.glob(f"{pol_tag}4-Final_annotated_results_normalized*.csv")
                            )
                            unk_candidates = sorted(
                                debug_folder.glob(f"{pol_tag}5-Final_unknowns_normalized*.csv")
                            )
                            if not ann_candidates or not unk_candidates:
                                raise FileNotFoundError(
                                    f"Could not locate normalized files for LOESS in {debug_folder}"
                                )
                            median_annotated_path = ann_candidates[0]
                            median_unknowns_path = unk_candidates[0]

                        loess_norm_path = loess_normalization(
                            annotated_csv=median_annotated_path,
                            unknowns_csv=median_unknowns_path,
                            sample_groups_csv=group_file,
                            output_folder=mode_folder,
                        )

                        # --- LOESS-correct the semi_quant file (if present) ---
                        sq_median = sorted(
                            debug_folder.glob(f"{pol_tag}8b-Final_annotated_median_normalized_semi_quant*.csv")
                        )
                        if sq_median:
                            loess_sq = loess_normalization(
                                annotated_csv=sq_median[0],
                                unknowns_csv=median_unknowns_path,
                                sample_groups_csv=group_file,
                                output_folder=mode_folder,
                                suffix="_semi_quant",
                            )

                        message_extra += (
                            f"\n{label}: LOESS-corrected results:\n{loess_norm_path}"
                        )
                    except Exception as e:
                        message_extra += (
                            f"\n{label}: LOESS correction skipped due to error: {e}"
                        )

                # === Step 4: Prepare final annotated and unknown files (per mode) ===
                self.update_status(f"Creating final output files ({label})...")
                annotated_path, annotated_path_semi_quant, unknowns_path, method_used = create_final_outputs(
                    mode_folder,
                    rsd_qc_thresh=self.config["data_cleansing"]["rsd_qc_thresh"],
                    max_group_rsd_thresh=self.config["data_cleansing"][
                        "max_group_rsd_thresh"
                    ],
                )

                # ----------------------------------------------
                #      PLOT RESULTS (from generate_plots.py)
                # ----------------------------------------------
                pol_tag = str(label) + "_"           
                print("\n\n[INFO] Plotting annotation results after normalization and RSD filtering.", flush = True)
                try:
                    plot_results(pol_tag, input_csv = annotated_path, output_folder= mode_folder, suffix=f"_{pol_tag}FINAL")
                except Exception as e:
                    print(f"\n\n ======= Plot normalized and RSD filtered annotation results failed due to error {e}. ========\n\n", flush = True)
                try:
                    plot_kendrick_mass_vs_defect(input_csv = annotated_path, results_folder = mode_folder, suffix=f"_{pol_tag}FINAL")
                except  Exception as e:
                    print(f"\n\n ======= Plot kendrick mass defect for normalized and RSD filtered annotation results failed due to error {e}. ========\n\n", flush = True)

                summary = (
                    f"{label} normalization-only complete ({method_used}).\n\n"
                    f"{label} imputed file:\n{imputed_file}\n\n"
                    f"{label} final annotated:\n{annotated_path}\n\n"
                    f"{label} final unknowns:\n{unknowns_path}"
                    f"{message_extra}\n"
                )
                summaries.append(summary)

            if not summaries:
                raise RuntimeError(
                    "No per-mode imputed files found. Make sure POS/NEG runs have completed."
                )

            final_message = ("\n" + ("-" * 60) + "\n").join(summaries)

            # Run polarity merging after normalization-only mode finishes.
            self._merge_polarities_silent()
            # Generate annotation plots for merged files
            self.plot_merged_annotation_results()

            self.finish_processing(final_message, error=False)

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

    # Force the window to open higher on the screen
    root.update_idletasks()
    w = 1700   # width of the window (adjust if needed)
    h = 820    # height of the window (adjust if needed)
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()

    # "higher" vertical placement → subtract 100–200 pixels
    x = (screen_w // 2) - (w // 2)
    y = (screen_h // 2) - (h // 2) - 150   # move up by x pixels

    root.geometry(f"{w}x{h}+{x}+{y}")
    root.minsize(w, h)
    root.update() # force Tk to accept geometry

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
    root.protocol("WM_DELETE_WINDOW", app.quit_app)
    app.stats_button.configure(style="Accent.TButton")
    app.merge_button.configure(style="Accent.TButton")
    app.start_button.configure(style="Accent.TButton")
    app.process_button.configure(style="Accent.TButton")

    root.mainloop()
