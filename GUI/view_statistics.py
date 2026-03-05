# GUI/view_statistics.py
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser
from pathlib import Path
from typing import Optional
import pandas as pd
import os, re, traceback, ctypes, inspect, time
import threading
import matplotlib.pyplot as plt
import matplotlib
import datetime, hashlib
import json

# analysis functions
from Stats.pca_analysis import run_pca
from Stats.plsda_analysis import run_plsda
from Stats.heatmap_analysis import run_heatmap
from Stats.volcano_analysis import run_volcano
from Stats.boxplots import run_boxplots
from Stats.violinplots import run_violinplots
from Stats.correlation_analysis import run_correlation_analysis
from Stats.class_distributions import run_from_stats as run_class_distributions
from Stats.summed_intensity_per_class import run_from_stats as run_class_sums
from Stats.class_violin_boxplots import run_from_stats as run_class_violin_box
from Stats.class_number_carbons_DB import run_from_stats as run_class_carbons_db

matplotlib.rcParams["figure.max_open_warning"] = 0  # suppress "too many open figures" warnings


class StatisticsPage(tk.Toplevel):
    """
    Statistics GUI
    - Prepares per-variant statistical datasets:
        Annotated: With_QCs / Without_QCs / HighConf_With_QCs / HighConf_Without_QCs
        Annotated (pre-normalization): BeforeNorm_With_QCs / BeforeNorm_Without_QCs
        Unknowns: With_QCs / Without_QCs
    - Runs analyses per selected dataset family, respecting tool requirements:
        * PCA runs on With_QCs variants
        * PLS-DA / Volcano / Heatmap / Boxplots / Violin run on Without_QCs variants
        * Boxplots/Violin skip HighConf variants
    """

    def __init__(self, parent, output_folder: Path, sample_type):
        super().__init__(parent)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.title("Statistics")
        self.geometry("1200x850")
        self.minsize(1000, 800)
        self.configure(bg="white")

        # --- Scrollable container setup ---
        container = tk.Frame(self, bg="white")
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container, bg="white", highlightthickness=0)
        vscroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)

        vscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # The real content frame (everything below lives inside this)
        self.main_frame = tk.Frame(canvas, bg="white")
        canvas.create_window((0, 0), window=self.main_frame, anchor="nw")

        def _on_frame_configure(_):
            canvas.configure(scrollregion=canvas.bbox("all"))
        self.main_frame.bind("<Configure>", _on_frame_configure)

        def _on_mousewheel(event):
            canvas.yview_scroll(-1 * int(event.delta / 120), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        self.output_folder = Path(output_folder)
        self.parent = parent
        self.sample_type = sample_type.get() if hasattr(sample_type, "get") else sample_type
        self.session_dir = None

        # User-chosen order for groups in plots (None = natural order)
        self.group_order: Optional[list[str]] = None
        # Colour chooser
        self.group_colors: dict[str, str] = {}  # {group: "#RRGGBB"}

        # --- single-run guard + HARD STOP support ---
        self._is_running = False
        self._run_lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None  # handle to kill immediately

        # Volcano threshold variables (defaults)
        self.var_fc  = tk.StringVar(value="1.5")
        self.var_fdr = tk.StringVar(value="0.10")
        self.var_p   = tk.StringVar(value="0.05")

        # Dataset selector state (exactly these 3 options)
        self.var_dataset = tk.StringVar(value="Annotated (normalized and merged)")

        # Try to reuse styles
        self._configure_local_style_if_needed()

        # --- Load data (for summary only) ---
        self.df_annotated = None
        self.df_unknowns = None
        self.df_before_norm = None
        self.df_groups = None
        self.missing_files = self._load_data_files()

        # === Header ===
        header = tk.Frame(self.main_frame, bg="white")
        header.pack(fill="x", pady=(14, 8), padx=24)

        ttk.Label(header, text="Statistics", style="Header.TLabel").pack(side="left")
        ttk.Label(self.main_frame, text=f"Output folder: {self.output_folder}", style="Subtle.TLabel").pack(
            anchor="w", padx=24, pady=(0, 12)
        )

        # === Summary ===
        summary_text = self._make_summary_text()
        self.summary_label = ttk.Label(self.main_frame, text=summary_text, style="Body.TLabel", justify="left")
        self.summary_label.pack(fill="x", padx=24, pady=(0, 14))

        ttk.Separator(self.main_frame, orient="horizontal").pack(fill="x", padx=24, pady=(0, 16))

        # === Prepare datasets ===
        prepare_frame = tk.Frame(self.main_frame, bg="white")
        prepare_frame.pack(pady=(4, 18))

        self.prepare_btn = ttk.Button(
            prepare_frame,
            text="Prepare Statistical Datasets",
            command=self.prepare_statistical_datasets,
            width=32,
            style="Accent.TButton"
        )
        self.prepare_btn.pack()

        # === Group selection + results folder (modal) ===
        row = tk.Frame(self.main_frame, bg="white")
        row.pack(fill="x", padx=24, pady=(6, 10))

        self.selection_label = ttk.Label(row, text="No selection yet", style="Subtle.TLabel")
        self.selection_label.pack(side="left")

        right_btns = tk.Frame(row, bg="white")
        right_btns.pack(side="right")

        ttk.Button(right_btns, text="Colors…", width=14, command=self.open_color_dialog).pack(side="right", padx=(8, 0))
        ttk.Button(right_btns, text="Select groups & output…", width=24, command=self.open_group_dialog).pack(side="right")

        # === Volcano thresholds (user-configurable) ===
        th = tk.Frame(self.main_frame, bg="white")
        th.pack(fill="x", padx=24, pady=(6, 6))

        ttk.Label(th, text="\nVolcano thresholds", style="Section.TLabel").grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 6))

        ttk.Label(th, text="Fold-change (FC ≥)", style="Body.TLabel").grid(row=1, column=0, sticky="e", padx=(0, 6))
        fc_entry = ttk.Entry(th, textvariable=self.var_fc, width=8); fc_entry.grid(row=1, column=1, sticky="w")

        ttk.Label(th, text="FDR p <", style="Body.TLabel").grid(row=1, column=2, sticky="e", padx=(16, 6))
        fdr_entry = ttk.Entry(th, textvariable=self.var_fdr, width=8); fdr_entry.grid(row=1, column=3, sticky="w")

        ttk.Label(th, text="raw p <", style="Body.TLabel").grid(row=1, column=4, sticky="e", padx=(16, 6))
        p_entry = ttk.Entry(th, textvariable=self.var_p, width=8); p_entry.grid(row=1, column=5, sticky="w")

        def _sanitize_thresholds(_evt=None):
            fc, fdr, p = self._get_volcano_thresholds()
            self.var_fc.set(f"{fc:.3g}"); self.var_fdr.set(f"{fdr:.3g}"); self.var_p.set(f"{p:.3g}")
        for e in (fc_entry, fdr_entry, p_entry):
            e.bind("<FocusOut>", _sanitize_thresholds)

        # === Dataset selector (3 options) ===
        ds = tk.Frame(self.main_frame, bg="white")
        ds.pack(fill="x", padx=24, pady=(8, 6))
        ttk.Label(ds, text="\nDataset selection", style="Section.TLabel").grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))
        ttk.Label(ds, text="Dataset:", style="Body.TLabel").grid(row=1, column=0, sticky="e", padx=(0, 6))
        ds_combo = ttk.Combobox(
            ds, textvariable=self.var_dataset, state="readonly", width=32,
            values=[
                "Annotated (normalized and merged)",
                "Annotated (pre-normalization, merged)",
                "Unknowns (normalized and merged)",
                "Annotated (POS only)",
                "Unknowns (POS only)",
                "Annotated (NEG only)",
                "Unknowns (NEG only)",
            ]
        )
        ds_combo.grid(row=1, column=1, sticky="w")

        # === Tools ===
        tools = tk.Frame(self.main_frame, bg="white")
        tools.pack(pady=(10, 28), padx=24, fill="x")

        ttk.Label(tools, text="\nAvailable Statistical Tools", style="Section.TLabel").grid(row=0, column=0, columnspan=5, pady=(0, 12), sticky="w")

        self.pca_button      = ttk.Button(tools, text="Run PCA", width=25, command=self.run_pca)
        self.plsda_button    = ttk.Button(tools, text="Run PLS-DA", width=25, command=self.run_plsda)
        self.heatmap_button  = ttk.Button(tools, text="Run Clustered Heatmap", width=25, command=self.run_heatmap)
        self.volcano_button  = ttk.Button(tools, text="Run Volcano", width=25, command=self.run_volcano)
        self.boxplots_button = ttk.Button(tools, text="Run Boxplots", width=25, command=self.run_boxplots)
        self.violin_button   = ttk.Button(tools, text="Run Violin Plots", width=25, command=self.run_violin)
        self.correlations_button = ttk.Button(tools, text="Run Correlations", width=25, command=self.run_correlation_analysis)
        self.classdist_button = ttk.Button(tools, text="Run Class Distributions", width=25, command=self.run_class_distributions)
        self.summint_button   = ttk.Button(tools, text="Run Summed Int. per Class", width=25, command=self.run_class_sums)
        self.classviolinbox_button   = ttk.Button(tools, text="Run Class Violin+Boxplots", width=25, command=self.run_class_violin_box)
        self.classcarbons_button     = ttk.Button(tools, text="Run Class Carbon Stacked Bars", width=25, command=self.run_class_carbons_db)

               
        # RUN ALL (cooperative cancel)
        self.runall_button   = ttk.Button(tools, text="RUN ALL", width=25, command=self.run_all)
        self.runall_button.grid(row=1, column=0, padx=8, pady=6)
        
        # STOP button (cooperative cancel)
        self.stop_button = ttk.Button(tools, text="⛔ STOP NOW", width=14, command=self.hard_stop_now)
        self.stop_button.state(["disabled"])
        self.stop_button.grid(row=1, column=1, padx=8, pady=6, sticky="w")
        
        ttk.Label(tools, text="\n", style="Section.TLabel").grid(row=2, column=0, columnspan=5, pady=(0, 12), sticky="w")
        
        self.pca_button.grid(row=3, column=0, padx=8, pady=6)
        self.plsda_button.grid(row=3, column=1, padx=8, pady=6)
        self.heatmap_button.grid(row=3, column=2, padx=8, pady=6)
        self.volcano_button.grid(row=3, column=3, padx=8, pady=6)
        self.boxplots_button.grid(row=4, column=0, padx=8, pady=6)
        self.violin_button.grid(row=4, column=1, padx=8, pady=6)
        self.correlations_button.grid(row=4, column=2, padx=8, pady=6)
        self.classdist_button.grid(row=5, column=0, padx=8, pady=6)
        self.summint_button.grid(row=5, column=1, padx=8, pady=6)
        self.classviolinbox_button.grid(row=5, column=2, padx=8, pady=6)
        self.classcarbons_button.grid(row=5, column=3, padx=8, pady=6)
        
        # Auto-prepare on first open
        self.after(200, self._auto_prepare_or_warn)

        # Disable buttons if required files are missing
        if self.missing_files:
            for btn in (self.pca_button, self.plsda_button, self.heatmap_button, self.volcano_button, self.boxplots_button, self.violin_button,
                self.correlations_button, self.classdist_button, self.summint_button, self.classviolinbox_button, self.classcarbons_button):
                btn.config(state="disabled")
            self._add_tooltip(tools, f"Missing files: {', '.join(self.missing_files)}")

        ttk.Separator(self.main_frame, orient="horizontal").pack(fill="x", padx=24, pady=(0, 16))

        # === Navigation ===
        nav = tk.Frame(self.main_frame, bg="white")
        nav.pack(pady=(4, 16))
        ttk.Button(nav, text="← Return to Processing", command=self.return_to_processing, width=22).pack(side="left", padx=8)
        ttk.Button(nav, text="Quit", command=self.quit_app, width=14).pack(side="left", padx=8)

    # ---------- style helpers ----------
    def _configure_local_style_if_needed(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        if "Accent.TButton" not in style.element_names():
            style.configure("Accent.TButton", background="#0078D7", foreground="white", font=("Segoe UI", 9, "bold"), padding=6)
            style.map("Accent.TButton", background=[("active", "#005A9E"), ("disabled", "#d0d0d0")], foreground=[("disabled", "#888888")])
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), background="white")
        style.configure("Section.TLabel", font=("Segoe UI", 11, "bold"), background="white")
        style.configure("Body.TLabel", font=("Segoe UI", 10), background="white")
        style.configure("Subtle.TLabel", font=("Segoe UI", 9, "italic"), foreground="#666", background="white")

    # ==========================================================
    # COLOUR CHOOSER
    # ==========================================================
    def open_color_dialog(self):
        self._load_palette()
        groups = self._active_groups()
        if not groups:
            messagebox.showwarning("No groups", "Define group selection first.")
            return

        dlg = tk.Toplevel(self); dlg.title("Group colors"); dlg.configure(bg="white"); dlg.transient(self); dlg.grab_set()
        frame = tk.Frame(dlg, bg="white"); frame.pack(fill="both", expand=True, padx=12, pady=10)

        tree = ttk.Treeview(frame, columns=("group", "color"), show="headings", height=min(12, len(groups)))
        tree.heading("group", text="Group"); tree.heading("color", text="Color")
        tree.column("group", width=220, anchor="w"); tree.column("color", width=120, anchor="center")
        tree.grid(row=0, column=0, columnspan=3, sticky="nsew", pady=(0,8))
        frame.grid_columnconfigure(0, weight=1); frame.grid_rowconfigure(0, weight=1)

        def _hex_or_default(g):
            c = self.group_colors.get(g)
            if isinstance(c, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", c):
                return c.upper()
            return None

        fallback_cycle = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])
        for i, g in enumerate(groups):
            hexcol = _hex_or_default(g)
            if hexcol is None:
                hexcol = matplotlib.colors.to_hex(fallback_cycle[i % len(fallback_cycle)]).upper() if fallback_cycle else "#1f77b4"
            tree.insert("", "end", values=(g, hexcol))

        def pick_for_selection():
            sel = tree.selection()
            if not sel: return
            for item in sel:
                g, current = tree.item(item, "values")
                _rgb, hexcol = colorchooser.askcolor(color=current, parent=dlg, title=f"Pick color for {g}")
                if hexcol: tree.set(item, "color", hexcol.upper())

        def reset_row():
            sel = tree.selection()
            if not sel: return
            for item in sel: tree.set(item, "color", "")

        def apply_and_close():
            pal = {}
            for item in tree.get_children(""):
                g, hexcol = tree.item(item, "values")
                hexcol = str(hexcol).strip()
                if re.fullmatch(r"#[0-9A-Fa-f]{6}", hexcol):
                    pal[g] = hexcol.upper()
            self.group_colors = pal; self._save_palette(); dlg.destroy()

        btns = tk.Frame(frame, bg="white"); btns.grid(row=1, column=0, sticky="e", pady=(4,0))
        ttk.Button(btns, text="Pick color", width=14, command=pick_for_selection).pack(side="left", padx=4)
        ttk.Button(btns, text="Clear", width=10, command=reset_row).pack(side="left", padx=4)
        ttk.Button(btns, text="OK", width=10, command=apply_and_close).pack(side="left", padx=4)
        ttk.Button(btns, text="Cancel", width=10, command=dlg.destroy).pack(side="left", padx=4)

        dlg.update_idletasks(); dlg.geometry(f"+{self.winfo_rootx()+120}+{self.winfo_rooty()+120}")

    # ==========================================================
    # DATA LOADING (for summary + guardrails)
    # ==========================================================
    def _load_data_files(self):
        """Load annotated, unknown, before-normalization, and group files if available. Return list of missing must-haves."""
        
        # ALL FEATURES
        candidates = {
            "Final_Annotated.csv": [
                # merged polarity
                self.output_folder / "Final_Annotated.csv",

                # polarity-specific
                self.output_folder / "POS" / "Pos_Final_Annotated.csv",
                self.output_folder / "NEG" / "Neg_Final_Annotated.csv",
            ],

            "Final_Unknowns.csv": [
                # merged polarity
                self.output_folder / "Final_Unknowns.csv",

                # polarity-specific
                self.output_folder / "POS" / "Pos_Final_Unknowns.csv",
                self.output_folder / "NEG" / "Neg_Final_Unknowns.csv",
            ],
            
            "Final_Annotated_Before_Normalization.csv": [
                self.output_folder / "Final_Annotated_Before_Normalization.csv",
            ],
            
            "sample_groups.csv": [
                self.output_folder / "sample_groups.csv",
                self.output_folder / "statistics" / "sample_groups_cleaned.csv",
            ],
        }
        mapping = {
            "Final_Annotated.csv": "df_annotated",
            "Final_Unknowns.csv": "df_unknowns",
            "Final_Annotated_Before_Normalization.csv": "df_before_norm",
            "sample_groups.csv": "df_groups",
        }

        missing = []
        for logical_name, paths in candidates.items():
            loaded = False
            for p in paths:
                if p.exists():
                    try:
                        setattr(self, mapping[logical_name], pd.read_csv(p))
                        loaded = True 
                        print(f'\nLoaded the file {p}.', flush = True)
                        break
                    except Exception as e:
                        messagebox.showwarning("File Error", f"Failed to read {p.name}:\n{e}", flush = True)
            if not loaded:
                missing.append(logical_name)
        # Only Annotated and groups are mandatory to start; others optional
        return [m for m in missing if m in ("Final_Annotated.csv", "sample_groups.csv")]

    def _make_summary_text(self):
        if self.missing_files:
            return (f"⚠ Some required files are missing:\n  - {', '.join(self.missing_files)}\n\n"
                    f"Please run the processing pipeline completely before proceeding.")
        n_ann = len(self.df_annotated) if getattr(self, "df_annotated", None) is not None else 0
        n_unk = len(self.df_unknowns) if getattr(self, "df_unknowns", None) is not None else 0
        n_bfn = len(self.df_before_norm) if getattr(self, "df_before_norm", None) is not None else 0
        n_grp = len(self.df_groups) if getattr(self, "df_groups", None) is not None else 0
        return (f"Loaded {n_ann} annotated compounds\n"
                f"Loaded {n_unk} unknown features\n"
                f"Loaded {n_bfn} rows from Before-Normalization file\n"
                f"Loaded {n_grp} sample group assignments")

    # ==========================================================
    # STATS DIR + READY CHECK
    # ==========================================================
    def _get_stats_dir(self) -> Path:
        d = Path(self.session_dir) if getattr(self, "session_dir", None) else (self.output_folder / "statistics")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _stats_ready(self) -> bool:
        stats_dir = self._get_stats_dir()
        known = [
            # Annotated (normalized)
            "Final_Annotated.csv",
            "Final_Annotated_Without_QCs.csv",
            "Final_Annotated_HighConf.csv",
            "Final_Annotated_Without_QCs_HighConf.csv",
            # Unknowns
            "Final_Unknowns.csv",
            "Final_Unknowns_Without_QCs.csv",
            # Before-Norm
            "Final_Annotated_BeforeNorm.csv",
            "Final_Annotated_BeforeNorm_Without_QCs.csv",
            # Groups
            "sample_groups_cleaned.csv",
        ]
        return any((stats_dir / k).exists() for k in known)

    # ==========================================================
    # GROUP DIALOG
    # ==========================================================
    def open_group_dialog(self):
        if self.df_groups is None or "Group" not in self.df_groups.columns:
            messagebox.showwarning("Missing groups", "sample_groups.csv not loaded.")
            return

        dlg = GroupSelectionDialog(self, self.df_groups, self.output_folder, self.session_dir)
        self.wait_window(dlg)

        if getattr(dlg, "success", False):
            self.session_dir = Path(dlg.session_dir)
            try:
                self._load_palette()
            except Exception:
                pass
            self.group_order = dlg.group_order

            self.prepare_statistical_datasets(
                allowed_groups=dlg.selected_groups,
                exclude_qc=dlg.exclude_qc,
                output_override=self.session_dir
            )

            label_groups = ", ".join(dlg.selected_groups) if dlg.selected_groups else "All"
            order_note = f"\nOrder: [{', '.join(self.group_order)}]" if self.group_order else ""
            self.selection_label.config(text=f"Selection: [{label_groups}]\n{self.session_dir}{order_note}")

    # ==========================================================
    # TOOLTIP
    # ==========================================================
    def _add_tooltip(self, widget, text):
        tooltip = tk.Label(self.main_frame, text=text, bg="lightyellow", fg="black", relief="solid", bd=1, wraplength=300)
        tooltip.place_forget()
        def on_enter(event):
            x = event.x_root - self.winfo_rootx() + 12
            y = event.y_root - self.winfo_rooty() + 18
            tooltip.place(x=x, y=y)
        def on_leave(event):
            tooltip.place_forget()
        widget.bind("<Enter>", on_enter); widget.bind("<Leave>", on_leave)

    # ==========================================================
    # PALETTE HELPERS
    # ==========================================================
    def _palette_path(self) -> Path:
        return self._get_stats_dir() / "group_colors.json"

    def _load_palette(self):
        p = self._palette_path()
        if p.exists():
            try:
                self.group_colors = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                self.group_colors = {}

    def _save_palette(self):
        try:
            self._palette_path().write_text(json.dumps(self.group_colors, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _active_groups(self) -> list[str]:
        stats_dir = self._get_stats_dir()
        gfile = stats_dir / "sample_groups_cleaned.csv"
        if gfile.exists():
            try:
                gdf = pd.read_csv(gfile)
                return sorted(gdf["Group"].astype(str).str.strip().unique().tolist())
            except Exception:
                pass
        if self.df_groups is not None and "Group" in self.df_groups.columns:
            return sorted(self.df_groups["Group"].astype(str).str.strip().unique().tolist())
        return []

    # ==========================================================
    # VOLCANO THRESHOLDS
    # ==========================================================
    def _get_volcano_thresholds(self):
        def _to_float(s, default):
            try: return float(str(s).strip())
            except Exception: return default
        fc  = _to_float(self.var_fc.get(), 1.5)
        fdr = _to_float(self.var_fdr.get(), 0.10)
        p   = _to_float(self.var_p.get(), 0.05)
        if fc < 1.0: fc = 1.0
        eps = 1e-12
        fdr = min(max(fdr, eps), 1.0)
        p   = min(max(p,   eps), 1.0)
        return fc, fdr, p

    # ==========================================================
    # AUTO-PREP
    # ==========================================================
    def _auto_prepare_or_warn(self):
        if not self._stats_ready():
            try:
                threading.Thread(
                    target=self.prepare_statistical_datasets,
                    kwargs={"allowed_groups": None, "exclude_qc": False, "output_override": self._get_stats_dir()},
                    daemon=True
                ).start()
            except Exception:
                self._toast("⚠ Datasets not prepared. Click 'Prepare Statistical Datasets' before running analyses.")
                print("⚠ Datasets not prepared. Click 'Prepare Statistical Datasets' before running analyses.", flush = True)

    # ==========================================================
    # BUTTON STATE
    # ==========================================================
    def _all_analysis_buttons(self):
        return (
            self.pca_button, self.plsda_button, self.heatmap_button,
            self.volcano_button, self.boxplots_button, self.violin_button,
            self.correlations_button, self.classdist_button, self.summint_button, self.classviolinbox_button,
            self.classcarbons_button
        )

    def _set_busy(self, busy: bool, label: str | None = None):
        state = "disabled" if busy else "normal"
        for btn in self._all_analysis_buttons(): btn.config(state=state)
        try: self.prepare_btn.config(state=state)
        except Exception: pass

        # Stop button: enabled only while busy (something is running)
        try:
            if busy:
                self.stop_button.state(["!disabled"])
            else:
                self.stop_button.state(["disabled"])
        except Exception:
            pass

        self.configure(cursor="watch" if busy else "")
        self.update_idletasks()
        if label: self._toast(label)


    def _acquire_runner(self, label: str) -> bool:
        with self._run_lock:
            if self._is_running:
                self._toast("Another analysis is already running")
                return False
            self._is_running = True
            # DO NOT touch self._worker_thread here; the launcher set it.
        self._set_busy(True, f"Running {label}…")
        return True


    def _release_runner(self, label: str):
        with self._run_lock:
            self._is_running = False
        self._set_busy(False, f"{label} completed ✓")



        # ==========================================================
    # HARD STOP: forcibly kill the worker thread immediately
    # ==========================================================
    @staticmethod
    def _async_raise(tid: int, exctype: type[BaseException]) -> None:
        """Raise an exception in the threads with id 'tid'."""
        if not inspect.isclass(exctype):
            raise TypeError("Only exception types can be raised")
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exctype))
        if res == 0:
            raise ValueError("Invalid thread id")
        elif res > 1:
            # if it returns >1, call again with NULL to revert, then fail hard
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
            raise SystemError("PyThreadState_SetAsyncExc affected multiple threads")

    def hard_stop_now(self):
        """Immediately stop the currently running analysis thread."""
        th = getattr(self, "_worker_thread", None)
        if th is None or not th.is_alive():
            # Fallback: try to find a live worker by name (in case handle was lost)
            for t in threading.enumerate():
                if t is not threading.current_thread() and getattr(t, "name", "").startswith("LQ-StatsWorker"):
                    th = t
                    break
        if th is None or not th.is_alive():
            self._toast("Nothing is running")
            return

        self._toast("⛔ Stopping NOW…")
        try:
            self._async_raise(th.ident, SystemExit)
        except Exception as e:
            print("[STOP] async raise failed:", e, flush=True)
        for _ in range(10):
            if not th.is_alive():
                break
            time.sleep(0.05)
        if th.is_alive():
            try:
                self._async_raise(th.ident, KeyboardInterrupt)
            except Exception as e:
                print("[STOP] async raise (KeyboardInterrupt) failed:", e, flush=True)
        for _ in range(10):
            if not th.is_alive():
                break
            time.sleep(0.05)

        with self._run_lock:
            self._is_running = False
        self._set_busy(False, "Stopped ✗")
        if th.is_alive():
            self._toast("⚠ Could not kill cleanly. Use Quit to force-close.")
            print("[STOP] Thread resisted termination.", flush=True)



    # ==========================================================
    # ANALYSIS LAUNCHERS
    # ==========================================================
    def run_pca(self):
        if self._is_running: 
            self._toast("Another analysis is already running"); 
            return
        self._worker_thread = threading.Thread(target=self._run_analysis, args=("PCA",), daemon=True, name="LQ-StatsWorker-PCA")
        self._worker_thread.start()

    def run_plsda(self):
        if self._is_running: 
            self._toast("Another analysis is already running")
            return
        self._worker_thread = threading.Thread(target=self._run_analysis, args=("PLS-DA",), daemon=True, name="LQ-StatsWorker-PLSDA")
        self._worker_thread.start()

    def run_heatmap(self):
        if self._is_running:
            self._toast("Another analysis is already running")
            return
        self._worker_thread = threading.Thread(target=self._run_analysis, args=("Heatmap",), daemon=True, name="LQ-StatsWorker-Heatmap")
        self._worker_thread.start()

    def run_volcano(self):
        if self._is_running:
            self._toast("Another analysis is already running")
            return
        self._worker_thread = threading.Thread(target=self._run_analysis, args=("Volcano",), daemon=True, name="LQ-StatsWorker-Volcano")
        self._worker_thread.start()

    def run_boxplots(self):
        if self._is_running: 
            self._toast("Another analysis is already running")
            return
        self._worker_thread = threading.Thread(target=self._run_analysis, args=("Boxplots",), daemon=True, name="LQ-StatsWorker-Boxplots")
        self._worker_thread.start()

    def run_violin(self):
        if self._is_running: 
            self._toast("Another analysis is already running")
            return
        self._worker_thread = threading.Thread(target=self._run_analysis, args=("Violin",), daemon=True, name="LQ-StatsWorker-Violin")
        self._worker_thread.start()

    def run_correlation_analysis(self):
        if self._is_running:
            self._toast("Another analysis is already running")
            return
        self._worker_thread = threading.Thread(target=self._run_analysis, args=("Correlations",), daemon=True, name="LQ-StatsWorker-Corr")
        self._worker_thread.start()
        
    def run_class_distributions(self):
        if self._is_running:
            self._toast("Another analysis is already running")
            return
        self._worker_thread = threading.Thread(target=self._run_analysis, args=("Class_Distributions",), daemon=True, name="LQ-StatsWorker-ClassDist")
        self._worker_thread.start()
        
    def run_class_sums(self):
        if self._is_running:
            self._toast("Another analysis is already running")
            return
        self._worker_thread = threading.Thread(target=self._run_analysis, args=("Class_Sums",), daemon=True, name="LQ-StatsWorker-ClassSums")
        self._worker_thread.start()
        
    def run_class_violin_box(self):
        if self._is_running:
            self._toast("Another analysis is already running")
            return
        self._worker_thread = threading.Thread(target=self._run_analysis, args=("Class_violin_box",), daemon=True, name="LQ-StatsWorker-ClassVB")
        self._worker_thread.start()

    def run_class_carbons_db(self):
        if self._is_running:
            self._toast("Another analysis is already running")
            return
        self._worker_thread = threading.Thread(
            target=self._run_analysis,
            args=("Class_Carbons_DB",),
            daemon=True,
            name="LQ-StatsWorker-ClassCarbonsDB",)
        self._worker_thread.start()
        
    def run_all(self):
        if self._is_running:
            self._toast("Another analysis is already running")
            return
        self._worker_thread = threading.Thread(
            target=self._run_all_worker, daemon=True, name="LQ-StatsWorker-RunAll"
        )
        self._worker_thread.start()


    def _run_all_worker(self):
        if not self._acquire_runner("All analyses"):
            return
        stopped = False
        try:
            order = ["PCA", "PLS-DA", "Heatmap", "Correlations", "Class_Distributions", "Class_Sums", "Class_violin_box", "Class_Carbons_DB", "Volcano", "Boxplots", "Violin",]
            for at in order:
                self._toast(f"Running {at}…")
                self._run_analysis(at, _sequence_mode=True)
            self._toast("Run All completed ✓")

        except Exception:
            print(traceback.format_exc(), flush=True)
            self._toast("Run All failed")
        finally:
            self._release_runner("All analyses" if not stopped else "All analyses (stopped)")


    # ==========================================================
    # CORE RUNNER
    # ==========================================================
    def _datasets_for_selection(self) -> list[tuple[str, str]]:
        ds = (self.var_dataset.get() or "").strip()

        if ds == "Annotated (normalized and merged)":
            return [
                    ("Final_Annotated.csv", "With_QCs"),
                    ("Final_Annotated_Without_QCs.csv", "Without_QCs"),
                    ("Final_Annotated_HighConf.csv", "HighConf_With_QCs"),
                    ("Final_Annotated_Without_QCs_HighConf.csv", "HighConf_Without_QCs"),
                ]

        elif ds == "Annotated (pre-normalization, merged)":
            return [
                    ("Final_Annotated_BeforeNorm.csv", "BeforeNorm_With_QCs"),
                    ("Final_Annotated_BeforeNorm_Without_QCs.csv", "BeforeNorm_Without_QCs"),
                ]

        elif ds == "Unknowns (normalized and merged)":
            return [
                    ("Final_Unknowns.csv", "Unknowns_With_QCs"),
                    ("Final_Unknowns_Without_QCs.csv", "Unknowns_Without_QCs"),
                ]

        elif ds == "Annotated (POS only)":
            return [
                    ("POS_Final_Annotated.csv", "POS_Only"),
                    ("POS_Final_Annotated_Without_QCs.csv", "POS_Only_Without_QCs"),
                ]

        elif ds == "Unknowns (POS only)":
            return [
                    ("POS_Final_Unknowns.csv", "POS_Only"),
                    ("POS_Final_Unknowns_Without_QCs.csv", "POS_Only_Without_QCs"),
                ]

        elif ds == "Annotated (NEG only)":
            return [
                    ("NEG_Final_Annotated.csv", "NEG_Only"),
                    ("NEG_Final_Annotated_Without_QCs.csv", "NEG_Only_Without_QCs"),
                ]

        elif ds == "Unknowns (NEG only)":
            return [
                    ("NEG_Final_Unknowns.csv", "NEG_Only"),
                    ("NEG_Final_Unknowns_Without_QCs.csv", "NEG_Only_Without_QCs"),
                ]

        else:
            raise FileNotFoundError(f"Unsupported dataset selection: {ds}")



    def _run_analysis(self, analysis_type, _sequence_mode: bool = False):
        outer_manages = _sequence_mode
        if not outer_manages:
            if not self._acquire_runner(analysis_type): return

        try:
            stats_dir = self._get_stats_dir()
            cleaned_group_file = stats_dir / "sample_groups_cleaned.csv"
            group_file = cleaned_group_file if cleaned_group_file.exists() else (self.output_folder / "sample_groups.csv")

            if not self._stats_ready():
                self._toast("⚠ Prepare datasets first (use the button above)")
                return
            if not group_file.exists():
                group_file = None

            if not outer_manages:
                self._toast(f"Running {analysis_type}…")

            plt.close("all")
            
            for fname, label in self._datasets_for_selection():

                fpath = stats_dir / fname
                if not fpath.exists():
                    continue

                # Tool-specific gating
                needs_no_qc = {"PLS-DA", "Volcano", "Heatmap", "Boxplots", "Violin", "Correlations", "Class_Distributions", "Class_Sums", "Class_violin_box", "Class_Carbons_DB"}
                if analysis_type in needs_no_qc and ("Without_QCs" not in label):
                    continue
                if analysis_type in {"Boxplots", "Violin"} and ("HighConf" in label):
                    continue

                subfolder = stats_dir / analysis_type / label
                subfolder.mkdir(parents=True, exist_ok=True)

                # Optional palette
                try:
                    self._load_palette()
                    palette = self.group_colors or None
                except Exception:
                    palette = None

                try:

                    if analysis_type == "PCA":
                        run_pca(fpath, group_file, subfolder, group_colors=palette, group_order=self.group_order)
                    elif analysis_type == "PLS-DA":
                        run_plsda(fpath, group_file, subfolder, group_colors=palette, group_order=self.group_order)
                    elif analysis_type == "Heatmap":
                        run_heatmap(fpath, group_file, subfolder, group_colors=palette, group_order=self.group_order)
                    elif analysis_type == "Volcano":
                        fc, fdr, p = self._get_volcano_thresholds()
                        run_volcano(
                            fpath, group_file, subfolder,
                            sample_type=self.sample_type,
                            p_value_threshold=p, fdr_threshold=fdr, fold_change_threshold=fc,
                            group_colors=palette, group_order=self.group_order,
                        )
                    elif analysis_type == "Boxplots":
                        run_boxplots(fpath, group_file, subfolder, group_order=self.group_order, group_colors=palette)
                    elif analysis_type == "Violin":
                        run_violinplots(fpath, group_file, subfolder, group_order=self.group_order, group_colors=palette)
                    elif analysis_type == "Correlations":
                        run_correlation_analysis(fpath, group_file, subfolder, group_order=self.group_order)
                    elif analysis_type == "Class_Distributions":
                        run_class_distributions(fpath, group_file, subfolder, group_colors=palette, group_order=self.group_order, sample_type=self.sample_type, unknown_policy="append")
                    elif analysis_type == "Class_Sums":
                        run_class_sums(fpath, group_file, subfolder, group_colors=palette, group_order=self.group_order, sample_type=self.sample_type)
                    elif analysis_type == "Class_violin_box":
                        run_class_violin_box(fpath, group_file, subfolder, group_colors=palette, group_order=self.group_order)
                    elif analysis_type == "Class_Carbons_DB":
                        run_class_carbons_db(fpath, group_file, subfolder, group_colors=palette, group_order=self.group_order, exclude_qc=True,)

                except Exception:
                    print(traceback.format_exc(), flush=True)
                finally:
                    plt.close('all')

            if not outer_manages:
                self._toast(f"{analysis_type} completed ✓")

        except Exception:
            print(traceback.format_exc(), flush=True)
            if not outer_manages:
                self._toast(f"{analysis_type} failed")
        finally:
            if not outer_manages:
                self._release_runner(analysis_type)

    # ==========================================================
    # NAVIGATION / CLOSE
    # ==========================================================
    def return_to_processing(self):
        self.destroy(); self.parent.deiconify()

    def _on_close(self):
        try:
            self.destroy(); self.parent.destroy()
        except Exception:
            os._exit(0)

    # ==========================================================
    # SAMPLE NAME CLEANUP
    # ==========================================================
    def _clean_sample_name(self, name: str) -> str:
        if not isinstance(name, str): return name
        cleaned = name
        cleaned = re.sub(r"\[?POS\]?|\[?NEG\]?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^(P_|N_)", "", cleaned)
        cleaned = re.split(r"_P[12]", cleaned)[0]
        return cleaned.strip("_- ")

    # ==========================================================
    # PREPARE STATISTICAL DATASETS
    # ==========================================================
    def prepare_statistical_datasets(self, allowed_groups=None, exclude_qc=False, output_override: Path = None):
        """Generate CSVs (+ transposed) under /statistics for:
           - Annotated (normalized): With_QCs / Without_QCs / HighConf_* variants
           - Unknowns (normalized): With_QCs / Without_QCs (if Unknowns file exists)
           - Annotated (pre-normalization): BeforeNorm_With_QCs / BeforeNorm_Without_QCs (if file exists)
        """
        try:
            ann_path = self.output_folder / "Final_Annotated.csv"
            grp_path = self.output_folder / "sample_groups.csv"

            if not ann_path.exists():
                messagebox.showwarning("Missing File", "Final_Annotated.csv not found in output folder."); return
            if not grp_path.exists():
                messagebox.showwarning("Missing File", "sample_groups.csv not found in output folder."); return

            stats_dir = Path(output_override) if output_override else (self.output_folder / "statistics")
            stats_dir.mkdir(parents=True, exist_ok=True)

            df_groups_raw = pd.read_csv(grp_path)

            # --- Clean raw groups (unfiltered) and keep a QC list from the full file ---
            df_raw_clean = df_groups_raw.copy()
            df_raw_clean["Group"] = df_raw_clean["Group"].astype(str).str.strip()
            df_raw_clean["Sample"] = df_raw_clean["Sample"].apply(self._clean_sample_name)
            df_raw_clean = df_raw_clean.drop_duplicates(subset=["Sample"], keep="first").reset_index(drop=True)

            # QC list from the FULL groups file (never filtered)
            qc_samples_full = df_raw_clean.loc[df_raw_clean["Group"].str.upper() == "QC", "Sample"].tolist()
            qc_set_full = set(qc_samples_full)

            # --- Now apply user selection/filtering to produce the session groups file ---
            df_groups = df_raw_clean.copy()
            if allowed_groups is not None:
                df_groups = df_groups[df_groups["Group"].isin(list(allowed_groups))].copy()
            if exclude_qc:
                df_groups = df_groups[df_groups["Group"].str.upper() != "QC"].copy()

            # Save the cleaned, session-specific groups
            try:
                df_groups.to_csv(stats_dir / "sample_groups_cleaned.csv", index=False, encoding="utf-8-sig")
            except Exception:
                print("[Warning] Could not create cleaned sample_groups file:", flush=True)
                print(traceback.format_exc(), flush=True)

            # Helpers
            def _detect_meta(df):
                meta_keep = [
                    "UniqueID","RT (min)","m/z","Polarity","Annotation","Annotation Type",
                    "Annotation Source","Headgroup","Lipid Class","Δm/z (mDa)","Δm/z (ppm)",
                    "MS/MS score","Annotation tier","mSigma","Molecular Formula","Plasmenyl?",
                    "Number of carbons in fatty acyls","Double bond equivalents","Chain type",
                    "PUFA?","Modifications","# of modifications","Oxidized?"
                ]
                return [c for c in meta_keep if c in df.columns]

            def _detect_samples(df, meta_cols):
                cols = [
                    c for c in df.columns
                    if c not in meta_cols
                    and "rsd" not in c.lower()
                    and pd.api.types.is_numeric_dtype(df[c])
                ]
                if not cols: return []
                newnames = {c: self._clean_sample_name(c) for c in cols}
                df.rename(columns=newnames, inplace=True)
                return list(newnames.values())

            def _save_with_T(df, meta_cols, samples, base_name):
                path = stats_dir / base_name
                df_to_save = df[meta_cols + samples]
                df_to_save.to_csv(path, index=False, encoding="utf-8-sig")
                transposed = df_to_save.set_index("UniqueID").transpose()
                transposed.index.name = "UniqueID"
                (stats_dir / f"{path.stem}_T.csv").write_text(transposed.to_csv(index=True), encoding="utf-8")

            # ---------- Annotated (normalized) ----------
            df_ann = pd.read_csv(ann_path, low_memory=False)
            meta_ann = _detect_meta(df_ann)
            samples_ann = _detect_samples(df_ann, meta_ann)

            # --- POS-only and NEG-only filtering ---
            # Extract sample columns again (after renaming)
            samples_all = samples_ann

            samples_pos = [s for s in samples_all if s.startswith("P_")]
            samples_neg = [s for s in samples_all if s.startswith("N_")]

            # POS-only datasets
            if samples_pos:
                _save_with_T(df_ann, meta_ann, samples_pos, "POS_Final_Annotated.csv")
                samples_pos_noqc = [s for s in samples_pos if s not in qc_set_full]
                if samples_pos_noqc:
                    _save_with_T(df_ann, meta_ann, samples_pos_noqc, "POS_Final_Annotated_Without_QCs.csv")

            # NEG-only datasets
            if samples_neg:
                _save_with_T(df_ann, meta_ann, samples_neg, "NEG_Final_Annotated.csv")
                samples_neg_noqc = [s for s in samples_neg if s not in qc_set_full]
                if samples_neg_noqc:
                    _save_with_T(df_ann, meta_ann, samples_neg_noqc, "NEG_Final_Annotated_Without_QCs.csv")


            if not samples_ann:
                messagebox.showwarning("No Sample Columns", "No numeric sample intensity columns in Final_Annotated.csv"); return

            # Build selected non-QC sample set and ensure QC columns are present for With_QCs
            selected_set = set(df_groups["Sample"].dropna().astype(str)) if (allowed_groups is not None or exclude_qc) else set(samples_ann)

            # With_QCs = selected non-QC samples PLUS all QC columns from the full groups file (if present in table)
            samples_ann_with_qc = [c for c in samples_ann if (c in selected_set) or (c in qc_set_full)]
            if not samples_ann_with_qc:
                messagebox.showwarning("No Samples", "No valid samples found for Annotated (With_QCs).")
            else:
                _save_with_T(df_ann, meta_ann, samples_ann_with_qc, "Final_Annotated.csv")

            # Without_QCs = selected non-QC samples (strictly exclude all QC)
            samples_ann_without_qc = [c for c in samples_ann if (c in selected_set) and (c not in qc_set_full)]
            if samples_ann_without_qc:
                _save_with_T(df_ann, meta_ann, samples_ann_without_qc, "Final_Annotated_Without_QCs.csv")
            else:
                messagebox.showwarning("No Non-QC Samples", "No non-QC samples selected for Annotated (Without_QCs).")

            # High confidence (if present)
            tier_col = next((c for c in df_ann.columns if c.strip().lower() == "annotation tier"), None)
            if tier_col:
                high_mask = df_ann[tier_col].fillna("").str.lower() == "high confidence"
                df_high = df_ann.loc[high_mask].copy()
            else:
                df_high = df_ann.copy()

            if samples_ann_with_qc:
                _save_with_T(df_high, meta_ann, samples_ann_with_qc, "Final_Annotated_HighConf.csv")
            if samples_ann_without_qc:
                _save_with_T(df_high, meta_ann, samples_ann_without_qc, "Final_Annotated_Without_QCs_HighConf.csv")


            # ---------- Unknowns (optional) ----------
            unk_src = None
            for p in [
                # merged polarity
                self.output_folder / "Final_Unknowns.csv",

                # polarity-specific
                self.output_folder / "POS" / "Pos_Final_Unknowns.csv",
                self.output_folder / "NEG" / "Neg_Final_Unknowns.csv",
            ]:
                if p.exists():
                    unk_src = p
                    break


            if unk_src is not None:
                df_unk = pd.read_csv(unk_src, low_memory=False)
                meta_unk = [c for c in ("UniqueID","RT (min)","m/z","Polarity","RSD QCs (%)","RSD Samples (%)") if c in df_unk.columns]
                samples_unk = _detect_samples(df_unk, meta_unk)
                
                # --- POS-only and NEG-only unknowns ---
                samples_unk_pos = [s for s in samples_unk if s.startswith("P_")]
                samples_unk_neg = [s for s in samples_unk if s.startswith("N_")]

                # POS-only unknowns
                if samples_unk_pos:
                    _save_with_T(df_unk, meta_unk, samples_unk_pos, "POS_Final_Unknowns.csv")
                    samples_unk_pos_noqc = [s for s in samples_unk_pos if s not in qc_set_full]
                    if samples_unk_pos_noqc:
                        _save_with_T(df_unk, meta_unk, samples_unk_pos_noqc, "POS_Final_Unknowns_Without_QCs.csv")

                # NEG-only unknowns
                if samples_unk_neg:
                    _save_with_T(df_unk, meta_unk, samples_unk_neg, "NEG_Final_Unknowns.csv")
                    samples_unk_neg_noqc = [s for s in samples_unk_neg if s not in qc_set_full]
                    if samples_unk_neg_noqc:
                        _save_with_T(df_unk, meta_unk, samples_unk_neg_noqc, "NEG_Final_Unknowns_Without_QCs.csv")

                sel_unk = set(df_groups["Sample"].dropna().astype(str)) if (allowed_groups is not None or exclude_qc) else set(samples_unk)
                samples_unk_with_qc = [c for c in samples_unk if (c in sel_unk) or (c in qc_set_full)]
                samples_unk_without_qc = [c for c in samples_unk if (c in sel_unk) and (c not in qc_set_full)]
                if samples_unk_with_qc:
                    _save_with_T(df_unk, meta_unk, samples_unk_with_qc, "Final_Unknowns.csv")
                if samples_unk_without_qc:
                    _save_with_T(df_unk, meta_unk, samples_unk_without_qc, "Final_Unknowns_Without_QCs.csv")


            # ---------- Before-Normalization (optional) ----------
            bfn_src = None
            for p in (
                self.output_folder / "Final_Annotated_Before_Normalization.csv",
                self.output_folder / "debug" / "4-Final_annotated_results_imputed_filtered.csv",
                self.output_folder / "debug" / "3-Final_annotated_results_imputed.csv",
            ):
                if p.exists(): bfn_src = p; break

            if bfn_src is not None:
                df_bfn = pd.read_csv(bfn_src, low_memory=False)
                meta_bfn = _detect_meta(df_bfn)
                samples_bfn = _detect_samples(df_bfn, meta_bfn)
                sel_bfn = set(df_groups["Sample"].dropna().astype(str)) if (allowed_groups is not None or exclude_qc) else set(samples_bfn)
                samples_bfn_with_qc = [c for c in samples_bfn if (c in sel_bfn) or (c in qc_set_full)]
                samples_bfn_without_qc = [c for c in samples_bfn if (c in sel_bfn) and (c not in qc_set_full)]
                if samples_bfn_with_qc:
                    _save_with_T(df_bfn, meta_bfn, samples_bfn_with_qc, "Final_Annotated_BeforeNorm.csv")
                if samples_bfn_without_qc:
                    _save_with_T(df_bfn, meta_bfn, samples_bfn_without_qc, "Final_Annotated_BeforeNorm_Without_QCs.csv")


            # Log + UI feedback
            with open(stats_dir / "statistics_log.txt", "a", encoding="utf-8") as log:
                log.write(f"\n[{pd.Timestamp.now()}] Generated statistical datasets (Annotated + optional Unknowns/BeforeNorm)\n")

            messagebox.showinfo(
                "Statistics Files Created",
                f"Statistical datasets created successfully in:\n\n{stats_dir}\n\n"
                "• Annotated (+ variants)\n"
                "• Unknowns (+ Without_QCs) if available\n"
                "• BeforeNorm (+ Without_QCs) if available\n"
            )
            self.summary_label.config(text=f"✅ Statistical datasets (and transposed versions) saved to {stats_dir}")

            # Enable buttons
            for btn in (self.pca_button, self.plsda_button, self.heatmap_button, self.volcano_button, self.boxplots_button, self.violin_button,
                        self.correlations_button, self.classdist_button, self.summint_button, self.classviolinbox_button, self.classcarbons_button):
                btn.config(state="normal")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to prepare datasets:\n{e}")

    # ==========================================================
    # QUIT + TOAST
    # ==========================================================
    def quit_app(self):
        try:
            self.destroy(); self.parent.destroy()
        except Exception:
            pass
        finally:
            os._exit(0)

    def _toast(self, text):
        if hasattr(self, "_current_toast") and self._current_toast.winfo_exists():
            self._current_toast.destroy()
        toast = tk.Toplevel(self); toast.overrideredirect(True); toast.configure(bg="#333333")
        self._current_toast = toast
        label = tk.Label(toast, text=text, fg="white", bg="#333333", font=("Segoe UI", 10, "bold"), padx=15, pady=8); label.pack()
        self.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() // 2) - 75
        y = self.winfo_y() + (self.winfo_height() // 2) + 380
        toast.geometry(f"+{x}+{y}")
        if any(kw in text.lower() for kw in ["done", "completed", "success", "first"]):
            def close_on_click(_event=None):
                if toast.winfo_exists(): toast.destroy()
            toast.bind("<Button-1>", close_on_click)


class GroupSelectionDialog(tk.Toplevel):
    def __init__(self, parent: "StatisticsPage", df_groups: pd.DataFrame, base_output: Path, current_session: Optional[Path]):
        super().__init__(parent)
        self.title("Select groups & output")
        self.configure(bg="white")
        self.transient(parent)
        self.grab_set()

        self.parent = parent
        self.df = df_groups.copy()
        self.df["Group"] = self.df["Group"].astype(str).str.strip()
        self.session_dir = current_session
        self.success = False
        self.exclude_qc = tk.BooleanVar(value=False)

        frame = tk.Frame(self, bg="white"); frame.pack(fill="both", expand=True, padx=16, pady=12)

        ttk.Label(frame, text="Available groups (click on each group to select)", style="Section.TLabel").grid(row=0, column=0, sticky="w")

        self.exclude_qc_cb = ttk.Checkbutton(frame, text="Exclude QC groups from selection", variable=self.exclude_qc, command=self._refresh_groups)
        self.exclude_qc_cb.grid(row=0, column=1, sticky="e", padx=(8, 0))

        self.listbox = tk.Listbox(frame, selectmode="multiple", exportselection=False, height=15, width=35)
        self.listbox.grid(row=1, column=0, sticky="nsw")

        btns = tk.Frame(frame, bg="white"); btns.grid(row=1, column=1, sticky="n", padx=(10, 0))
        ttk.Button(btns, text="Select All", width=16, command=lambda: self.listbox.select_set(0, tk.END)).pack(pady=2)
        ttk.Button(btns, text="Clear", width=16, command=lambda: self.listbox.selection_clear(0, tk.END)).pack(pady=2)
        ttk.Button(btns, text="Move Up", width=16, command=self._move_up).pack(pady=(8, 2))
        ttk.Button(btns, text="Move Down", width=16, command=self._move_down).pack(pady=2)

        self.counts_label = ttk.Label(frame, text="", style="Subtle.TLabel", justify="left")
        self.counts_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 8))

        ttk.Separator(frame, orient="horizontal").grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 8))
        ttk.Label(frame, text="Results output folder", style="Section.TLabel").grid(row=4, column=0, sticky="w", pady=(0, 4))

        out_row = tk.Frame(frame, bg="white"); out_row.grid(row=5, column=0, columnspan=2, sticky="ew"); out_row.grid_columnconfigure(0, weight=1)
        self.output_entry = ttk.Entry(out_row, width=120); self.output_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(out_row, text="Choose…", width=12, command=self._choose_dir).grid(row=0, column=1, padx=(0, 4))
        ttk.Button(out_row, text="Auto-name", width=12, command=self._auto_name).grid(row=0, column=2)

        actions = tk.Frame(frame, bg="white"); actions.grid(row=7, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(actions, text="Cancel", width=12, command=self._cancel).pack(side="right", padx=6)
        ttk.Button(actions, text="Confirm selection", width=18, command=self._prepare).pack(side="right", padx=6)

        self._refresh_groups(); self._auto_name()

        self.resizable(False, False)
        self.update_idletasks()
        self.geometry(f"+{parent.winfo_rootx()+80}+{parent.winfo_rooty()+80}")

    # ---- helpers ----
    def _choose_dir(self):
        base = self.parent.output_folder / "statistics"; base.mkdir(parents=True, exist_ok=True)
        chosen = filedialog.askdirectory(parent=self, initialdir=str(base), title="Select output folder")
        if chosen:
            self.session_dir = Path(chosen)
            self.output_entry.delete(0, "end"); self.output_entry.insert(0, str(self.session_dir))

    def _auto_name(self):
        base = self.parent.output_folder / "statistics" / "GS"; base.mkdir(parents=True, exist_ok=True)
        groups = self._get_selected_groups() or self._all_group_names(exclude_qc=self.exclude_qc.get())
        label = "+".join(groups); slug = hashlib.sha1(label.encode("utf-8")).hexdigest()[:6]
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_dir = base / f"GS_{slug}__{ts}"
        self.output_entry.delete(0, "end"); self.output_entry.insert(0, str(self.session_dir))
        try:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            (self.session_dir / "groups_selected.txt").write_text("Selected groups (in order):\n" + "\n".join(groups), encoding="utf-8")
        except Exception:
            pass

    def _all_group_names(self, exclude_qc=False):
        s = self.df["Group"]
        if exclude_qc: s = s[s.str.upper() != "QC"]
        return sorted(s.unique())

    def _refresh_groups(self):
        names = self._all_group_names(exclude_qc=self.exclude_qc.get())
        self.listbox.delete(0, tk.END)
        for g in names: self.listbox.insert(tk.END, g)

        counts = self.df.copy()
        if self.exclude_qc.get(): counts = counts[counts["Group"].str.upper() != "QC"]
        counts = counts.groupby("Group")["Sample"].nunique().sort_values(ascending=False)
        lines = [f"• {g}: {int(n)} samples" for g, n in counts.items()]
        max_per_col = 5; nrows = max_per_col
        while len(lines) % nrows: lines.append("")
        col_width = max((len(s) for s in lines if s), default=0) + 4
        cols = [lines[i:i+nrows] for i in range(0, len(lines), nrows)]
        rows = ["  ".join((c[i].ljust(col_width) for c in cols if i < len(c))) for i in range(nrows)]
        self.counts_label.config(text="Current group counts:\n" + "\n".join(rows))

    def _move_up(self):
        sel = list(self.listbox.curselection())
        if not sel: return
        for i in sel:
            if i == 0: continue
            text = self.listbox.get(i)
            self.listbox.delete(i); self.listbox.insert(i-1, text)
        self.listbox.selection_clear(0, tk.END)
        for i in [max(0, s-1) for s in sel]: self.listbox.selection_set(i)

    def _move_down(self):
        sel = list(self.listbox.curselection())
        if not sel: return
        for i in reversed(sel):
            if i == self.listbox.size()-1: continue
            text = self.listbox.get(i)
            self.listbox.delete(i); self.listbox.insert(i+1, text)
        self.listbox.selection_clear(0, tk.END)
        for i in [min(self.listbox.size()-1, s+1) for s in sel]: self.listbox.selection_set(i)

    def _get_selected_groups(self):
        return [self.listbox.get(i) for i in self.listbox.curselection()]

    def _prepare(self):
        sel = self._get_selected_groups()
        if not sel:
            messagebox.showwarning("No groups selected", "Please select at least one group."); return
        if not self.session_dir: self._auto_name()
        Path(self.session_dir).mkdir(parents=True, exist_ok=True)

        df = self.df.copy()
        if self.exclude_qc.get():
            df = df[df["Group"].str.upper() != "QC"]
        df = df[df["Group"].isin(sel)].copy()
        df["Sample"] = df["Sample"].apply(self.parent._clean_sample_name)
        df = df.drop_duplicates(subset=["Sample"], keep="first").reset_index(drop=True)
        df.to_csv(Path(self.session_dir) / "sample_groups_cleaned.csv", index=False, encoding="utf-8-sig")

        full_in_listbox_order = [self.listbox.get(i) for i in range(self.listbox.size())]
        ordered_sel = [g for g in full_in_listbox_order if g in sel]

        self.selected_groups = sel
        self.group_order = ordered_sel
        self.exclude_qc = self.exclude_qc.get()
        self.success = True
        self.destroy()

    def _cancel(self):
        self.destroy()
