# GUI/view_statistics.py
from __future__ import annotations
import tkinter as tk
import numpy as np
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
from Stats.heatmap_analysis import (
    get_available_annotations as get_available_heatmap_annotations,
    run_heatmap,
    run_selected_lipid_heatmap,
)
from Stats.volcano_analysis import run_volcano
from Stats.boxplots import run_boxplots
from Stats.violinplots import run_violinplots
from Stats.correlation_analysis import run_correlation_analysis
from Stats.class_distributions import run_from_stats as run_class_distributions
from Stats.summed_intensity_per_class import run_from_stats as run_class_sums
from Stats.class_violin_boxplots import run_from_stats as run_class_violin_box
from Stats.class_number_carbons_DB import run_from_stats as run_class_carbons_db
from Stats.enrichment_analysis import run_from_stats as run_enrichment_analysis
from Stats.ratio_analysis import (
    DEFAULT_CLASS_RATIO_DEFS,
    DEFAULT_PRODUCT_RATIO_DEFS,
    get_available_annotation_labels,
    run_from_stats as run_ratio_analysis,
)
from Stats.upset_plot import run_from_stats as run_upset_plot
from Stats.advanced_differential_analysis import run_from_stats as run_advanced_differential_analysis

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
        self._worker_thread = None  # handle to kill immediately

        # Volcano threshold variables (defaults)
        self.var_fc  = tk.StringVar(value="1.5")
        self.var_fdr = tk.StringVar(value="0.05")
        self.var_p   = tk.StringVar(value="0.05")
        self.var_dpi = tk.StringVar(value="100")
        self.var_publication_theme = tk.BooleanVar(value=False)
        self.var_volcano_labels = tk.BooleanVar(value=False)
        self.var_ratio_settings_summary = tk.StringVar(value="Default ratio settings")
        self.var_selected_heatmap_summary = tk.StringVar(value="Selected heatmap: no lipids selected")

        # Dataset selector state
        self.var_dataset = tk.StringVar(value="Annotated (normalized and merged)")
        self.ratio_settings = self._default_ratio_settings()
        self._load_ratio_settings()
        self.selected_heatmap_annotations: list[str] = []
        self._load_selected_heatmap_settings()

        # Try to reuse styles
        self._configure_local_style_if_needed()

        # --- Load data (for summary only) ---
        self.df_annotated = None
        self.df_annotated_semi = None
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

        # === Analysis configuration ===
        config = tk.Frame(self.main_frame, bg="white")
        config.pack(fill="x", padx=24, pady=(6, 8))
        config.grid_columnconfigure(0, weight=1)
        config.grid_columnconfigure(1, weight=1)

        volcano_panel = ttk.LabelFrame(config, text="Volcano Analysis")
        volcano_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        volcano_panel.grid_columnconfigure(1, weight=1)
        volcano_panel.grid_columnconfigure(3, weight=1)

        ttk.Label(volcano_panel, text="Fold-change (FC ≥)", style="Body.TLabel").grid(row=0, column=0, sticky="e", padx=(8, 6), pady=(8, 4))
        fc_entry = ttk.Entry(volcano_panel, textvariable=self.var_fc, width=10)
        fc_entry.grid(row=0, column=1, sticky="w", pady=(8, 4))

        ttk.Label(volcano_panel, text="FDR p <", style="Body.TLabel").grid(row=0, column=2, sticky="e", padx=(12, 6), pady=(8, 4))
        fdr_entry = ttk.Entry(volcano_panel, textvariable=self.var_fdr, width=10)
        fdr_entry.grid(row=0, column=3, sticky="w", pady=(8, 4))

        ttk.Label(volcano_panel, text="raw p <", style="Body.TLabel").grid(row=1, column=0, sticky="e", padx=(8, 6), pady=(4, 4))
        p_entry = ttk.Entry(volcano_panel, textvariable=self.var_p, width=10)
        p_entry.grid(row=1, column=1, sticky="w", pady=(4, 4))

        ttk.Checkbutton(volcano_panel, text="Volcano labels", variable=self.var_volcano_labels).grid(
            row=1, column=2, columnspan=2, sticky="w", padx=(12, 0), pady=(4, 4)
        )

        self.volcano_button = ttk.Button(volcano_panel, text="Run Volcano Plot", width=22, command=self.run_volcano)
        self.volcano_button.grid(row=2, column=0, columnspan=4, sticky="w", padx=8, pady=(8, 10))

        figure_panel = ttk.LabelFrame(config, text="Figure Style")
        figure_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        ttk.Label(figure_panel, text="Figure DPI", style="Body.TLabel").grid(row=0, column=0, sticky="e", padx=(8, 6), pady=(8, 4))
        dpi_combo = ttk.Combobox(figure_panel, textvariable=self.var_dpi, state="readonly", width=8, values=["100", "150", "200", "300", "600"])
        dpi_combo.grid(row=0, column=1, sticky="w", pady=(8, 4))
        ttk.Checkbutton(figure_panel, text="Publication theme", variable=self.var_publication_theme).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=8, pady=(4, 10)
        )

        def _sanitize_thresholds(_evt=None):
            fc, fdr, p = self._get_volcano_thresholds()
            self.var_fc.set(f"{fc:.3g}"); self.var_fdr.set(f"{fdr:.3g}"); self.var_p.set(f"{p:.3g}")
        for e in (fc_entry, fdr_entry, p_entry):
            e.bind("<FocusOut>", _sanitize_thresholds)

        # === Dataset selector ===
        ds = tk.Frame(self.main_frame, bg="white")
        ds.pack(fill="x", padx=24, pady=(8, 6))
        ttk.Label(ds, text="\nDataset selection", style="Section.TLabel").grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))
        ttk.Label(ds, text="Dataset:", style="Body.TLabel").grid(row=1, column=0, sticky="e", padx=(0, 6))
        ds_combo = ttk.Combobox(
            ds, textvariable=self.var_dataset, state="readonly", width=60,
            values=[
                "Annotated (normalized and merged)",
                "Annotated (with missing values)",
                "Annotated semi-quant (normalized and merged)",
                "Annotated semi-quant (with missing values)",
                "Annotated (pre-normalization, merged)",
                "Unknowns (normalized and merged)",
                "Annotated (POS only)",
                "Annotated semi-quant (POS only)",
                "Unknowns (POS only)",
                "Annotated (NEG only)",
                "Annotated semi-quant (NEG only)",
                "Unknowns (NEG only)",
            ]
        )
        ds_combo.grid(row=1, column=1, sticky="w")

        # === Tools ===
        tools = tk.Frame(self.main_frame, bg="white")
        tools.pack(pady=(10, 28), padx=24, fill="x")
        tools.grid_columnconfigure(0, weight=1)
        tools.grid_columnconfigure(1, weight=1)

        ttk.Label(tools, text="Analysis Tools", style="Section.TLabel").grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky="w")

        action_row = tk.Frame(tools, bg="white")
        action_row.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 12))

        self.runall_button = ttk.Button(action_row, text="RUN ALL", width=22, command=self.run_all)
        self.runall_button.pack(side="left")

        self.stop_button = ttk.Button(action_row, text="⛔ STOP NOW", width=14, command=self.hard_stop_now)
        self.stop_button.state(["disabled"])
        self.stop_button.pack(side="left", padx=(10, 0))

        multivariate_panel = ttk.LabelFrame(tools, text="Multivariate Analysis")
        multivariate_panel.grid(row=2, column=0, sticky="nsew", padx=(0, 8), pady=(0, 10))
        multivariate_panel.grid_columnconfigure(0, weight=1)
        multivariate_panel.grid_columnconfigure(1, weight=1)
        self.pca_button = ttk.Button(multivariate_panel, text="Run PCA", width=25, command=self.run_pca)
        self.plsda_button = ttk.Button(multivariate_panel, text="Run PLS-DA", width=25, command=self.run_plsda)
        self.heatmap_button = ttk.Button(multivariate_panel, text="Run Clustered Heatmap", width=25, command=self.run_heatmap)
        self.correlations_button = ttk.Button(multivariate_panel, text="Run Correlations", width=25, command=self.run_correlation_analysis)
        self.selected_heatmap_button = ttk.Button(multivariate_panel, text="Run Selected Lipid Heatmap", width=25, command=self.run_selected_heatmap)
        self.selected_heatmap_settings_button = ttk.Button(multivariate_panel, text="Select lipids for heatmap...", width=25, command=self.open_selected_heatmap_dialog)
        self.pca_button.grid(row=0, column=0, padx=8, pady=(8, 6), sticky="w")
        self.plsda_button.grid(row=0, column=1, padx=8, pady=(8, 6), sticky="w")
        self.heatmap_button.grid(row=1, column=0, padx=8, pady=(0, 8), sticky="w")
        self.correlations_button.grid(row=1, column=1, padx=8, pady=(0, 8), sticky="w")
        self.selected_heatmap_button.grid(row=2, column=0, padx=8, pady=(0, 6), sticky="w")
        self.selected_heatmap_settings_button.grid(row=2, column=1, padx=8, pady=(0, 6), sticky="w")
        ttk.Label(multivariate_panel, textvariable=self.var_selected_heatmap_summary, style="Subtle.TLabel").grid(
            row=3, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 8)
        )

        univariate_panel = ttk.LabelFrame(tools, text="Univariate Analysis")
        univariate_panel.grid(row=2, column=1, sticky="nsew", padx=(8, 0), pady=(0, 10))
        univariate_panel.grid_columnconfigure(0, weight=1)
        univariate_panel.grid_columnconfigure(1, weight=1)
        self.boxplots_button = ttk.Button(univariate_panel, text="Run Boxplots", width=25, command=self.run_boxplots)
        self.violin_button = ttk.Button(univariate_panel, text="Run Violin Plots", width=25, command=self.run_violin)
        self.enrichment_button = ttk.Button(univariate_panel, text="Run Enrichment", width=25, command=self.run_enrichment_analysis)
        self.upset_button = ttk.Button(univariate_panel, text="Run UpSet Plot", width=25, command=self.run_upset)
        self.advanceddiff_button = ttk.Button(univariate_panel, text="Run Advanced Differential", width=25, command=self.run_advanced_differential)
        self.boxplots_button.grid(row=0, column=0, padx=8, pady=(8, 6), sticky="w")
        self.violin_button.grid(row=0, column=1, padx=8, pady=(8, 6), sticky="w")
        self.enrichment_button.grid(row=1, column=0, padx=8, pady=(0, 6), sticky="w")
        self.upset_button.grid(row=1, column=1, padx=8, pady=(0, 6), sticky="w")
        self.advanceddiff_button.grid(row=2, column=0, padx=8, pady=(0, 8), sticky="w")

        distributions_panel = ttk.LabelFrame(tools, text="Distributions")
        distributions_panel.grid(row=3, column=0, sticky="nsew", padx=(0, 8), pady=(0, 10))
        distributions_panel.grid_columnconfigure(0, weight=1)
        distributions_panel.grid_columnconfigure(1, weight=1)
        self.classdist_button = ttk.Button(distributions_panel, text="Run Class Distributions", width=25, command=self.run_class_distributions)
        self.summint_button = ttk.Button(distributions_panel, text="Run Summed Int. per Class", width=25, command=self.run_class_sums)
        self.classviolinbox_button = ttk.Button(distributions_panel, text="Run Class Violin+Boxplots", width=25, command=self.run_class_violin_box)
        self.classcarbons_button = ttk.Button(distributions_panel, text="Run Carbon# DB", width=25, command=self.run_class_carbons_db)
        self.classdist_button.grid(row=0, column=0, padx=8, pady=(8, 6), sticky="w")
        self.summint_button.grid(row=0, column=1, padx=8, pady=(8, 6), sticky="w")
        self.classviolinbox_button.grid(row=1, column=0, padx=8, pady=(0, 8), sticky="w")
        self.classcarbons_button.grid(row=1, column=1, padx=8, pady=(0, 8), sticky="w")

        ratio_panel = ttk.LabelFrame(tools, text="Ratio Analysis")
        ratio_panel.grid(row=3, column=1, sticky="nsew", padx=(8, 0), pady=(0, 10))
        ratio_panel.grid_columnconfigure(0, weight=1)
        ratio_panel.grid_columnconfigure(1, weight=1)
        self.ratio_button = ttk.Button(ratio_panel, text="Run Ratio Analysis", width=25, command=self.run_ratio_analysis)
        self.ratio_settings_button = ttk.Button(ratio_panel, text="Ratio settings...", width=25, command=self.open_ratio_settings_dialog)
        ttk.Label(
            ratio_panel,
            text="Choose class ratios and annotation-specific ratios here before running.",
            style="Subtle.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(8, 6))
        self.ratio_button.grid(row=1, column=0, padx=8, pady=(0, 6), sticky="w")
        self.ratio_settings_button.grid(row=1, column=1, padx=8, pady=(0, 6), sticky="w")
        ttk.Label(ratio_panel, textvariable=self.var_ratio_settings_summary, style="Subtle.TLabel").grid(
            row=2, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 8)
        )
        
        # Auto-prepare on first open
        self.after(200, self._auto_prepare_or_warn)

        # Disable buttons if required files are missing
        if self.missing_files:
            for btn in (self.pca_button, self.plsda_button, self.heatmap_button, self.selected_heatmap_button, self.selected_heatmap_settings_button, self.volcano_button, self.boxplots_button, self.violin_button,
                self.correlations_button, self.classdist_button, self.summint_button, self.classviolinbox_button, self.classcarbons_button,
                self.enrichment_button, self.ratio_button, self.ratio_settings_button, self.upset_button, self.advanceddiff_button):
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
    # RATIO SETTINGS
    # ==========================================================
    def _ratio_settings_path(self) -> Path:
        return self._get_stats_dir() / "ratio_settings.json"

    def _default_ratio_settings(self) -> dict:
        return {
            "include_selected_class_ratios": True,
            "include_selected_product_ratios": True,
            "include_structural_class_ratios": True,
            "include_global_structural_ratios": True,
            "selected_class_ratios": [
                {"numerator": num, "denominator": den, "ratio_name": name, "category": "Class ratios"}
                for num, den, name in DEFAULT_CLASS_RATIO_DEFS
            ],
            "selected_product_ratios": [
                {"numerator": num, "denominator": den, "ratio_name": name, "category": "Product/substrate-like ratios"}
                for num, den, name in DEFAULT_PRODUCT_RATIO_DEFS
            ],
            "annotation_ratios": [],
        }

    def _normalize_ratio_settings(self, settings: Optional[dict] = None) -> dict:
        src = dict(self._default_ratio_settings())
        if isinstance(settings, dict):
            src.update(settings)

        def _normalize_defs(items, default_category):
            out = []
            seen = set()
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                numerator = str(item.get("numerator", "")).strip()
                denominator = str(item.get("denominator", "")).strip()
                ratio_name = str(item.get("ratio_name", "")).strip() or f"{numerator}/{denominator}"
                category = str(item.get("category", "")).strip() or default_category
                if not numerator or not denominator or not ratio_name:
                    continue
                key = (numerator.casefold(), denominator.casefold(), ratio_name.casefold(), category.casefold())
                if key in seen:
                    continue
                seen.add(key)
                out.append({
                    "numerator": numerator,
                    "denominator": denominator,
                    "ratio_name": ratio_name,
                    "category": category,
                })
            return out

        normalized = {
            "include_selected_class_ratios": bool(src.get("include_selected_class_ratios", True)),
            "include_selected_product_ratios": bool(src.get("include_selected_product_ratios", True)),
            "include_structural_class_ratios": bool(src.get("include_structural_class_ratios", True)),
            "include_global_structural_ratios": bool(src.get("include_global_structural_ratios", True)),
            "selected_class_ratios": _normalize_defs(src.get("selected_class_ratios", []), "Class ratios"),
            "selected_product_ratios": _normalize_defs(src.get("selected_product_ratios", []), "Product/substrate-like ratios"),
            "annotation_ratios": _normalize_defs(src.get("annotation_ratios", []), "Annotation-specific ratios"),
        }
        return normalized

    def _load_ratio_settings(self):
        p = self._ratio_settings_path()
        if p.exists():
            try:
                self.ratio_settings = self._normalize_ratio_settings(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                self.ratio_settings = self._default_ratio_settings()
        else:
            self.ratio_settings = self._default_ratio_settings()
        self._update_ratio_settings_summary()

    def _save_ratio_settings(self):
        self.ratio_settings = self._normalize_ratio_settings(self.ratio_settings)
        self._update_ratio_settings_summary()
        try:
            self._ratio_settings_path().write_text(json.dumps(self.ratio_settings, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _update_ratio_settings_summary(self):
        rs = self._normalize_ratio_settings(getattr(self, "ratio_settings", None))
        parts = []
        if rs["include_selected_class_ratios"]:
            parts.append(f"{len(rs['selected_class_ratios'])} class")
        if rs["include_selected_product_ratios"]:
            parts.append(f"{len(rs['selected_product_ratios'])} product")
        if rs["annotation_ratios"]:
            parts.append(f"{len(rs['annotation_ratios'])} annotation")
        if rs["include_structural_class_ratios"]:
            parts.append("class-structural")
        if rs["include_global_structural_ratios"]:
            parts.append("global-structural")
        self.var_ratio_settings_summary.set("Ratio settings: " + (", ".join(parts) if parts else "none selected"))

    def _get_ratio_settings(self) -> dict:
        return self._normalize_ratio_settings(getattr(self, "ratio_settings", None))

    def _candidate_ratio_dataset_paths(self) -> list[Path]:
        stats_dir = self._get_stats_dir()
        paths: list[Path] = []
        try:
            dataset_iter = self._datasets_for_selection()
        except Exception:
            return []
        for fname, label in dataset_iter:
            label_upper = str(label).upper()
            is_no_qc_label = ("WITHOUT_QCS" in label_upper or "NO_QCS" in label_upper)
            if not is_no_qc_label:
                continue
            fpath = stats_dir / fname
            if fpath.exists():
                paths.append(fpath)
        return paths

    def _enable_annotation_typeahead(self, combo: ttk.Combobox, all_values: list[str]):
        values = list(all_values or [])
        combo.configure(values=values)

        def _apply_matches(matches: list[str], typed: str):
            combo.configure(values=matches if matches else values)
            if matches:
                first = matches[0]
                if typed and first.casefold().startswith(typed.casefold()):
                    combo.set(first)
                    combo.selection_range(len(typed), tk.END)
                    combo.icursor(len(typed))

        def _filter_matches(_event=None):
            typed = combo.get().strip()
            if not typed:
                combo.configure(values=values)
                return
            starts = [item for item in values if item.casefold().startswith(typed.casefold())]
            contains = [item for item in values if typed.casefold() in item.casefold() and item not in starts]
            _apply_matches(starts + contains[:200], typed)

        def _reset_values(_event=None):
            if not combo.get().strip():
                combo.configure(values=values)

        combo.bind("<KeyRelease>", _filter_matches, add="+")
        combo.bind("<Button-1>", _reset_values, add="+")
        combo.bind("<FocusIn>", _reset_values, add="+")

    def _selected_heatmap_settings_path(self) -> Path:
        return self._get_stats_dir() / "selected_heatmap_annotations.json"

    def _update_selected_heatmap_summary(self):
        count = len(getattr(self, "selected_heatmap_annotations", []))
        if count == 0:
            text = "Selected heatmap: no lipids selected"
        else:
            text = f"Selected heatmap: {count} lipid{'s' if count != 1 else ''} selected"
        self.var_selected_heatmap_summary.set(text)

    def _load_selected_heatmap_settings(self):
        p = self._selected_heatmap_settings_path()
        items: list[str] = []
        if p.exists():
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
                items = [str(x).strip() for x in payload.get("selected_annotations", []) if str(x).strip()]
            except Exception:
                items = []
        self.selected_heatmap_annotations = items
        self._update_selected_heatmap_summary()

    def _save_selected_heatmap_settings(self):
        cleaned = [str(x).strip() for x in getattr(self, "selected_heatmap_annotations", []) if str(x).strip()]
        self.selected_heatmap_annotations = cleaned
        self._update_selected_heatmap_summary()
        try:
            self._selected_heatmap_settings_path().write_text(
                json.dumps({"selected_annotations": cleaned}, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def open_selected_heatmap_dialog(self):
        dataset_paths = self._candidate_ratio_dataset_paths()
        annotation_values = get_available_heatmap_annotations(str(dataset_paths[0])) if dataset_paths else []

        dlg = tk.Toplevel(self)
        dlg.title("Selected lipid heatmap")
        dlg.configure(bg="white")
        dlg.transient(self)
        dlg.grab_set()

        frame = tk.Frame(dlg, bg="white")
        frame.pack(fill="both", expand=True, padx=14, pady=12)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(3, weight=1)

        ttk.Label(frame, text="Choose lipids for the unclustered heatmap", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        helper_text = "Type to jump to matching annotations; the selected order becomes the heatmap row order."
        ttk.Label(frame, text=helper_text, style="Subtle.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 8))

        controls = tk.Frame(frame, bg="white")
        controls.grid(row=2, column=0, sticky="ew")
        controls.grid_columnconfigure(1, weight=1)

        ann_var = tk.StringVar()
        ttk.Label(controls, text="Lipid annotation", style="Body.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 6))
        ann_combo = ttk.Combobox(controls, textvariable=ann_var, values=annotation_values, width=60)
        ann_combo.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self._enable_annotation_typeahead(ann_combo, annotation_values)

        selected_tree = ttk.Treeview(frame, columns=("annotation",), show="headings", height=10)
        selected_tree.heading("annotation", text="Selected lipids in heatmap order")
        selected_tree.column("annotation", width=520, anchor="w")
        selected_tree.grid(row=3, column=0, sticky="nsew", pady=(8, 0))

        selected_rows = [str(x).strip() for x in self.selected_heatmap_annotations if str(x).strip()]

        def _refresh_tree():
            selected_tree.delete(*selected_tree.get_children(""))
            for idx, item in enumerate(selected_rows):
                selected_tree.insert("", "end", iid=str(idx), values=(item,))

        _refresh_tree()

        def _add_selected():
            annotation = ann_var.get().strip()
            if not annotation:
                return
            selected_rows.append(annotation)
            _refresh_tree()
            ann_var.set("")

        def _remove_selected():
            selected = selected_tree.selection()
            if not selected:
                return
            remove_ids = {int(item_id) for item_id in selected}
            selected_rows[:] = [row for idx, row in enumerate(selected_rows) if idx not in remove_ids]
            _refresh_tree()

        def _move_selected(delta: int):
            selected = selected_tree.selection()
            if len(selected) != 1:
                return
            idx = int(selected[0])
            new_idx = idx + delta
            if new_idx < 0 or new_idx >= len(selected_rows):
                return
            selected_rows[idx], selected_rows[new_idx] = selected_rows[new_idx], selected_rows[idx]
            _refresh_tree()
            selected_tree.selection_set(str(new_idx))

        btns = tk.Frame(frame, bg="white")
        btns.grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Button(btns, text="Add lipid", width=12, command=_add_selected).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="Remove", width=12, command=_remove_selected).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="Move up", width=12, command=lambda: _move_selected(-1)).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="Move down", width=12, command=lambda: _move_selected(1)).pack(side="left")

        actions = tk.Frame(frame, bg="white")
        actions.grid(row=5, column=0, sticky="e", pady=(12, 0))

        def _save_and_close():
            self.selected_heatmap_annotations = [str(x).strip() for x in selected_rows if str(x).strip()]
            self._save_selected_heatmap_settings()
            dlg.destroy()

        ttk.Button(actions, text="Save", width=12, command=_save_and_close).pack(side="left", padx=4)
        ttk.Button(actions, text="Cancel", width=12, command=dlg.destroy).pack(side="left", padx=4)

        dlg.update_idletasks()
        dlg.geometry(f"+{self.winfo_rootx()+110}+{self.winfo_rooty()+90}")

    def open_ratio_settings_dialog(self):
        settings = self._get_ratio_settings()
        dataset_paths = self._candidate_ratio_dataset_paths()
        annotation_values = get_available_annotation_labels(str(dataset_paths[0])) if dataset_paths else []

        dlg = tk.Toplevel(self)
        dlg.title("Ratio settings")
        dlg.configure(bg="white")
        dlg.transient(self)
        dlg.grab_set()

        frame = tk.Frame(dlg, bg="white")
        frame.pack(fill="both", expand=True, padx=14, pady=12)

        ttk.Label(frame, text="Predefined ratios", style="Section.TLabel").grid(row=0, column=0, columnspan=4, sticky="w")

        include_class_var = tk.BooleanVar(value=settings["include_selected_class_ratios"])
        include_product_var = tk.BooleanVar(value=settings["include_selected_product_ratios"])
        include_structural_var = tk.BooleanVar(value=settings["include_structural_class_ratios"])
        include_global_var = tk.BooleanVar(value=settings["include_global_structural_ratios"])

        ttk.Checkbutton(frame, text="Include selected class ratios", variable=include_class_var).grid(row=1, column=0, sticky="w", pady=(6, 2))
        ttk.Checkbutton(frame, text="Include selected product/substrate ratios", variable=include_product_var).grid(row=1, column=1, sticky="w", pady=(6, 2), padx=(16, 0))
        ttk.Checkbutton(frame, text="Include within-class structural ratios", variable=include_structural_var).grid(row=2, column=0, sticky="w", pady=(2, 8))
        ttk.Checkbutton(frame, text="Include global structural ratios", variable=include_global_var).grid(row=2, column=1, sticky="w", pady=(2, 8), padx=(16, 0))

        class_defs = [
            {"numerator": num, "denominator": den, "ratio_name": name, "category": "Class ratios"}
            for num, den, name in DEFAULT_CLASS_RATIO_DEFS
        ]
        product_defs = [
            {"numerator": num, "denominator": den, "ratio_name": name, "category": "Product/substrate-like ratios"}
            for num, den, name in DEFAULT_PRODUCT_RATIO_DEFS
        ]
        selected_class_names = {item["ratio_name"] for item in settings["selected_class_ratios"]}
        selected_product_names = {item["ratio_name"] for item in settings["selected_product_ratios"]}

        class_box = ttk.LabelFrame(frame, text="Class ratios")
        class_box.grid(row=3, column=0, sticky="nsew", padx=(0, 8))
        product_box = ttk.LabelFrame(frame, text="Product/substrate-like ratios")
        product_box.grid(row=3, column=1, sticky="nsew", padx=(8, 0))
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        class_ratio_vars = {}
        for idx, item in enumerate(class_defs):
            var = tk.BooleanVar(value=item["ratio_name"] in selected_class_names)
            class_ratio_vars[item["ratio_name"]] = var
            ttk.Checkbutton(class_box, text=item["ratio_name"], variable=var).grid(row=idx, column=0, sticky="w", padx=8, pady=2)

        product_ratio_vars = {}
        for idx, item in enumerate(product_defs):
            var = tk.BooleanVar(value=item["ratio_name"] in selected_product_names)
            product_ratio_vars[item["ratio_name"]] = var
            ttk.Checkbutton(product_box, text=item["ratio_name"], variable=var).grid(row=idx, column=0, sticky="w", padx=8, pady=2)

        ttk.Label(frame, text="Annotation-specific ratios", style="Section.TLabel").grid(row=4, column=0, columnspan=4, sticky="w", pady=(12, 4))
        helper_text = "Choose from the current no-QC dataset" if annotation_values else "No prepared no-QC dataset found yet; manual entry still works"
        ttk.Label(frame, text=helper_text, style="Subtle.TLabel").grid(row=5, column=0, columnspan=4, sticky="w", pady=(0, 6))

        ann_controls = tk.Frame(frame, bg="white")
        ann_controls.grid(row=6, column=0, columnspan=2, sticky="ew")
        ann_controls.grid_columnconfigure(1, weight=1)
        ann_controls.grid_columnconfigure(3, weight=1)

        ann_num_var = tk.StringVar()
        ann_den_var = tk.StringVar()
        ann_name_var = tk.StringVar()

        ttk.Label(ann_controls, text="Numerator", style="Body.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 6))
        num_combo = ttk.Combobox(ann_controls, textvariable=ann_num_var, values=annotation_values, width=42)
        num_combo.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        ttk.Label(ann_controls, text="Denominator", style="Body.TLabel").grid(row=0, column=2, sticky="w", padx=(0, 6))
        den_combo = ttk.Combobox(ann_controls, textvariable=ann_den_var, values=annotation_values, width=42)
        den_combo.grid(row=0, column=3, sticky="ew")
        self._enable_annotation_typeahead(num_combo, annotation_values)
        self._enable_annotation_typeahead(den_combo, annotation_values)
        ttk.Label(ann_controls, text="Label", style="Body.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=(8, 0))
        ttk.Entry(ann_controls, textvariable=ann_name_var, width=42).grid(row=1, column=1, sticky="ew", pady=(8, 0))

        ann_tree = ttk.Treeview(frame, columns=("ratio", "numerator", "denominator"), show="headings", height=7)
        ann_tree.heading("ratio", text="Ratio")
        ann_tree.heading("numerator", text="Numerator")
        ann_tree.heading("denominator", text="Denominator")
        ann_tree.column("ratio", width=220, anchor="w")
        ann_tree.column("numerator", width=260, anchor="w")
        ann_tree.column("denominator", width=260, anchor="w")
        ann_tree.grid(row=7, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        frame.grid_rowconfigure(7, weight=1)

        annotation_rows = []

        def _refresh_annotation_tree():
            ann_tree.delete(*ann_tree.get_children(""))
            for idx, item in enumerate(annotation_rows):
                ann_tree.insert("", "end", iid=str(idx), values=(item["ratio_name"], item["numerator"], item["denominator"]))

        for item in settings["annotation_ratios"]:
            annotation_rows.append({
                "numerator": item["numerator"],
                "denominator": item["denominator"],
                "ratio_name": item["ratio_name"],
                "category": "Annotation-specific ratios",
            })
        _refresh_annotation_tree()

        def _add_annotation_ratio():
            numerator = ann_num_var.get().strip()
            denominator = ann_den_var.get().strip()
            ratio_name = ann_name_var.get().strip() or f"{numerator}/{denominator}"
            if not numerator or not denominator:
                messagebox.showwarning("Missing annotation", "Choose both numerator and denominator annotations.", parent=dlg)
                return
            annotation_rows.append({
                "numerator": numerator,
                "denominator": denominator,
                "ratio_name": ratio_name,
                "category": "Annotation-specific ratios",
            })
            _refresh_annotation_tree()
            ann_name_var.set("")

        def _remove_annotation_ratio():
            selected = ann_tree.selection()
            if not selected:
                return
            remove_ids = {int(item_id) for item_id in selected}
            annotation_rows[:] = [row for idx, row in enumerate(annotation_rows) if idx not in remove_ids]
            _refresh_annotation_tree()

        btn_row = tk.Frame(frame, bg="white")
        btn_row.grid(row=8, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(btn_row, text="Add annotation ratio", command=_add_annotation_ratio, width=22).pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text="Remove selected", command=_remove_annotation_ratio, width=16).pack(side="left")

        def _save_and_close():
            selected_class = [item for item in class_defs if class_ratio_vars[item["ratio_name"]].get()]
            selected_product = [item for item in product_defs if product_ratio_vars[item["ratio_name"]].get()]
            self.ratio_settings = self._normalize_ratio_settings({
                "include_selected_class_ratios": include_class_var.get(),
                "include_selected_product_ratios": include_product_var.get(),
                "include_structural_class_ratios": include_structural_var.get(),
                "include_global_structural_ratios": include_global_var.get(),
                "selected_class_ratios": selected_class,
                "selected_product_ratios": selected_product,
                "annotation_ratios": annotation_rows,
            })
            self._save_ratio_settings()
            dlg.destroy()

        action_row = tk.Frame(frame, bg="white")
        action_row.grid(row=9, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(action_row, text="Save", command=_save_and_close, width=12).pack(side="left", padx=4)
        ttk.Button(action_row, text="Cancel", command=dlg.destroy, width=12).pack(side="left", padx=4)

        dlg.update_idletasks()
        dlg.geometry(f"+{self.winfo_rootx()+100}+{self.winfo_rooty()+80}")

    # ==========================================================
    # DATA LOADING (for summary + guardrails)
    # ==========================================================
    def _load_data_files(self):
        """Load annotated, unknown, before-normalization, and group files if available. Return list of missing must-haves."""
        
        # ALL FEATURES
        candidates = {
            "Final_Annotated.csv": [
                self.output_folder / "Final_Annotated.csv",
                self.output_folder / "POS" / "Pos_Final_Annotated.csv",
                self.output_folder / "NEG" / "Neg_Final_Annotated.csv",
            ],

            "Final_Annotated_semi_quant.csv": [
                self.output_folder / "Final_Annotated_semi_quant.csv",
                self.output_folder / "POS" / "Pos_Final_Annotated_semi_quant.csv",
                self.output_folder / "NEG" / "Neg_Final_Annotated_semi_quant.csv",
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
            "Final_Annotated_semi_quant.csv": "df_annotated_semi",
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
                        messagebox.showwarning("File Error", f"Failed to read {p.name}:\n{e}")
            if not loaded:
                missing.append(logical_name)
        # Only Annotated and groups are mandatory to start; others optional
        return [m for m in missing if m in ("Final_Annotated.csv", "sample_groups.csv")]

    def _make_summary_text(self):
        if self.missing_files:
            return (f"⚠ Some required files are missing:\n  - {', '.join(self.missing_files)}\n\n"
                    f"Please run the processing pipeline completely before proceeding.")
        n_ann = len(self.df_annotated) if getattr(self, "df_annotated", None) is not None else 0
        n_ann_semi = len(self.df_annotated_semi) if getattr(self, "df_annotated_semi", None) is not None else 0
        n_unk = len(self.df_unknowns) if getattr(self, "df_unknowns", None) is not None else 0
        n_bfn = len(self.df_before_norm) if getattr(self, "df_before_norm", None) is not None else 0
        n_grp = len(self.df_groups) if getattr(self, "df_groups", None) is not None else 0
        return (f"Loaded {n_ann} annotated compounds\n"
                f"Loaded {n_ann_semi} annotated semi-quant compounds\n"
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

            # Annotated semi-quant
            "Final_Annotated_semi_quant.csv",
            "Final_Annotated_semi_quant_Without_QCs.csv",
            "Final_Annotated_semi_quant_HighConf.csv",
            "Final_Annotated_semi_quant_Without_QCs_HighConf.csv",
            
            # Annotated with missing
            "Final_Annotated_with_missing.csv",
            "Final_Annotated_with_missing_Without_QCs.csv",
            "Final_Annotated_with_missing_HighConf.csv",
            "Final_Annotated_with_missing_Without_QCs_HighConf.csv",

            # Annotated semi-quant with missing
            "Final_Annotated_semi_quant_with_missing.csv",
            "Final_Annotated_semi_quant_with_missing_Without_QCs.csv",
            "Final_Annotated_semi_quant_with_missing_HighConf.csv",
            "Final_Annotated_semi_quant_with_missing_Without_QCs_HighConf.csv",

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
            try:
                self._load_ratio_settings()
            except Exception:
                self.ratio_settings = self._default_ratio_settings()
                self._update_ratio_settings_summary()
            try:
                self._load_selected_heatmap_settings()
            except Exception:
                self.selected_heatmap_annotations = []
                self._update_selected_heatmap_summary()
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

    def _get_figure_options(self):
        try:
            dpi = int(str(self.var_dpi.get()).strip())
        except Exception:
            dpi = 100
        dpi = max(72, min(dpi, 1200))
        return dpi, bool(self.var_publication_theme.get())

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
            self.pca_button, self.plsda_button, self.heatmap_button, self.selected_heatmap_button, self.selected_heatmap_settings_button,
            self.volcano_button, self.boxplots_button, self.violin_button,
            self.correlations_button, self.classdist_button, self.summint_button, self.classviolinbox_button,
            self.classcarbons_button, self.enrichment_button, self.ratio_button, self.ratio_settings_button, self.upset_button,
            self.advanceddiff_button,
        )

    def _set_busy(self, busy: bool, label: Optional[str] = None):
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

    def run_selected_heatmap(self):
        if self._is_running:
            self._toast("Another analysis is already running")
            return
        if not self.selected_heatmap_annotations:
            self._toast("Choose lipids for the selected heatmap first")
            return
        self._worker_thread = threading.Thread(
            target=self._run_analysis,
            args=("Selected_Heatmap",),
            daemon=True,
            name="LQ-StatsWorker-SelectedHeatmap",
        )
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

    def run_enrichment_analysis(self):
        if self._is_running:
            self._toast("Another analysis is already running")
            return
        self._worker_thread = threading.Thread(
            target=self._run_analysis,
            args=("Enrichment",),
            daemon=True,
            name="LQ-StatsWorker-Enrichment",
        )
        self._worker_thread.start()

    def run_ratio_analysis(self):
        if self._is_running:
            self._toast("Another analysis is already running")
            return
        self._worker_thread = threading.Thread(
            target=self._run_analysis,
            args=("Ratios",),
            daemon=True,
            name="LQ-StatsWorker-Ratios",
        )
        self._worker_thread.start()

    def run_upset(self):
        if self._is_running:
            self._toast("Another analysis is already running")
            return
        self._worker_thread = threading.Thread(
            target=self._run_analysis,
            args=("UpSet",),
            daemon=True,
            name="LQ-StatsWorker-UpSet",
        )
        self._worker_thread.start()

    def run_advanced_differential(self):
        if self._is_running:
            self._toast("Another analysis is already running")
            return
        self._worker_thread = threading.Thread(
            target=self._run_analysis,
            args=("Advanced_Differential",),
            daemon=True,
            name="LQ-StatsWorker-AdvancedDifferential",
        )
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
            order = ["PCA", "PLS-DA", "Heatmap", "Selected_Heatmap", "Correlations", "Class_Distributions", "Class_Sums", "Class_violin_box", "Class_Carbons_DB", "Enrichment", "Ratios", "Advanced_Differential", "UpSet", "Volcano", "Boxplots", "Violin",]
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
                    ("Final_Annotated_Without_QCs.csv", "No_QCs"),
                    ("Final_Annotated_HighConf.csv", "HighConf_With_QCs"),
                    ("Final_Annotated_Without_QCs_HighConf.csv", "HighConf_No_QCs"),
                ]
            
        elif ds == "Annotated (with missing values)":
            return [
                    ("Final_Annotated_with_missing.csv", "Missing_With_QCs"),
                    ("Final_Annotated_with_missing_Without_QCs.csv", "Missing_No_QCs"),
                    ("Final_Annotated_with_missing_HighConf.csv", "Missing_HighConf_With_QCs"),
                    ("Final_Annotated_with_missing_Without_QCs_HighConf.csv", "Missing_HighConf_No_QCs"),
                ]
            
        elif ds == "Annotated semi-quant (normalized and merged)":
            return [
                    ("Final_Annotated_semi_quant.csv", "SemiQuant_With_QCs"),
                    ("Final_Annotated_semi_quant_Without_QCs.csv", "SemiQuant_No_QCs"),
                    ("Final_Annotated_semi_quant_HighConf.csv", "SemiQuant_HighConf_With_QCs"),
                    ("Final_Annotated_semi_quant_Without_QCs_HighConf.csv", "SemiQuant_HighConf_No_QCs"),
                ]
            
        elif ds == "Annotated semi-quant (with missing values)":
            return [
                    ("Final_Annotated_semi_quant_with_missing.csv", "Missing_SemiQuant_With_QCs"),
                    ("Final_Annotated_semi_quant_with_missing_Without_QCs.csv", "Missing_SemiQuant_No_QCs"),
                    ("Final_Annotated_semi_quant_with_missing_HighConf.csv", "Missing_SemiQuant_HighConf_With_QCs"),
                    ("Final_Annotated_semi_quant_with_missing_Without_QCs_HighConf.csv", "Missing_SemiQuant_HighConf_No_QCs"),
                ]

        elif ds == "Annotated (pre-normalization, merged)":
            return [
                    ("Final_Annotated_BeforeNorm.csv", "BeforeNorm_With_QCs"),
                    ("Final_Annotated_BeforeNorm_Without_QCs.csv", "BeforeNorm_No_QCs"),
                ]

        elif ds == "Unknowns (normalized and merged)":
            return [
                    ("Final_Unknowns.csv", "Unk_With_QCs"),
                    ("Final_Unknowns_Without_QCs.csv", "Unk_No_QCs"),
                ]

        elif ds == "Annotated (POS only)":
            return [
                    ("POS_Final_Annotated.csv", "POS"),
                    ("POS_Final_Annotated_Without_QCs.csv", "POS_No_QCs"),
                ]
            
        elif ds == "Annotated semi-quant (POS only)":
            return [
                    ("POS_Final_Annotated_semi_quant.csv", "POS_SemiQuant"),
                    ("POS_Final_Annotated_semi_quant_Without_QCs.csv", "POS_SemiQuant_No_QCs"),
                ]

        elif ds == "Unknowns (POS only)":
            return [
                    ("POS_Final_Unknowns.csv", "POS"),
                    ("POS_Final_Unknowns_Without_QCs.csv", "POS_Without_QCs"),
                ]

        elif ds == "Annotated (NEG only)":
            return [
                    ("NEG_Final_Annotated.csv", "NEG"),
                    ("NEG_Final_Annotated_Without_QCs.csv", "NEG_Without_QCs"),
                ]
        
        elif ds == "Annotated semi-quant (NEG only)":
            return [
                    ("NEG_Final_Annotated_semi_quant.csv", "NEG_SemiQuant"),
                    ("NEG_Final_Annotated_semi_quant_Without_QCs.csv", "NEG_SemiQuant_No_QCs"),
                ]

        elif ds == "Unknowns (NEG only)":
            return [
                    ("NEG_Final_Unknowns.csv", "NEG"),
                    ("NEG_Final_Unknowns_Without_QCs.csv", "NEG_No_QCs"),
                ]

        else:
            raise FileNotFoundError(f"Unsupported dataset selection: {ds}")


    def _datasets_for_upset_selection(self) -> list[tuple[str, str]]:
        ds = (self.var_dataset.get() or "").strip()

        if ds in {"Annotated (normalized and merged)", "Annotated (with missing values)"}:
            return [
                ("Final_Annotated_with_missing_Without_QCs.csv", "No_QCs"),
                ("Final_Annotated_with_missing_Without_QCs_HighConf.csv", "HighConf_No_QCs"),
            ]

        elif ds in {"Annotated semi-quant (normalized and merged)", "Annotated semi-quant (with missing values)"}:
            return [
                ("Final_Annotated_semi_quant_with_missing_Without_QCs.csv", "SemiQuant_No_QCs"),
                ("Final_Annotated_semi_quant_with_missing_Without_QCs_HighConf.csv", "SemiQuant_HighConf_No_QCs"),
            ]

        elif ds == "Annotated (POS only)":
            return [
                ("POS_Final_Annotated_with_missing_Without_QCs.csv", "POS_No_QCs"),
            ]

        elif ds == "Annotated semi-quant (POS only)":
            return [
                ("POS_Final_Annotated_semi_quant_with_missing_Without_QCs.csv", "POS_SemiQuant_No_QCs"),
            ]

        elif ds == "Annotated (NEG only)":
            return [
                ("NEG_Final_Annotated_with_missing_Without_QCs.csv", "NEG_No_QCs"),
            ]

        elif ds == "Annotated semi-quant (NEG only)":
            return [
                ("NEG_Final_Annotated_semi_quant_with_missing_Without_QCs.csv", "NEG_SemiQuant_No_QCs"),
            ]

        else:
            return []

    
    def _run_analysis(self, analysis_type, _sequence_mode: bool = False):
        outer_manages = _sequence_mode
        if not outer_manages:
            if not self._acquire_runner(analysis_type): return

        try:
            stats_dir = self._get_stats_dir()
            cleaned_group_file = stats_dir / "sample_groups_cleaned.csv"
            base_group_file = cleaned_group_file if cleaned_group_file.exists() else (self.output_folder / "sample_groups.csv")

            if not self._stats_ready():
                self._toast("⚠ Prepare datasets first (use the button above)")
                return

            if not base_group_file.exists():
                base_group_file = None

            if not outer_manages:
                self._toast(f"Running {analysis_type}…")

            plt.close("all")
            
            dataset_iter = self._datasets_for_upset_selection() if analysis_type == "UpSet" else self._datasets_for_selection()
            ran_any = False
            for fname, label in dataset_iter:

                fpath = stats_dir / fname
                if not fpath.exists():
                    continue

                # Tool-specific gating
                needs_no_qc = {"PLS-DA", "Volcano", "Heatmap", "Selected_Heatmap", "Boxplots", "Violin", "Correlations", "Class_Distributions", "Class_Sums", "Class_violin_box", "Class_Carbons_DB", "Enrichment", "Ratios", "Advanced_Differential", "UpSet"}
                label_upper = str(label).upper()
                is_no_qc_label = (
                    "WITHOUT_QCS" in label_upper
                    or "NO_QCS" in label_upper
                )
                if analysis_type in needs_no_qc and not is_no_qc_label:
                    continue
                if analysis_type in {"Boxplots", "Violin"} and ("HIGHCONF" in label_upper):
                    continue
                subfolder = stats_dir / analysis_type / label
                subfolder.mkdir(parents=True, exist_ok=True)

                matched_group_file = None
                if base_group_file is not None:
                    matched_group_file = self._build_dataset_specific_group_file(
                        dataset_path=fpath,
                        base_group_file=base_group_file,
                        out_dir=subfolder
                    )

                # Optional palette
                try:
                    self._load_palette()
                    palette = self.group_colors or None
                except Exception:
                    palette = None
                figure_dpi, publication_theme = self._get_figure_options()

                try:
                    ran_any = True
                    if analysis_type == "PCA":
                        run_pca(
                            fpath,
                            matched_group_file,
                            subfolder,
                            group_colors=palette,
                            group_order=self.group_order,
                            dpi=figure_dpi,
                            publication_theme=publication_theme,
                        )
                    elif analysis_type == "PLS-DA":
                        run_plsda(
                            fpath,
                            matched_group_file,
                            subfolder,
                            group_colors=palette,
                            group_order=self.group_order,
                            dpi=figure_dpi,
                            publication_theme=publication_theme,
                        )
                    elif analysis_type == "Heatmap":
                        run_heatmap(
                            fpath,
                            matched_group_file,
                            subfolder,
                            group_colors=palette,
                            group_order=self.group_order,
                            dpi=figure_dpi,
                            publication_theme=publication_theme,
                        )
                    elif analysis_type == "Selected_Heatmap":
                        run_selected_lipid_heatmap(
                            fpath,
                            matched_group_file,
                            subfolder,
                            selected_annotations=self.selected_heatmap_annotations,
                            group_colors=palette,
                            group_order=self.group_order,
                            dpi=figure_dpi,
                            publication_theme=publication_theme,
                        )
                    elif analysis_type == "Volcano":
                        fc, fdr, p = self._get_volcano_thresholds()
                        run_volcano(
                            fpath, matched_group_file, subfolder,
                            sample_type=self.sample_type,
                            p_value_threshold=p, fdr_threshold=fdr, fold_change_threshold=fc,
                            group_colors=palette, group_order=self.group_order,
                            annotate_labels=bool(self.var_volcano_labels.get()),
                            dpi=figure_dpi, publication_theme=publication_theme,
                        )
                    elif analysis_type == "Boxplots":
                        run_boxplots(
                            fpath,
                            matched_group_file,
                            subfolder,
                            group_order=self.group_order,
                            group_colors=palette,
                            dpi=figure_dpi,
                            publication_theme=publication_theme,
                        )
                    elif analysis_type == "Violin":
                        run_violinplots(
                            fpath,
                            matched_group_file,
                            subfolder,
                            group_order=self.group_order,
                            group_colors=palette,
                            dpi=figure_dpi,
                            publication_theme=publication_theme,
                        )
                    elif analysis_type == "Correlations":
                        run_correlation_analysis(fpath, matched_group_file, subfolder, group_order=self.group_order)
                    elif analysis_type == "Class_Distributions":
                        run_class_distributions(
                            fpath,
                            matched_group_file,
                            subfolder,
                            group_colors=palette,
                            group_order=self.group_order,
                            sample_type=self.sample_type,
                            unknown_policy="append",
                            dataset_label=self.var_dataset.get(),
                            dpi=figure_dpi,
                            publication_theme=publication_theme,
                        )
                    elif analysis_type == "Class_Sums":
                        run_class_sums(
                            fpath,
                            matched_group_file,
                            subfolder,
                            group_colors=palette,
                            group_order=self.group_order,
                            sample_type=self.sample_type,
                            dataset_label=self.var_dataset.get(),
                        )
                    elif analysis_type == "Class_violin_box":
                        run_class_violin_box(
                            fpath,
                            matched_group_file,
                            subfolder,
                            group_colors=palette,
                            group_order=self.group_order,
                            dpi=figure_dpi,
                            publication_theme=publication_theme,
                        )
                    elif analysis_type == "Class_Carbons_DB":
                        run_class_carbons_db(fpath, matched_group_file, subfolder, group_colors=palette, group_order=self.group_order, exclude_qc=True,)
                    elif analysis_type == "Enrichment":
                        run_enrichment_analysis(
                            fpath,
                            matched_group_file,
                            subfolder,
                            group_colors=palette,
                            group_order=self.group_order,
                            exclude_qc=True,
                            dpi=figure_dpi,
                            publication_theme=publication_theme,
                        )
                    elif analysis_type == "Ratios":
                        run_ratio_analysis(
                            fpath,
                            matched_group_file,
                            subfolder,
                            group_colors=palette,
                            group_order=self.group_order,
                            exclude_qc=True,
                            dpi=figure_dpi,
                            publication_theme=publication_theme,
                            ratio_settings=self._get_ratio_settings(),
                        )
                    elif analysis_type == "Advanced_Differential":
                        run_advanced_differential_analysis(
                            fpath,
                            matched_group_file,
                            subfolder,
                            group_colors=palette,
                            group_order=self.group_order,
                            exclude_qc=True,
                            dpi=figure_dpi,
                            publication_theme=publication_theme,
                        )
                    elif analysis_type == "UpSet":
                        run_upset_plot(
                            fpath,
                            matched_group_file,
                            subfolder,
                            group_order=self.group_order,
                            group_colors=palette,
                            min_fraction=0.5,
                            min_samples=1,
                            top_n_intersections=10,
                            max_classes=20,
                            dpi=figure_dpi,
                            publication_theme=publication_theme,
                        )

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
    
    # polarity prefixes must be preserved for POS/NEG-specific datasets.
    def _clean_sample_name(self, name: str) -> str:
        if not isinstance(name, str): return name
        cleaned = name
        cleaned = re.sub(r"\[?POS\]?|\[?NEG\]?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^(P_|N_)", "", cleaned)
        cleaned = re.split(r"_P[12]", cleaned)[0]
        return cleaned.strip("_- ")

    def _normalize_sample_name_for_matching(self, name: str) -> str:
        """
        Normalize sample names so group-file names can be matched to stats-table columns.
        Use this only for matching, not for rewriting raw source files.
        """
        if not isinstance(name, str):
            return str(name)

        cleaned = str(name).strip()

        # remove bracketed or plain polarity tags
        cleaned = re.sub(r"\[?POS\]?|\[?NEG\]?", "", cleaned, flags=re.IGNORECASE)

        # remove leading polarity prefixes
        cleaned = re.sub(r"^(P_|N_)", "", cleaned, flags=re.IGNORECASE)

        # remove trailing polarity suffixes if they exist
        cleaned = re.sub(r"(_P[12]|_N[12])$", "", cleaned, flags=re.IGNORECASE)

        return cleaned.strip("_- ")

    def _build_dataset_specific_group_file(self, dataset_path: Path, base_group_file: Path, out_dir: Path) -> Optional[Path]:
        """
        Create a group file whose Sample names match the actual sample columns in dataset_path.
        This prevents mismatches when stats tables use cleaned sample names but group files still
        contain P_/N_ prefixes or related polarity tags.
        """
        if not dataset_path.exists() or not base_group_file.exists():
            return None

        try:
            df_data = pd.read_csv(dataset_path, nrows=5, low_memory=False)
            df_groups = pd.read_csv(base_group_file, low_memory=False)
        except Exception:
            print("[GroupAlign] Failed to read dataset or group file.", flush=True)
            print(traceback.format_exc(), flush=True)
            return None

        if "Sample" not in df_groups.columns or "Group" not in df_groups.columns:
            return None

        # detect metadata columns exactly as in preparation
        meta_keep = [
            "UniqueID","RT (min)","m/z","Polarity","Annotation","Annotation Type",
            "Annotation Source","Headgroup","Lipid Class","Δm/z (mDa)","Δm/z (ppm)",
            "MS/MS score","Annotation tier","mSigma","Molecular Formula","Plasmenyl?",
            "Number of carbons in fatty acyls","Double bond equivalents","Chain type",
            "PUFA?","Modifications","# of modifications","Oxidized?",
            "RSD QCs (%)","RSD Samples (%)"
        ]
        meta_cols = [c for c in meta_keep if c in df_data.columns]

        data_sample_cols = [
            c for c in df_data.columns
            if c not in meta_cols
            and "rsd" not in c.lower()
        ]

        data_sample_set = set(map(str, data_sample_cols))

        g = df_groups.copy()
        g["Sample"] = g["Sample"].astype(str).str.strip()

        # first try exact match
        exact = g[g["Sample"].isin(data_sample_set)].copy()

        if len(exact) == len(data_sample_set):
            aligned = exact.drop_duplicates(subset=["Sample"], keep="first").copy()
        else:
            # fall back to normalized matching
            g["Sample_normalized"] = g["Sample"].map(self._normalize_sample_name_for_matching)

            mapping = {}
            for sample_col in data_sample_cols:
                normalized_col = self._normalize_sample_name_for_matching(sample_col)
                mapping[normalized_col] = sample_col

            g = g[g["Sample_normalized"].isin(mapping)].copy()
            g["Sample"] = g["Sample_normalized"].map(mapping)
            aligned = g.drop_duplicates(subset=["Sample"], keep="first").copy()
            aligned = aligned.drop(columns=["Sample_normalized"], errors="ignore")

        # keep only rows that truly exist in the dataset
        aligned = aligned[aligned["Sample"].isin(data_sample_set)].copy()

        if aligned.empty:
            print(f"[GroupAlign] No matching samples for dataset: {dataset_path.name}", flush=True)
            return None

        # preserve dataset column order
        aligned["__order"] = aligned["Sample"].map({s: i for i, s in enumerate(data_sample_cols)})
        aligned = aligned.sort_values("__order").drop(columns="__order").reset_index(drop=True)

        out_path = out_dir / f"{dataset_path.stem}__sample_groups_matched.csv"
        aligned.to_csv(out_path, index=False, encoding="utf-8-sig")
        return out_path
    
    # ==========================================================
    # PREPARE STATISTICAL DATASETS
    # ==========================================================
    
    def _restore_missing_values_from_debug(self, df_final: pd.DataFrame) -> pd.DataFrame:
        """
        Start from the fully processed Final_Annotated-style table and restore NaN where
        the pre-imputation polarity-specific collapsed debug files had missing values.
        It preserves all processed intensities and metadata from df_final. It only turns
        imputed cells back into NaN.
        """
        if df_final is None or df_final.empty:
            return df_final

        out = df_final.copy()

        if "UniqueID" not in out.columns:
            print("[MissingRestore] 'UniqueID' not found in final dataframe. Skipping missing restoration.", flush=True)
            return out

        meta_keep = [
            "UniqueID", "RT (min)", "m/z", "Polarity", "Annotation", "Annotation Type",
            "Annotation Source", "Headgroup", "Lipid Class", "Δm/z (mDa)", "Δm/z (ppm)",
            "MS/MS score", "Annotation tier", "mSigma", "Molecular Formula", "Plasmenyl?",
            "Number of carbons in fatty acyls", "Double bond equivalents", "Chain type",
            "PUFA?", "Modifications", "# of modifications", "Oxidized?",
            "RSD QCs (%)", "RSD Samples (%)"
        ]
        meta_cols = [c for c in meta_keep if c in out.columns]

        final_sample_cols = [
            c for c in out.columns
            if c not in meta_cols and "rsd" not in str(c).lower()
        ]

        if not final_sample_cols:
            print("[MissingRestore] No sample columns detected in final dataframe.", flush=True)
            return out

        out["UniqueID"] = out["UniqueID"].astype(str).str.strip()
        final_uid_to_idx = pd.Series(out.index.values, index=out["UniqueID"]).to_dict()

        final_sample_norm_to_col = {
            self._normalize_sample_name_for_matching(str(c)): c
            for c in final_sample_cols
        }

        debug_files = [
            ("POS", self.output_folder / "POS" / "debug" / "Pos_2-Final_annotated_results_adducts_collapsed.csv"),
            ("NEG", self.output_folder / "NEG" / "debug" / "Neg_2-Final_annotated_results_adducts_collapsed.csv"),
        ]

        total_restored = 0

        for polarity_label, debug_path in debug_files:
            if not debug_path.exists():
                print(f"[MissingRestore] Debug file not found for {polarity_label}: {debug_path}", flush=True)
                continue

            try:
                df_debug = pd.read_csv(debug_path, low_memory=False)
            except Exception:
                print(f"[MissingRestore] Failed to load {debug_path}", flush=True)
                print(traceback.format_exc(), flush=True)
                continue

            if "UniqueID" not in df_debug.columns:
                print(f"[MissingRestore] 'UniqueID' missing in {debug_path.name}. Skipping this file.", flush=True)
                continue

            df_debug["UniqueID"] = df_debug["UniqueID"].astype(str).str.strip()

            # Debug files use bare numeric IDs. Final_Annotated uses polarity-prefixed IDs.
            if polarity_label == "POS":
                df_debug["UniqueID"] = "P_" + df_debug["UniqueID"]
            elif polarity_label == "NEG":
                df_debug["UniqueID"] = "N_" + df_debug["UniqueID"]

            df_debug = df_debug.drop_duplicates(subset=["UniqueID"], keep="first").copy()

            debug_sample_cols = []
            for c in df_debug.columns:
                norm_c = self._normalize_sample_name_for_matching(str(c))
                if norm_c in final_sample_norm_to_col:
                    debug_sample_cols.append(c)

            if not debug_sample_cols:
                print(f"[MissingRestore] No matching debug sample columns found in {debug_path.name}", flush=True)
                continue

            debug_uid_to_idx = pd.Series(df_debug.index.values, index=df_debug["UniqueID"]).to_dict()
            shared_uids = [uid for uid in debug_uid_to_idx if uid in final_uid_to_idx]

            if not shared_uids:
                print(f"[MissingRestore] No shared UniqueID values with {debug_path.name}", flush=True)
                continue

            restored_this_file = 0

            for dbg_col in debug_sample_cols:
                norm_name = self._normalize_sample_name_for_matching(str(dbg_col))
                final_col = final_sample_norm_to_col.get(norm_name)
                if final_col is None:
                    continue

                dbg_idx = [debug_uid_to_idx[uid] for uid in shared_uids]
                fin_idx = [final_uid_to_idx[uid] for uid in shared_uids]

                dbg_vals = df_debug.loc[dbg_idx, dbg_col]
                dbg_as_str = dbg_vals.astype(str).str.strip().str.lower()

                missing_mask = (
                    dbg_vals.isna()
                    | dbg_as_str.isin({"", "nan", "na", "none", "null"})
                ).to_numpy()

                if not missing_mask.any():
                    continue

                rows_to_null = np.asarray(fin_idx)[missing_mask]
                out.loc[rows_to_null, final_col] = np.nan

                restored_this_file += int(missing_mask.sum())
                total_restored += int(missing_mask.sum())

            print(
                f"[MissingRestore] Processed {debug_path.name}: "
                f"{len(shared_uids)} shared features, {restored_this_file} values restored.",
                flush=True
            )

        print(f"[MissingRestore] Total restored missing values: {total_restored}", flush=True)
        return out

    
    def prepare_statistical_datasets(self, allowed_groups=None, exclude_qc=False, output_override: Path = None):
        """Generate CSVs (+ transposed) under /statistics for:
           - Annotated (normalized): With_QCs / No_QCs / HighConf_* variants
           - Unknowns (normalized): With_QCs / No_QCs (if Unknowns file exists)
           - Annotated (pre-normalization): BeforeNorm_With_QCs / BeforeNorm_No_QCs (if file exists)
        """
        try:
            ann_path = self.output_folder / "Final_Annotated.csv"
            grp_path = self.output_folder / "sample_groups.csv"
            semi_ann_path = self.output_folder / "Final_Annotated_semi_quant.csv"

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
            df_raw_clean["Sample"] = df_raw_clean["Sample"].astype(str).str.strip()
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
                return cols

            def _save_with_T(df, meta_cols, samples, base_name):
                path = stats_dir / base_name
                df_to_save = df[meta_cols + samples]
                df_to_save.to_csv(path, index=False, encoding="utf-8-sig")
                transposed = df_to_save.set_index("UniqueID").transpose()
                transposed.index.name = "UniqueID"
                (stats_dir / f"{path.stem}_T.csv").write_text(transposed.to_csv(index=True), encoding="utf-8")

            # ---------- Annotated (normalized) ----------
            df_ann = pd.read_csv(ann_path, low_memory=False)
            df_ann_with_missing = self._restore_missing_values_from_debug(df_ann)
            meta_ann = _detect_meta(df_ann)
            samples_ann = _detect_samples(df_ann, meta_ann)
            meta_ann_missing = _detect_meta(df_ann_with_missing)
            samples_ann_missing = _detect_samples(df_ann_with_missing, meta_ann_missing)

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
                    
            # POS-only / NEG-only datasets with restored missingness
            samples_pos_missing = [s for s in samples_ann_missing if s.startswith("P_")]
            samples_neg_missing = [s for s in samples_ann_missing if s.startswith("N_")]

            if samples_pos_missing:
                _save_with_T(df_ann_with_missing, meta_ann_missing, samples_pos_missing, "POS_Final_Annotated_with_missing.csv")
                samples_pos_missing_noqc = [s for s in samples_pos_missing if s not in qc_set_full]
                if samples_pos_missing_noqc:
                    _save_with_T(df_ann_with_missing, meta_ann_missing, samples_pos_missing_noqc, "POS_Final_Annotated_with_missing_Without_QCs.csv")

            if samples_neg_missing:
                _save_with_T(df_ann_with_missing, meta_ann_missing, samples_neg_missing, "NEG_Final_Annotated_with_missing.csv")
                samples_neg_missing_noqc = [s for s in samples_neg_missing if s not in qc_set_full]
                if samples_neg_missing_noqc:
                    _save_with_T(df_ann_with_missing, meta_ann_missing, samples_neg_missing_noqc, "NEG_Final_Annotated_with_missing_Without_QCs.csv")

            if not samples_ann:
                messagebox.showwarning("No Sample Columns", "No numeric sample intensity columns in Final_Annotated.csv"); return

            if (allowed_groups is not None or exclude_qc):
                selected_group_samples = set(df_groups["Sample"].dropna().astype(str).str.strip())
                selected_group_samples_norm = {self._normalize_sample_name_for_matching(s) for s in selected_group_samples}
            else:
                selected_group_samples_norm = {self._normalize_sample_name_for_matching(s) for s in samples_ann}

            qc_set_full_norm = {self._normalize_sample_name_for_matching(s) for s in qc_set_full}

            # With_QCs = selected non-QC samples PLUS all QCs from the full group file
            samples_ann_with_qc = [
                c for c in samples_ann
                if (self._normalize_sample_name_for_matching(c) in selected_group_samples_norm)
                or (self._normalize_sample_name_for_matching(c) in qc_set_full_norm)
            ]
            if not samples_ann_with_qc:
                messagebox.showwarning("No Samples", "No valid samples found for Annotated (With_QCs).")
            else:
                _save_with_T(df_ann, meta_ann, samples_ann_with_qc, "Final_Annotated.csv")

            # Without_QCs = selected non-QC samples only
            samples_ann_without_qc = [
                c for c in samples_ann
                if (self._normalize_sample_name_for_matching(c) in selected_group_samples_norm)
                and (self._normalize_sample_name_for_matching(c) not in qc_set_full_norm)
            ]
            
            if samples_ann_without_qc:
                _save_with_T(df_ann, meta_ann, samples_ann_without_qc, "Final_Annotated_Without_QCs.csv")
            
            else:
                messagebox.showwarning("No Non-QC Samples", "No non-QC samples selected for Annotated (Without_QCs).")
                
            samples_ann_with_qc_missing = [
                c for c in samples_ann_missing
                if (self._normalize_sample_name_for_matching(c) in selected_group_samples_norm)
                or (self._normalize_sample_name_for_matching(c) in qc_set_full_norm)
            ]
            samples_ann_without_qc_missing = [
                c for c in samples_ann_missing
                if (self._normalize_sample_name_for_matching(c) in selected_group_samples_norm)
                and (self._normalize_sample_name_for_matching(c) not in qc_set_full_norm)
            ]

            if samples_ann_with_qc_missing:
                _save_with_T(df_ann_with_missing, meta_ann_missing, samples_ann_with_qc_missing, "Final_Annotated_with_missing.csv")

            if samples_ann_without_qc_missing:
                _save_with_T(df_ann_with_missing, meta_ann_missing, samples_ann_without_qc_missing, "Final_Annotated_with_missing_Without_QCs.csv")
                
            # ---------- Annotated semi-quant (normalized, optional) ----------
            if semi_ann_path.exists():
                df_ann_semi = pd.read_csv(semi_ann_path, low_memory=False)
                df_ann_semi_with_missing = self._restore_missing_values_from_debug(df_ann_semi)

                meta_ann_semi = _detect_meta(df_ann_semi)
                samples_ann_semi = _detect_samples(df_ann_semi, meta_ann_semi)

                meta_ann_semi_missing = _detect_meta(df_ann_semi_with_missing)
                samples_ann_semi_missing = _detect_samples(df_ann_semi_with_missing, meta_ann_semi_missing)

                # POS-only semi-quant
                samples_semi_pos = [s for s in samples_ann_semi if s.startswith("P_")]
                samples_semi_neg = [s for s in samples_ann_semi if s.startswith("N_")]

                if samples_semi_pos:
                    _save_with_T(df_ann_semi, meta_ann_semi, samples_semi_pos, "POS_Final_Annotated_semi_quant.csv")
                    samples_semi_pos_noqc = [s for s in samples_semi_pos if s not in qc_set_full]
                    if samples_semi_pos_noqc:
                        _save_with_T(df_ann_semi, meta_ann_semi, samples_semi_pos_noqc, "POS_Final_Annotated_semi_quant_Without_QCs.csv")

                if samples_semi_neg:
                    _save_with_T(df_ann_semi, meta_ann_semi, samples_semi_neg, "NEG_Final_Annotated_semi_quant.csv")
                    samples_semi_neg_noqc = [s for s in samples_semi_neg if s not in qc_set_full]
                    if samples_semi_neg_noqc:
                        _save_with_T(df_ann_semi, meta_ann_semi, samples_semi_neg_noqc, "NEG_Final_Annotated_semi_quant_Without_QCs.csv")

                # POS-only / NEG-only semi-quant with restored missingness
                samples_semi_pos_missing = [s for s in samples_ann_semi_missing if s.startswith("P_")]
                samples_semi_neg_missing = [s for s in samples_ann_semi_missing if s.startswith("N_")]

                if samples_semi_pos_missing:
                    _save_with_T(
                        df_ann_semi_with_missing,
                        meta_ann_semi_missing,
                        samples_semi_pos_missing,
                        "POS_Final_Annotated_semi_quant_with_missing.csv"
                    )
                    samples_semi_pos_missing_noqc = [s for s in samples_semi_pos_missing if s not in qc_set_full]
                    if samples_semi_pos_missing_noqc:
                        _save_with_T(
                            df_ann_semi_with_missing,
                            meta_ann_semi_missing,
                            samples_semi_pos_missing_noqc,
                            "POS_Final_Annotated_semi_quant_with_missing_Without_QCs.csv"
                        )

                if samples_semi_neg_missing:
                    _save_with_T(
                        df_ann_semi_with_missing,
                        meta_ann_semi_missing,
                        samples_semi_neg_missing,
                        "NEG_Final_Annotated_semi_quant_with_missing.csv"
                    )
                    samples_semi_neg_missing_noqc = [s for s in samples_semi_neg_missing if s not in qc_set_full]
                    if samples_semi_neg_missing_noqc:
                        _save_with_T(
                            df_ann_semi_with_missing,
                            meta_ann_semi_missing,
                            samples_semi_neg_missing_noqc,
                            "NEG_Final_Annotated_semi_quant_with_missing_Without_QCs.csv"
                        )

                if (allowed_groups is not None or exclude_qc):
                    selected_group_samples_semi = set(df_groups["Sample"].dropna().astype(str).str.strip())
                    selected_group_samples_semi_norm = {
                        self._normalize_sample_name_for_matching(s) for s in selected_group_samples_semi
                    }
                else:
                    selected_group_samples_semi_norm = {
                        self._normalize_sample_name_for_matching(s) for s in samples_ann_semi
                    }

                samples_ann_semi_with_qc = [
                    c for c in samples_ann_semi
                    if (self._normalize_sample_name_for_matching(c) in selected_group_samples_semi_norm)
                    or (self._normalize_sample_name_for_matching(c) in qc_set_full_norm)
                ]
                samples_ann_semi_without_qc = [
                    c for c in samples_ann_semi
                    if (self._normalize_sample_name_for_matching(c) in selected_group_samples_semi_norm)
                    and (self._normalize_sample_name_for_matching(c) not in qc_set_full_norm)
                ]

                if samples_ann_semi_with_qc:
                    _save_with_T(df_ann_semi, meta_ann_semi, samples_ann_semi_with_qc, "Final_Annotated_semi_quant.csv")

                if samples_ann_semi_without_qc:
                    _save_with_T(
                        df_ann_semi,
                        meta_ann_semi,
                        samples_ann_semi_without_qc,
                        "Final_Annotated_semi_quant_Without_QCs.csv"
                    )

                samples_ann_semi_with_qc_missing = [
                    c for c in samples_ann_semi_missing
                    if (self._normalize_sample_name_for_matching(c) in selected_group_samples_semi_norm)
                    or (self._normalize_sample_name_for_matching(c) in qc_set_full_norm)
                ]
                samples_ann_semi_without_qc_missing = [
                    c for c in samples_ann_semi_missing
                    if (self._normalize_sample_name_for_matching(c) in selected_group_samples_semi_norm)
                    and (self._normalize_sample_name_for_matching(c) not in qc_set_full_norm)
                ]

                if samples_ann_semi_with_qc_missing:
                    _save_with_T(
                        df_ann_semi_with_missing,
                        meta_ann_semi_missing,
                        samples_ann_semi_with_qc_missing,
                        "Final_Annotated_semi_quant_with_missing.csv"
                    )

                if samples_ann_semi_without_qc_missing:
                    _save_with_T(
                        df_ann_semi_with_missing,
                        meta_ann_semi_missing,
                        samples_ann_semi_without_qc_missing,
                        "Final_Annotated_semi_quant_with_missing_Without_QCs.csv"
                    )

                tier_col_semi = next((c for c in df_ann_semi.columns if c.strip().lower() == "annotation tier"), None)
                if tier_col_semi:
                    high_mask_semi = df_ann_semi[tier_col_semi].fillna("").str.lower() == "high confidence"
                    df_high_semi = df_ann_semi.loc[high_mask_semi].copy()

                    high_mask_semi_missing = (
                        df_ann_semi_with_missing[tier_col_semi].fillna("").str.lower() == "high confidence"
                    )
                    df_high_semi_missing = df_ann_semi_with_missing.loc[high_mask_semi_missing].copy()
                else:
                    df_high_semi = df_ann_semi.copy()
                    df_high_semi_missing = df_ann_semi_with_missing.copy()

                if samples_ann_semi_with_qc:
                    _save_with_T(
                        df_high_semi,
                        meta_ann_semi,
                        samples_ann_semi_with_qc,
                        "Final_Annotated_semi_quant_HighConf.csv"
                    )

                if samples_ann_semi_without_qc:
                    _save_with_T(
                        df_high_semi,
                        meta_ann_semi,
                        samples_ann_semi_without_qc,
                        "Final_Annotated_semi_quant_Without_QCs_HighConf.csv"
                    )

                if samples_ann_semi_with_qc_missing:
                    _save_with_T(
                        df_high_semi_missing,
                        meta_ann_semi_missing,
                        samples_ann_semi_with_qc_missing,
                        "Final_Annotated_semi_quant_with_missing_HighConf.csv"
                    )

                if samples_ann_semi_without_qc_missing:
                    _save_with_T(
                        df_high_semi_missing,
                        meta_ann_semi_missing,
                        samples_ann_semi_without_qc_missing,
                        "Final_Annotated_semi_quant_with_missing_Without_QCs_HighConf.csv"
                    )


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
                
            if tier_col:
                high_mask_missing = df_ann_with_missing[tier_col].fillna("").str.lower() == "high confidence"
                df_high_missing = df_ann_with_missing.loc[high_mask_missing].copy()
            else:
                df_high_missing = df_ann_with_missing.copy()

            if samples_ann_with_qc_missing:
                _save_with_T(df_high_missing, meta_ann_missing, samples_ann_with_qc_missing, "Final_Annotated_with_missing_HighConf.csv")
            if samples_ann_without_qc_missing:
                _save_with_T(df_high_missing, meta_ann_missing, samples_ann_without_qc_missing, "Final_Annotated_with_missing_Without_QCs_HighConf.csv")


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

                if (allowed_groups is not None or exclude_qc):
                    selected_group_samples_unk = set(df_groups["Sample"].dropna().astype(str).str.strip())
                    selected_group_samples_unk_norm = {self._normalize_sample_name_for_matching(s) for s in selected_group_samples_unk}
                else:
                    selected_group_samples_unk_norm = {self._normalize_sample_name_for_matching(s) for s in samples_unk}

                samples_unk_with_qc = [
                    c for c in samples_unk
                    if (self._normalize_sample_name_for_matching(c) in selected_group_samples_unk_norm)
                    or (self._normalize_sample_name_for_matching(c) in qc_set_full_norm)
                ]
                samples_unk_without_qc = [
                    c for c in samples_unk
                    if (self._normalize_sample_name_for_matching(c) in selected_group_samples_unk_norm)
                    and (self._normalize_sample_name_for_matching(c) not in qc_set_full_norm)
                ]
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
                if (allowed_groups is not None or exclude_qc):
                    selected_group_samples_bfn = set(df_groups["Sample"].dropna().astype(str).str.strip())
                    selected_group_samples_bfn_norm = {self._normalize_sample_name_for_matching(s) for s in selected_group_samples_bfn}
                else:
                    selected_group_samples_bfn_norm = {self._normalize_sample_name_for_matching(s) for s in samples_bfn}

                samples_bfn_with_qc = [
                    c for c in samples_bfn
                    if (self._normalize_sample_name_for_matching(c) in selected_group_samples_bfn_norm)
                    or (self._normalize_sample_name_for_matching(c) in qc_set_full_norm)
                ]
                samples_bfn_without_qc = [
                    c for c in samples_bfn
                    if (self._normalize_sample_name_for_matching(c) in selected_group_samples_bfn_norm)
                    and (self._normalize_sample_name_for_matching(c) not in qc_set_full_norm)
                ]
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
                "• Annotated semi-quant (+ variants) if available\n"
                "• Annotated semi-quant (+ variants, including with_missing) if available\n"
                "• Unknowns (+ Without_QCs) if available\n"
                "• BeforeNorm (+ Without_QCs) if available\n"
            )
            self.summary_label.config(text=f"✅ Statistical datasets (and transposed versions) saved to {stats_dir}")

            # Enable buttons
            for btn in (self.pca_button, self.plsda_button, self.heatmap_button, self.selected_heatmap_button, self.selected_heatmap_settings_button, self.volcano_button, self.boxplots_button, self.violin_button,
                        self.correlations_button, self.classdist_button, self.summint_button, self.classviolinbox_button, self.classcarbons_button,
                        self.enrichment_button, self.ratio_button, self.ratio_settings_button, self.upset_button, self.advanceddiff_button):
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
        df["Sample"] = df["Sample"].astype(str).str.strip()
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
