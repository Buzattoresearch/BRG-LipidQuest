# GUI/view_statistics.py
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import pandas as pd
import os, re, traceback
import threading

# If you use these analysis functions, keep the imports
from Stats.pca_analysis import run_pca
from Stats.plsda_analysis import run_plsda
from Stats.heatmap_analysis import run_heatmap
from Stats.volcano_analysis import run_volcano


class StatisticsPage(tk.Toplevel):
    """
    Statistics GUI
    Loads final processed files and enables statistical tools (PCA, PLS-DA, Heatmap)
    if required files are available.
    """

    def __init__(self, parent, output_folder: Path, sample_type):
        super().__init__(parent)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.title("Statistics")
        self.configure(bg="white")
        self.geometry("1000x700")

        self.output_folder = Path(output_folder)
        self.parent = parent
        self.df_annotated = None
        self.df_unknowns = None
        self.df_groups = None
        self.sample_type = sample_type

        # Try to reuse the same ttk style names as the start page (if defined there)
        self._configure_local_style_if_needed()

        # --- Load data ---
        self.missing_files = self._load_data_files()

        # === Header ===
        header = tk.Frame(self, bg="white")
        header.pack(fill="x", pady=(14, 8), padx=24)

        ttk.Label(
            header,
            text="Statistics",
            style="Header.TLabel"
        ).pack(side="left")

        ttk.Label(
            self,
            text=f"Output folder: {self.output_folder}",
            style="Subtle.TLabel"
        ).pack(anchor="w", padx=24, pady=(0, 12))

        # === Summary ===
        summary_text = self._make_summary_text()
        self.summary_label = ttk.Label(
            self, text=summary_text, style="Body.TLabel", justify="left"
        )
        self.summary_label.pack(fill="x", padx=24, pady=(0, 14))

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=24, pady=(0, 16))

        # === Prepare datasets ===
        prepare_frame = tk.Frame(self, bg="white")
        prepare_frame.pack(pady=(4, 18))

        self.prepare_btn = ttk.Button(
            prepare_frame,
            text="Prepare Statistical Datasets",
            command=self.prepare_statistical_datasets,
            width=32,
            style="Accent.TButton"  # will fall back if not defined
        )
        self.prepare_btn.pack()

        # === Tools ===
        tools = tk.Frame(self, bg="white")
        tools.pack(pady=(10, 28), padx=24, fill="x")

        ttk.Label(
            tools, text="Available Statistical Tools", style="Section.TLabel"
        ).grid(row=0, column=0, columnspan=5, pady=(0, 12), sticky="w")

        self.pca_button = ttk.Button(tools, text="Run PCA", width=25, command=self.run_pca)
        self.plsda_button = ttk.Button(tools, text="Run PLS-DA", width=25, command=self.run_plsda)
        self.heatmap_button = ttk.Button(tools, text="Run Clustered Heatmap", width=25, command=self.run_heatmap)
        self.volcano_button = ttk.Button(tools, text="Run Volcano", width=25, command=self.run_volcano)

        self.pca_button.grid(row=1, column=0, padx=8, pady=6)
        self.plsda_button.grid(row=1, column=1, padx=8, pady=6)
        self.heatmap_button.grid(row=1, column=2, padx=8, pady=6)
        self.volcano_button.grid(row=1, column=3, padx=8, pady=6)

        # Disable analysis buttons if required files are missing
        if self.missing_files:
            for btn in (self.pca_button, self.plsda_button, self.heatmap_button, self.volcano_button):
                btn.config(state="disabled")
            self._add_tooltip(tools, f"Missing files: {', '.join(self.missing_files)}")

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=24, pady=(0, 16))

        # === Navigation ===
        nav = tk.Frame(self, bg="white")
        nav.pack(pady=(4, 16))

        ttk.Button(
            nav,
            text="← Return to Processing",
            command=self.return_to_processing,
            width=22
        ).pack(side="left", padx=8)

        ttk.Button(
            nav,
            text="Quit",
            command=self.quit_app,
            width=14
        ).pack(side="left", padx=8)

    # ---------- style helpers ----------
    def _configure_local_style_if_needed(self):
        """If the main app didn't set these styles, define light defaults so this page still looks nice standalone."""
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        # --- Define Accent.TButton if missing ---
        if "Accent.TButton" not in style.element_names():
            style.configure(
                "Accent.TButton",
                background="#0078D7",
                foreground="white",
                font=("Segoe UI", 9, "bold"),
                padding=6,
            )
            style.map(
                "Accent.TButton",
                background=[("active", "#005A9E"), ("disabled", "#d0d0d0")],
                foreground=[("disabled", "#888888")],
            )

        # --- Label Styles ---
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), background="white")
        style.configure("Section.TLabel", font=("Segoe UI", 11, "bold"), background="white")
        style.configure("Body.TLabel", font=("Segoe UI", 10), background="white")
        style.configure("Subtle.TLabel", font=("Segoe UI", 9, "italic"), foreground="#666", background="white")


    # ==========================================================
    # DATA LOADING
    # ==========================================================
    def _load_data_files(self):
        """Load annotated, unknown, and group files if available. Return list of missing files."""
        required_files = {
            "Final_Annotated.csv": "df_annotated",
            "Final_Unknowns.csv": "df_unknowns",
            "sample_groups.csv": "df_groups"
        }

        missing = []
        for filename, attr_name in required_files.items():
            fpath = self.output_folder / filename
            if fpath.exists():
                try:
                    setattr(self, attr_name, pd.read_csv(fpath))
                except Exception as e:
                    messagebox.showwarning("File Error", f"Failed to read {filename}:\n{e}")
                    missing.append(filename)
            else:
                missing.append(filename)

        return missing

    def _make_summary_text(self):
        """Generate a summary text block."""
        if self.missing_files:
            return (
                f"⚠ Some required files are missing:\n"
                f"  - {', '.join(self.missing_files)}\n\n"
                f"Please run the processing pipeline completely before proceeding."
            )

        return (
            f"Loaded {len(self.df_annotated)} annotated compounds\n"
            f"Loaded {len(self.df_unknowns)} unknown features\n"
            f"Loaded {len(self.df_groups)} sample group assignments"
        )

    # ==========================================================
    # TOOLTIP
    # ==========================================================
    def _add_tooltip(self, widget, text):
        """Attach a tooltip message to a widget (hover anywhere on the frame)."""
        tooltip = tk.Label(self, text=text, bg="lightyellow", fg="black",
                           relief="solid", bd=1, wraplength=300)
        tooltip.place_forget()

        def on_enter(event):
            x = event.x_root - self.winfo_rootx() + 12
            y = event.y_root - self.winfo_rooty() + 18
            tooltip.place(x=x, y=y)

        def on_leave(event):
            tooltip.place_forget()

        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    # ==========================================================
    # ANALYSIS LAUNCHERS
    # ==========================================================
    def run_pca(self):
        threading.Thread(target=self._run_analysis, args=("PCA",), daemon=True).start()

    def run_plsda(self):
        threading.Thread(target=self._run_analysis, args=("PLS-DA",), daemon=True).start()

    def run_heatmap(self):
        threading.Thread(target=self._run_analysis, args=("Heatmap",), daemon=True).start()
    
    def run_volcano(self):
        threading.Thread(target=self._run_analysis, args=("Volcano",), daemon=True).start()

    def _run_analysis(self, analysis_type):
        """Unified analysis runner for PCA, PLS-DA, Heatmap, Volcano with variant-specific folders + toast feedback."""
        try:
            stats_dir = self.output_folder / "statistics"

            # Prefer cleaned sample groups if present
            cleaned_group_file = stats_dir / "sample_groups_cleaned.csv"
            group_file = cleaned_group_file if cleaned_group_file.exists() else (self.output_folder / "sample_groups.csv")
            if not stats_dir.exists():
                self._toast("⚠ Prepare datasets first")
                return

            if not group_file.exists():
                group_file = None  # analyses will run without QC info

            datasets = [
                ("Final_Annotated.csv", "With_QCs"),
                ("Final_Annotated_Without_QCs.csv", "Without_QCs"),
                ("Final_Annotated_HighConf.csv", "HighConf_With_QCs"),
                ("Final_Annotated_Without_QCs_HighConf.csv", "HighConf_Without_QCs"),
            ]

            self._toast(f"Running {analysis_type}...")

            for fname, label in datasets:
                fpath = stats_dir / fname
                if not fpath.exists():
                    continue

                # Skip PLS-DA on datasets that include QCs
                if analysis_type == "PLS-DA" and "Without_QCs" not in label:
                    continue
                
                # Skip Volcano on datasets that include QCs
                if analysis_type == "Volcano" and "Without_QCs" not in label:
                    continue
                
                # Skip Heatmap on datasets that include QCs
                if analysis_type == "Heatmap" and "Without_QCs" not in label:
                    continue

                subfolder = stats_dir / analysis_type / label
                subfolder.mkdir(parents=True, exist_ok=True)

                try:
                    if analysis_type == "PCA":
                        run_pca(fpath, group_file, subfolder)
                    elif analysis_type == "PLS-DA":
                        run_plsda(fpath, group_file, subfolder)
                    elif analysis_type == "Heatmap":
                        run_heatmap(fpath, group_file, subfolder)
                    elif analysis_type == "Volcano":
                        run_volcano(fpath, group_file, subfolder, sample_type = self.sample_type)
                except Exception:
                    print(traceback.format_exc(), flush = True)

            self._toast(f"{analysis_type} completed ✓")

        except Exception:
            print(traceback.format_exc(), flush = True)
            self._toast(f"{analysis_type} failed ❌")

    # ==========================================================
    # NAVIGATION / CLOSE
    # ==========================================================
    def return_to_processing(self):
        self.destroy()
        self.parent.deiconify()

    def _on_close(self):
        try:
            self.destroy()
            self.parent.destroy()
        except Exception:
            os._exit(0)

    # ==========================================================
    # SAMPLE NAME CLEANUP (used when preparing datasets)
    # ==========================================================
    def _clean_sample_name(self, name: str) -> str:
        """Clean sample names by removing polarity tags and run suffixes."""
        if not isinstance(name, str):
            return name

        cleaned = name
        cleaned = re.sub(r"\[?POS\]?|\[?NEG\]?", "", cleaned, flags=re.IGNORECASE)   # remove [POS]/[NEG]
        cleaned = re.sub(r"^(P_|N_)", "", cleaned)                                   # remove P_/N_ prefix
        cleaned = re.split(r"_P[12]", cleaned)[0]                                    # cut at _P1/_P2
        cleaned = re.sub(r"_[0-9]+(_[0-9]+)*$", "", cleaned)                         # drop trailing run ids
        cleaned = re.sub(r"[_\-]+$", "", cleaned)
        cleaned = re.sub(r"[_\-]{2,}", "_", cleaned)
        return cleaned.strip("_- ")

    # ==========================================================
    # PREPARE STATISTICAL DATASETS  (RESTORED)
    # ==========================================================
    def prepare_statistical_datasets(self):
        """Generate QC-filtered / HighConfidence-filtered annotated CSVs (+ transposed versions) under /statistics."""
        try:
            annotated_path = self.output_folder / "Final_Annotated.csv"
            group_path = self.output_folder / "sample_groups.csv"

            if not annotated_path.exists():
                messagebox.showwarning("Missing File", "Final_Annotated.csv not found in output folder.")
                return
            if not group_path.exists():
                messagebox.showwarning("Missing File", "sample_groups.csv not found in output folder.")
                return

            df = pd.read_csv(annotated_path)
            df_groups = pd.read_csv(group_path)

            # Clean and save sample_groups just for statistics
            try:
                df_groups_cleaned = df_groups.copy()
                if "Sample" in df_groups_cleaned.columns:
                    df_groups_cleaned["Sample"] = df_groups_cleaned["Sample"].apply(self._clean_sample_name)
                    df_groups_cleaned = df_groups_cleaned.drop_duplicates(subset=["Sample"], keep="first").reset_index(drop=True)

                stats_dir = self.output_folder / "statistics"
                stats_dir.mkdir(parents=True, exist_ok=True)
                (stats_dir / "sample_groups_cleaned.csv").write_text(
                    df_groups_cleaned.to_csv(index=False, encoding="utf-8-sig"),
                    encoding="utf-8-sig"
                )
                # After saving above, re-read to avoid writing whole CSV as text accidentally if using write_text
                df_groups_cleaned.to_csv(stats_dir / "sample_groups_cleaned.csv", index=False, encoding="utf-8-sig")
                df_groups = df_groups_cleaned
            except Exception:
                print("[Warning] Could not create cleaned sample_groups file:", flush = True)
                print(traceback.format_exc(), flush = True)

            # Identify QC samples
            df_groups["Group"] = df_groups["Group"].astype(str).str.strip().str.upper()
            qc_samples = df_groups.loc[df_groups["Group"] == "QC", "Sample"].tolist()

            # Keep key metadata columns if present
            meta_keep = [
                "UniqueID", "RT (min)", "m/z", "Polarity", "Annotation",
                "Annotation Type Headgroup", "Lipid Class", "Δm/z (mDa)", "Δm/z (ppm)",
                "MS/MS score", "Annotation tier", "mSigma", "Molecular Formula", "Plasmenyl?",
                "Number of carbons in fatty acyls", "Double bond equivalents", "Chain type",
                "PUFA?", "Modifications", "# of modifications", "Oxidized?"
            ]
            meta_cols = [c for c in meta_keep if c in df.columns]
            
            # --- Detect sample columns dynamically ---
            sample_cols = [
                c for c in df.columns
                if c not in meta_cols
                and "rsd" not in c.lower()              # exclude any RSD-related columns
                and pd.api.types.is_numeric_dtype(df[c]) # must be numeric
            ]

            if not sample_cols:
                messagebox.showwarning(
                    "No Sample Columns Found",
                    "No numeric sample intensity columns were detected (after removing RSD columns).\n"
                    "Please check your Final_Annotated.csv."
                )

            # Clean sample columns
            cleaned_names = {col: self._clean_sample_name(col) for col in sample_cols}
            df.rename(columns=cleaned_names, inplace=True)
            sample_cols = list(cleaned_names.values())

            stats_dir = self.output_folder / "statistics"
            stats_dir.mkdir(parents=True, exist_ok=True)

            def save_with_T(data: pd.DataFrame, filename: str):
                path = stats_dir / filename
                data.to_csv(path, index=False, encoding="utf-8-sig")

                # Transpose but keep "UniqueID" as the first column label
                transposed = data.set_index("UniqueID").transpose()
                transposed.index.name = "UniqueID"  # keep correct label for the first column

                # Write the corrected transposed file
                transposed.to_csv(
                    stats_dir / f"{path.stem}_T.csv",
                    encoding="utf-8-sig"
                )


            # Baseline
            base = df[meta_cols + sample_cols]
            save_with_T(base, "Final_Annotated.csv")

            # Without QCs
            samples_no_qc = [c for c in sample_cols if c not in qc_samples]
            save_with_T(base[meta_cols + samples_no_qc], "Final_Annotated_Without_QCs.csv")

            # High confidence
            tier_col = next((c for c in df.columns if c.strip().lower() == "annotation tier"), None)
            if tier_col:
                high = base[df[tier_col].fillna("").str.lower() == "high confidence"]
            else:
                messagebox.showwarning("Missing Column", "No 'Annotation tier' column found; saving all as HighConf.")
                high = base.copy()
            save_with_T(high, "Final_Annotated_HighConf.csv")

            # HighConf without QCs
            save_with_T(high[meta_cols + samples_no_qc], "Final_Annotated_Without_QCs_HighConf.csv")

            # Log + UI feedback
            with open(stats_dir / "statistics_log.txt", "a", encoding="utf-8") as log:
                log.write(f"\n[{pd.Timestamp.now()}] Generated filtered + transposed statistical datasets\n")
                for f in [
                    "Final_Annotated.csv",
                    "Final_Annotated_Without_QCs.csv",
                    "Final_Annotated_HighConf.csv",
                    "Final_Annotated_Without_QCs_HighConf.csv",
                ]:
                    log.write(f"  • {f}\n  • {f.replace('.csv', '_T.csv')}\n")

            messagebox.showinfo(
                "Statistics Files Created",
                f"Statistical datasets created successfully in:\n\n{stats_dir}\n\n"
                "• Final_Annotated.csv (+ _T.csv)\n"
                "• Final_Annotated_Without_QCs.csv (+ _T.csv)\n"
                "• Final_Annotated_HighConf.csv (+ _T.csv)\n"
                "• Final_Annotated_Without_QCs_HighConf.csv (+ _T.csv)"
            )
            self.summary_label.config(text=f"✅ Statistical datasets (and transposed versions) saved to {stats_dir}")

            # Enable buttons now that datasets exist
            for btn in (self.pca_button, self.plsda_button, self.heatmap_button, self.volcano_button):
                btn.config(state="normal")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to prepare datasets:\n{e}")

    # ==========================================================
    # QUIT + TOAST
    # ==========================================================
    def quit_app(self):
        try:
            self.destroy()
            self.parent.destroy()
        except Exception:
            pass
        finally:
            os._exit(0)

    def _toast(self, text):
        """popup message near the window center.
        - If text includes 'done', 'completed', or 'success', it stays until click or next toast.
        """
        # Destroy any previous toast before creating a new one
        if hasattr(self, "_current_toast") and self._current_toast.winfo_exists():
            self._current_toast.destroy()

        toast = tk.Toplevel(self)
        toast.overrideredirect(True)
        toast.configure(bg="#333333")
        self._current_toast = toast  # keep reference for later

        label = tk.Label(
            toast, text=text, fg="white", bg="#333333",
            font=("Segoe UI", 10, "bold"), padx=15, pady=8
        )
        label.pack()

        self.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() // 2) - 130  # toast position: x coordinate
        y = self.winfo_y() + (self.winfo_height() // 2) + 100  # toast position: y coordinate
        toast.geometry(f"+{x}+{y}")

        # If it’s a final “done” message, keep it until click
        if any(kw in text.lower() for kw in ["done", "completed", "success", "first"]):
            def close_on_click(_event=None):
                if toast.winfo_exists():
                    toast.destroy()
            toast.bind("<Button-1>", close_on_click)

# Optional: quick manual test
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    StatisticsPage(root, Path.cwd())
    root.mainloop()
