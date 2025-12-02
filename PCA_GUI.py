# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 16:22:05 2025

@author: User
"""

# -*- coding: utf-8 -*-
"""
PCA Analysis GUI with Export Functionality
Created on Wed Oct 15 09:10:37 2025
@author: User
"""

import sys
import os
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np

# Import PCA functions
try:
    from PCA_claude import (extract_data, perform_pca, plot_loadings, 
                              test_pc_significance, multi_stat, 
                              excel_to_pickle, crop_data, multi_loadings,
                              test_pc_significance_grouped_means, 
                              multi_stat_grouped_means, export_data,
                              export_stats)
except ImportError:
    print("Warning: Could not import PCA functions.")

class PCAAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PCA Analysis Tool")
        self.root.geometry("900x750")
        self.root.resizable(True, True)
        
        self.file_path = tk.StringVar()
        self.pickle_path = tk.StringVar()
        self.data = None
        self.categoricals = None
        self.pca_results = None
        self.stats_results = None  # Store stats results for export
        self.grouped_stats_results = None  # Store grouped stats results for export
        
        self.setup_ui()
        
    def setup_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.tab_load = ttk.Frame(notebook)
        notebook.add(self.tab_load, text='Step 1: Load Data')
        self.setup_load_tab()
        
        self.tab_crop = ttk.Frame(notebook)
        notebook.add(self.tab_crop, text='Step 1.5: Crop Data')
        self.setup_crop_tab()
        
        self.tab_pca = ttk.Frame(notebook)
        notebook.add(self.tab_pca, text='Step 2: PCA Analysis')
        self.setup_pca_tab()
        
        self.tab_viz = ttk.Frame(notebook)
        notebook.add(self.tab_viz, text='Step 3: Visualizations')
        self.setup_viz_tab()
        
        self.tab_stats = ttk.Frame(notebook)
        notebook.add(self.tab_stats, text='Step 4: Statistics')
        self.setup_stats_tab()
        
        self.tab_grouped_stats = ttk.Frame(notebook)
        notebook.add(self.tab_grouped_stats, text='Step 5: Grouped Means')
        self.setup_grouped_stats_tab()
        
        self.tab_export = ttk.Frame(notebook)
        notebook.add(self.tab_export, text='Step 6: Export Results')
        self.setup_export_tab()
        
    def setup_load_tab(self):
        frame = ttk.LabelFrame(self.tab_load, text="Data Loading", padding=15)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text="Select Excel file:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Entry(frame, textvariable=self.file_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5)
        
        ttk.Button(frame, text="Convert Excel to Pickle", command=self.convert_excel).grid(row=1, column=0, columnspan=3, pady=10)
        
        ttk.Label(frame, text="Or select Pickle file:").grid(row=2, column=0, sticky='w', pady=5)
        ttk.Entry(frame, textvariable=self.pickle_path, width=50).grid(row=2, column=1, padx=5)
        ttk.Button(frame, text="Browse", command=self.browse_pickle).grid(row=2, column=2, padx=5)
        
        ttk.Button(frame, text="Load Data", command=self.load_data).grid(row=3, column=0, columnspan=3, pady=15)
        
        self.load_status = ttk.Label(frame, text="No data loaded", foreground="red")
        self.load_status.grid(row=4, column=0, columnspan=3, pady=10)
        
    def setup_crop_tab(self):
        frame = ttk.LabelFrame(self.tab_crop, text="Crop Data by Wavenumber Range", padding=15)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text="Optional: Crop data to a specific wavenumber range", foreground="gray").grid(row=0, column=0, columnspan=3, pady=5, sticky='w')
        
        ttk.Label(frame, text="Start Wavenumber:").grid(row=1, column=0, sticky='w', pady=5)
        self.crop_start = ttk.Entry(frame, width=15)
        self.crop_start.grid(row=1, column=1, sticky='w', padx=5)
        self.start_auto_label = ttk.Label(frame, text="(Auto: min)", foreground="gray")
        self.start_auto_label.grid(row=1, column=2, sticky='w', padx=5)
        
        ttk.Label(frame, text="End Wavenumber:").grid(row=2, column=0, sticky='w', pady=5)
        self.crop_end = ttk.Entry(frame, width=15)
        self.crop_end.grid(row=2, column=1, sticky='w', padx=5)
        self.end_auto_label = ttk.Label(frame, text="(Auto: max)", foreground="gray")
        self.end_auto_label.grid(row=2, column=2, sticky='w', padx=5)
        
        ttk.Separator(frame, orient='horizontal').grid(row=3, column=0, columnspan=3, sticky='ew', pady=10)
        
        self.crop_info = ttk.Label(frame, text="No data loaded", foreground="gray")
        self.crop_info.grid(row=4, column=0, columnspan=3, pady=5, sticky='w')
        
        ttk.Button(frame, text="Crop Data", command=self.crop_data_action).grid(row=5, column=0, columnspan=3, pady=15)
        
        self.crop_status = ttk.Label(frame, text="Ready", foreground="blue")
        self.crop_status.grid(row=6, column=0, columnspan=3, pady=10)
        
    def setup_pca_tab(self):
        frame = ttk.LabelFrame(self.tab_pca, text="PCA Settings", padding=15)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text="Number of Components:").grid(row=0, column=0, sticky='w', pady=5)
        self.n_components = ttk.Spinbox(frame, from_=2, to=50, width=10)
        self.n_components.set(10)
        self.n_components.grid(row=0, column=1, sticky='w', padx=5)
        
        self.scale_data = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Scale Data (Standardize)", variable=self.scale_data).grid(row=1, column=0, columnspan=2, pady=5, sticky='w')
        
        self.detect_outliers = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Detect Outliers", variable=self.detect_outliers).grid(row=2, column=0, columnspan=2, pady=5, sticky='w')
        
        ttk.Label(frame, text="Outlier Significance Level (α):").grid(row=3, column=0, sticky='w', pady=5)
        self.outlier_alpha = ttk.Combobox(frame, values=['0.01', '0.05', '0.1'], width=10, state='readonly')
        self.outlier_alpha.set('0.05')
        self.outlier_alpha.grid(row=3, column=1, sticky='w', padx=5)
        
        ttk.Label(frame, text="Outlier Components (None=all):").grid(row=4, column=0, sticky='w', pady=5)
        self.outlier_components = ttk.Entry(frame, width=10)
        self.outlier_components.grid(row=4, column=1, sticky='w', padx=5)
        
        ttk.Label(frame, text="Outlier Action:").grid(row=5, column=0, sticky='w', pady=5)
        self.outlier_action = ttk.Combobox(frame, values=['show', 'hide', 'remove'], width=10, state='readonly')
        self.outlier_action.set('show')
        self.outlier_action.grid(row=5, column=1, sticky='w', padx=5)
        
        ttk.Label(frame, text="Color by (optional):").grid(row=6, column=0, sticky='w', pady=5)
        self.color_by = ttk.Combobox(frame, width=20, state='readonly')
        self.color_by.grid(row=6, column=1, sticky='w', padx=5)
        
        ttk.Label(frame, text="X-axis PC:").grid(row=7, column=0, sticky='w', pady=5)
        self.pc_x = ttk.Spinbox(frame, from_=1, to=50, width=10)
        self.pc_x.set(1)
        self.pc_x.grid(row=7, column=1, sticky='w', padx=5)
        
        ttk.Label(frame, text="Y-axis PC:").grid(row=8, column=0, sticky='w', pady=5)
        self.pc_y = ttk.Spinbox(frame, from_=1, to=50, width=10)
        self.pc_y.set(2)
        self.pc_y.grid(row=8, column=1, sticky='w', padx=5)
        
        ttk.Button(frame, text="Run PCA Analysis", command=self.run_pca).grid(row=9, column=0, columnspan=2, pady=15)
        
        self.pca_status = ttk.Label(frame, text="Ready to analyze", foreground="blue")
        self.pca_status.grid(row=10, column=0, columnspan=2, pady=10)
        
    def setup_viz_tab(self):
        frame = ttk.LabelFrame(self.tab_viz, text="Visualization Options", padding=15)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text="Loadings - PC 1:").grid(row=0, column=0, sticky='w', pady=5)
        self.load_pc1 = ttk.Spinbox(frame, from_=1, to=50, width=10)
        self.load_pc1.set(1)
        self.load_pc1.grid(row=0, column=1, sticky='w', padx=5)
        
        ttk.Label(frame, text="Loadings - PC 2:").grid(row=1, column=0, sticky='w', pady=5)
        self.load_pc2 = ttk.Spinbox(frame, from_=1, to=50, width=10)
        self.load_pc2.set(2)
        self.load_pc2.grid(row=1, column=1, sticky='w', padx=5)
        
        ttk.Button(frame, text="Plot Loadings", command=self.plot_loadings_viz).grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Separator(frame, orient='horizontal').grid(row=3, column=0, columnspan=2, sticky='ew', pady=10)
        
        ttk.Label(frame, text="Max PC for Multiple Loadings:").grid(row=4, column=0, sticky='w', pady=5)
        self.max_load_pc = ttk.Spinbox(frame, from_=2, to=20, width=10)
        self.max_load_pc.set(5)
        self.max_load_pc.grid(row=4, column=1, sticky='w', padx=5)
        
        ttk.Button(frame, text="Plot Multiple Loadings", command=self.multi_loadings_viz).grid(row=5, column=0, columnspan=2, pady=10)
        
        self.viz_status = ttk.Label(frame, text="Ready to visualize", foreground="blue")
        self.viz_status.grid(row=6, column=0, columnspan=2, pady=10)
        
    def setup_stats_tab(self):
        frame = ttk.LabelFrame(self.tab_stats, text="Statistical Tests", padding=15)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text="Group by:").grid(row=0, column=0, sticky='w', pady=5)
        self.group_by = ttk.Combobox(frame, width=20, state='readonly')
        self.group_by.grid(row=0, column=1, sticky='w', padx=5)
        
        ttk.Label(frame, text="Principal Component:").grid(row=1, column=0, sticky='w', pady=5)
        self.stat_pc = ttk.Spinbox(frame, from_=1, to=50, width=10)
        self.stat_pc.set(1)
        self.stat_pc.grid(row=1, column=1, sticky='w', padx=5)
        
        ttk.Label(frame, text="Significance Level (α):").grid(row=2, column=0, sticky='w', pady=5)
        self.stat_alpha = ttk.Combobox(frame, values=['0.01', '0.05', '0.1'], width=10, state='readonly')
        self.stat_alpha.set('0.05')
        self.stat_alpha.grid(row=2, column=1, sticky='w', padx=5)
        
        ttk.Button(frame, text="Test Single PC", command=self.test_single_pc).grid(row=3, column=0, columnspan=2, pady=10)
        
        ttk.Separator(frame, orient='horizontal').grid(row=4, column=0, columnspan=2, sticky='ew', pady=10)
        
        ttk.Label(frame, text="Max PC for Multiple Tests:").grid(row=5, column=0, sticky='w', pady=5)
        self.max_stat_pc = ttk.Spinbox(frame, from_=2, to=20, width=10)
        self.max_stat_pc.set(10)
        self.max_stat_pc.grid(row=5, column=1, sticky='w', padx=5)
        
        ttk.Button(frame, text="Test Multiple PCs", command=self.test_multi_pc).grid(row=6, column=0, columnspan=2, pady=10)
        
        ttk.Separator(frame, orient='horizontal').grid(row=7, column=0, columnspan=2, sticky='ew', pady=10)
        
        ttk.Label(frame, text="Export Statistics:", font=('TkDefaultFont', 9, 'bold')).grid(row=8, column=0, columnspan=2, sticky='w', pady=5)
        ttk.Button(frame, text="Export Stats Results", command=self.export_stats_results).grid(row=9, column=0, columnspan=2, pady=5)
        
        self.stats_status = ttk.Label(frame, text="Ready to analyze", foreground="blue")
        self.stats_status.grid(row=10, column=0, columnspan=2, pady=10)
        
    def setup_grouped_stats_tab(self):
        frame = ttk.LabelFrame(self.tab_grouped_stats, text="Grouped Means Statistical Tests", padding=15)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text="Test PC differences across groups using means calculated per sub-group", foreground="gray", wraplength=400).grid(row=0, column=0, columnspan=2, pady=10, sticky='w')
        
        ttk.Label(frame, text="Compare across (Group by):").grid(row=1, column=0, sticky='w', pady=5)
        self.grouped_group_by = ttk.Combobox(frame, width=20, state='readonly')
        self.grouped_group_by.grid(row=1, column=1, sticky='w', padx=5)
        
        ttk.Label(frame, text="Average within (Mean by):").grid(row=2, column=0, sticky='w', pady=5)
        self.grouped_mean_by = ttk.Combobox(frame, width=20, state='readonly')
        self.grouped_mean_by.grid(row=2, column=1, sticky='w', padx=5)
        
        ttk.Label(frame, text="Principal Component:").grid(row=3, column=0, sticky='w', pady=5)
        self.grouped_stat_pc = ttk.Spinbox(frame, from_=1, to=50, width=10)
        self.grouped_stat_pc.set(1)
        self.grouped_stat_pc.grid(row=3, column=1, sticky='w', padx=5)
        
        ttk.Label(frame, text="Significance Level (α):").grid(row=4, column=0, sticky='w', pady=5)
        self.grouped_stat_alpha = ttk.Combobox(frame, values=['0.01', '0.05', '0.1'], width=10, state='readonly')
        self.grouped_stat_alpha.set('0.05')
        self.grouped_stat_alpha.grid(row=4, column=1, sticky='w', padx=5)
        
        ttk.Button(frame, text="Test Single PC (Grouped Means)", command=self.test_single_pc_grouped).grid(row=5, column=0, columnspan=2, pady=10)
        
        ttk.Separator(frame, orient='horizontal').grid(row=6, column=0, columnspan=2, sticky='ew', pady=10)
        
        ttk.Label(frame, text="Max PC for Multiple Tests:").grid(row=7, column=0, sticky='w', pady=5)
        self.grouped_max_stat_pc = ttk.Spinbox(frame, from_=2, to=20, width=10)
        self.grouped_max_stat_pc.set(10)
        self.grouped_max_stat_pc.grid(row=7, column=1, sticky='w', padx=5)
        
        ttk.Button(frame, text="Test Multiple PCs (Grouped Means)", command=self.test_multi_pc_grouped).grid(row=8, column=0, columnspan=2, pady=10)
        
        ttk.Separator(frame, orient='horizontal').grid(row=9, column=0, columnspan=2, sticky='ew', pady=10)
        
        ttk.Label(frame, text="Export Statistics:", font=('TkDefaultFont', 9, 'bold')).grid(row=10, column=0, columnspan=2, sticky='w', pady=5)
        ttk.Button(frame, text="Export Grouped Stats Results", command=self.export_grouped_stats_results).grid(row=11, column=0, columnspan=2, pady=5)
        
        self.grouped_stats_status = ttk.Label(frame, text="Ready to analyze", foreground="blue")
        self.grouped_stats_status.grid(row=12, column=0, columnspan=2, pady=10)
        
    def setup_export_tab(self):
        frame = ttk.LabelFrame(self.tab_export, text="Export PCA Results", padding=15)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text="Export your PCA results to Excel or Pickle format", foreground="gray").grid(row=0, column=0, columnspan=3, pady=10, sticky='w')
        
        ttk.Label(frame, text="Output Directory:").grid(row=1, column=0, sticky='w', pady=5)
        self.export_dir = tk.StringVar(value=os.getcwd())
        ttk.Entry(frame, textvariable=self.export_dir, width=40).grid(row=1, column=1, padx=5, sticky='w')
        ttk.Button(frame, text="Browse", command=self.browse_export_dir).grid(row=1, column=2, padx=5)
        
        ttk.Separator(frame, orient='horizontal').grid(row=2, column=0, columnspan=3, sticky='ew', pady=10)
        
        ttk.Label(frame, text="Select data to export:", font=('TkDefaultFont', 9, 'bold')).grid(row=3, column=0, columnspan=3, sticky='w', pady=(10, 5))
        
        self.export_scores = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="PCA Scores", variable=self.export_scores).grid(row=4, column=0, sticky='w', pady=2)
        
        self.export_loadings = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="PCA Loadings", variable=self.export_loadings).grid(row=5, column=0, sticky='w', pady=2)
        
        self.export_variance = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Explained Variance", variable=self.export_variance).grid(row=6, column=0, sticky='w', pady=2)
        
        ttk.Label(frame, text="Note: Categorical data will be automatically included\n(cleaned if outliers were removed)", foreground="gray", font=('TkDefaultFont', 8)).grid(row=7, column=0, columnspan=3, sticky='w', pady=5)
        
        ttk.Separator(frame, orient='horizontal').grid(row=8, column=0, columnspan=3, sticky='ew', pady=10)
        
        ttk.Label(frame, text="File Format:").grid(row=9, column=0, sticky='w', pady=5)
        self.export_filetype = tk.StringVar(value='xlsx')
        
        filetype_frame = ttk.Frame(frame)
        filetype_frame.grid(row=9, column=1, columnspan=2, sticky='w', pady=5)
        ttk.Radiobutton(filetype_frame, text="Excel (.xlsx)", variable=self.export_filetype, value='xlsx').pack(side='left', padx=5)
        ttk.Radiobutton(filetype_frame, text="Pickle (.pkl)", variable=self.export_filetype, value='pkl').pack(side='left', padx=5)
        
        ttk.Button(frame, text="Export PCA Results", command=self.export_results).grid(row=10, column=0, columnspan=3, pady=20)
        
        self.export_status = ttk.Label(frame, text="Ready to export", foreground="blue")
        self.export_status.grid(row=11, column=0, columnspan=3, pady=10)
        
    def browse_file(self):
        filename = filedialog.askopenfilename(title="Select Excel file", filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")])
        if filename:
            self.file_path.set(filename)
            
    def browse_pickle(self):
        filename = filedialog.askopenfilename(title="Select Pickle file", filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        if filename:
            self.pickle_path.set(filename)
            
    def browse_export_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory", initialdir=self.export_dir.get())
        if directory:
            self.export_dir.set(directory)
            
    def convert_excel(self):
        if not self.file_path.get():
            messagebox.showerror("Error", "Please select an Excel file first")
            return
        try:
            output_path = self.file_path.get().replace('.xlsx', '.pkl').replace('.xls', '.pkl')
            excel_to_pickle(self.file_path.get(), output_path)
            self.pickle_path.set(output_path)
            messagebox.showinfo("Success", f"Converted to: {output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Conversion failed:\n{str(e)}")
            
    def load_data(self):
        if not self.pickle_path.get():
            messagebox.showerror("Error", "Please select a Pickle file")
            return
        try:
            self.categoricals, self.data = extract_data(self.pickle_path.get())
            cat_cols = list(self.categoricals.columns)
            self.color_by['values'] = cat_cols
            self.group_by['values'] = cat_cols
            self.grouped_group_by['values'] = cat_cols
            self.grouped_mean_by['values'] = cat_cols
            
            data_min = self.data.columns.min()
            data_max = self.data.columns.max()
            self.crop_info.config(text=f"Data range: {data_min:.1f} to {data_max:.1f} | Samples: {self.data.shape[0]}, Features: {self.data.shape[1]}", foreground="black")
            self.start_auto_label.config(text=f"(Auto: {data_min:.1f})")
            self.end_auto_label.config(text=f"(Auto: {data_max:.1f})")
            self.load_status.config(text=f"✓ Loaded: {self.data.shape[0]} samples, {self.data.shape[1]} features", foreground="green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data:\n{str(e)}")
            self.load_status.config(text="Error loading data", foreground="red")
            
    def crop_data_action(self):
        if self.data is None:
            messagebox.showerror("Error", "Please load data first")
            return
        try:
            start = None
            end = None
            if self.crop_start.get().strip():
                start = float(self.crop_start.get().strip())
            if self.crop_end.get().strip():
                end = float(self.crop_end.get().strip())
            self.data = crop_data(self.data, start=start, end=end)
            data_min = self.data.columns.min()
            data_max = self.data.columns.max()
            self.crop_info.config(text=f"✓ Cropped range: {data_min:.1f} to {data_max:.1f} | Samples: {self.data.shape[0]}, Features: {self.data.shape[1]}", foreground="green")
            self.crop_status.config(text="✓ Data cropped successfully", foreground="green")
        except ValueError:
            messagebox.showerror("Error", "Start and End values must be numbers")
            self.crop_status.config(text="Error: Invalid input", foreground="red")
        except Exception as e:
            messagebox.showerror("Error", f"Cropping failed:\n{str(e)}")
            self.crop_status.config(text="Error during cropping", foreground="red")
            
    def run_pca(self):
        if self.data is None:
            messagebox.showerror("Error", "Please load data first")
            return
        try:
            self.pca_status.config(text="Running PCA...", foreground="orange")
            self.root.update()
            
            color_by = self.color_by.get() if self.color_by.get() else None
            
            # Handle outlier_components parameter
            outlier_comp = None
            if self.outlier_components.get().strip():
                outlier_comp = int(self.outlier_components.get().strip())
            
            self.pca_results = perform_pca(
                data=self.data, 
                categoricals=self.categoricals, 
                color_by=color_by, 
                n_components=int(self.n_components.get()), 
                scale_data=self.scale_data.get(), 
                interactive=False, 
                pc_x=int(self.pc_x.get()), 
                pc_y=int(self.pc_y.get()), 
                detect_outliers=self.detect_outliers.get(), 
                outlier_alpha=float(self.outlier_alpha.get()),
                outlier_components=outlier_comp,
                outlier_action=self.outlier_action.get()
            )
            
            self.pca_status.config(text="✓ PCA completed successfully", foreground="green")
        except Exception as e:
            messagebox.showerror("Error", f"PCA failed:\n{str(e)}\n\n{traceback.format_exc()}")
            self.pca_status.config(text="Error during PCA", foreground="red")
            
    def plot_loadings_viz(self):
        if self.pca_results is None:
            messagebox.showerror("Error", "Please run PCA analysis first")
            return
        try:
            plot_loadings(self.pca_results, n_components=[int(self.load_pc1.get()), int(self.load_pc2.get())])
            self.viz_status.config(text="✓ Loadings plot displayed", foreground="green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot loadings:\n{str(e)}")
            
    def multi_loadings_viz(self):
        if self.pca_results is None:
            messagebox.showerror("Error", "Please run PCA analysis first")
            return
        try:
            multi_loadings(self.pca_results, max_PC=int(self.max_load_pc.get()))
            self.viz_status.config(text="✓ Multiple loadings plots displayed", foreground="green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot loadings:\n{str(e)}")
            
    def test_single_pc(self):
        if self.pca_results is None or self.group_by.get() == '':
            messagebox.showerror("Error", "Please run PCA and select a grouping variable")
            return
        try:
            result = test_pc_significance(
                self.pca_results, 
                self.categoricals, 
                group_by=self.group_by.get(), 
                pc_number=int(self.stat_pc.get()), 
                alpha=float(self.stat_alpha.get())
            )
            self.stats_results = result  # Store for export
            self.stats_status.config(text="✓ Statistical test completed", foreground="green")
        except Exception as e:
            messagebox.showerror("Error", f"Statistical test failed:\n{str(e)}")
            
    def test_multi_pc(self):
        if self.pca_results is None or self.group_by.get() == '':
            messagebox.showerror("Error", "Please run PCA and select a grouping variable")
            return
        try:
            results = multi_stat(
                self.pca_results, 
                self.categoricals, 
                group_by=self.group_by.get(), 
                max_PC=int(self.max_stat_pc.get())
            )
            self.stats_results = results  # Store for export
            self.stats_status.config(text="✓ Multiple tests completed", foreground="green")
        except Exception as e:
            messagebox.showerror("Error", f"Multiple tests failed:\n{str(e)}")

    def test_single_pc_grouped(self):
        if self.pca_results is None:
            messagebox.showerror("Error", "Please run PCA analysis first")
            return
        if self.grouped_group_by.get() == '' or self.grouped_mean_by.get() == '':
            messagebox.showerror("Error", "Please select both 'Group by' and 'Mean by' variables")
            return
        try:
            self.grouped_stats_status.config(text="Running test...", foreground="orange")
            self.root.update()
            result = test_pc_significance_grouped_means(
                self.pca_results, 
                self.categoricals, 
                group_by=self.grouped_group_by.get(), 
                mean_by=self.grouped_mean_by.get(), 
                pc_number=int(self.grouped_stat_pc.get()), 
                alpha=float(self.grouped_stat_alpha.get())
            )
            self.grouped_stats_results = result  # Store for export
            self.grouped_stats_status.config(text="✓ Grouped means test completed", foreground="green")
        except Exception as e:
            messagebox.showerror("Error", f"Grouped means test failed:\n{str(e)}\n\n{traceback.format_exc()}")
            self.grouped_stats_status.config(text="Error during test", foreground="red")
            
    def test_multi_pc_grouped(self):
        if self.pca_results is None:
            messagebox.showerror("Error", "Please run PCA analysis first")
            return
        if self.grouped_group_by.get() == '' or self.grouped_mean_by.get() == '':
            messagebox.showerror("Error", "Please select both 'Group by' and 'Mean by' variables")
            return
        try:
            self.grouped_stats_status.config(text="Running multiple tests...", foreground="orange")
            self.root.update()
            results = multi_stat_grouped_means(
                self.pca_results, 
                self.categoricals, 
                group_by=self.grouped_group_by.get(), 
                mean_by=self.grouped_mean_by.get(), 
                max_PC=int(self.grouped_max_stat_pc.get())
            )
            self.grouped_stats_results = results  # Store for export
            self.grouped_stats_status.config(text="✓ Multiple grouped means tests completed", foreground="green")
        except Exception as e:
            messagebox.showerror("Error", f"Multiple grouped means tests failed:\n{str(e)}\n\n{traceback.format_exc()}")
            self.grouped_stats_status.config(text="Error during tests", foreground="red")
    
    def export_results(self):
        if self.pca_results is None:
            messagebox.showerror("Error", "Please run PCA analysis first")
            return
        
        if not self.export_dir.get():
            messagebox.showerror("Error", "Please select an output directory")
            return
        
        try:
            self.export_status.config(text="Exporting...", foreground="orange")
            self.root.update()
            
            # Determine which categoricals to use
            if 'outlier_cats' in self.pca_results and self.pca_results['outlier_cats'] is not None:
                cats_to_export = self.pca_results['outlier_cats']
            else:
                cats_to_export = self.categoricals
            
            export_data(
                pca_results=self.pca_results,
                categoricals=cats_to_export,
                output_directory=self.export_dir.get(),
                scores=self.export_scores.get(),
                loadings=self.export_loadings.get(),
                ex_variance=self.export_variance.get(),
                filetype=self.export_filetype.get()
            )
            
            self.export_status.config(text="✓ Export completed successfully", foreground="green")
            messagebox.showinfo("Success", f"Results exported to:\n{self.export_dir.get()}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{str(e)}\n\n{traceback.format_exc()}")
            self.export_status.config(text="Error during export", foreground="red")
    
    def export_stats_results(self):
        if self.stats_results is None:
            messagebox.showerror("Error", "No statistical test results to export. Please run tests first.")
            return
        
        if not self.export_dir.get():
            messagebox.showerror("Error", "Please select an output directory")
            return
        
        try:
            self.stats_status.config(text="Exporting stats...", foreground="orange")
            self.root.update()
            
            export_stats(
                stats_results=self.stats_results,
                output_directory=self.export_dir.get(),
                filename="statistical_results",
                filetype=self.export_filetype.get(),
                include_pairwise=True
            )
            
            self.stats_status.config(text="✓ Stats export completed", foreground="green")
            messagebox.showinfo("Success", f"Statistical results exported to:\n{self.export_dir.get()}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Stats export failed:\n{str(e)}\n\n{traceback.format_exc()}")
            self.stats_status.config(text="Error during export", foreground="red")
    
    def export_grouped_stats_results(self):
        if self.grouped_stats_results is None:
            messagebox.showerror("Error", "No grouped means test results to export. Please run tests first.")
            return
        
        if not self.export_dir.get():
            messagebox.showerror("Error", "Please select an output directory")
            return
        
        try:
            self.grouped_stats_status.config(text="Exporting stats...", foreground="orange")
            self.root.update()
            
            export_stats(
                stats_results=self.grouped_stats_results,
                output_directory=self.export_dir.get(),
                filename="grouped_means_statistical_results",
                filetype=self.export_filetype.get(),
                include_pairwise=True
            )
            
            self.grouped_stats_status.config(text="✓ Stats export completed", foreground="green")
            messagebox.showinfo("Success", f"Grouped means statistical results exported to:\n{self.export_dir.get()}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Grouped stats export failed:\n{str(e)}\n\n{traceback.format_exc()}")
            self.grouped_stats_status.config(text="Error during export", foreground="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = PCAAnalysisGUI(root)
    root.mainloop()