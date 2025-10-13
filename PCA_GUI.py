# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 13:39:37 2025

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
from threading import Thread

# Import your PCA functions
# Make sure your PCA code is in the same directory or installed as a module
try:
    from PCA_claude import (extract_data, perform_pca, plot_loadings, 
                              test_pc_significance, multi_stat, 
                              excel_to_pickle, crop_data, multi_loadings)
except ImportError:
    print("Warning: Could not import PCA functions. Make sure 'pca_analysis.py' exists in the same directory.")

class PCAAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PCA Analysis Tool")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Data storage
        self.file_path = tk.StringVar()
        self.pickle_path = tk.StringVar()
        self.data = None
        self.categoricals = None
        self.pca_results = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Create the main UI layout"""
        # Create notebook (tabs)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Data Loading
        self.tab_load = ttk.Frame(notebook)
        notebook.add(self.tab_load, text='Step 1: Load Data')
        self.setup_load_tab()
        
        # Tab 1.5: Crop Data
        self.tab_crop = ttk.Frame(notebook)
        notebook.add(self.tab_crop, text='Step 1.5: Crop Data')
        self.setup_crop_tab()
        
        # Tab 2: PCA Settings
        self.tab_pca = ttk.Frame(notebook)
        notebook.add(self.tab_pca, text='Step 2: PCA Analysis')
        self.setup_pca_tab()
        
        # Tab 3: Visualization
        self.tab_viz = ttk.Frame(notebook)
        notebook.add(self.tab_viz, text='Step 3: Visualizations')
        self.setup_viz_tab()
        
        # Tab 4: Statistics
        self.tab_stats = ttk.Frame(notebook)
        notebook.add(self.tab_stats, text='Step 4: Statistics')
        self.setup_stats_tab()
        
    def setup_load_tab(self):
        """Setup data loading tab"""
        frame = ttk.LabelFrame(self.tab_load, text="Data Loading", padding=15)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # File selection
        ttk.Label(frame, text="Select Excel file:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Entry(frame, textvariable=self.file_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5)
        
        # Convert button
        ttk.Button(frame, text="Convert Excel to Pickle", 
                   command=self.convert_excel).grid(row=1, column=0, columnspan=3, pady=10)
        
        # Pickle file selection
        ttk.Label(frame, text="Or select Pickle file:").grid(row=2, column=0, sticky='w', pady=5)
        ttk.Entry(frame, textvariable=self.pickle_path, width=50).grid(row=2, column=1, padx=5)
        ttk.Button(frame, text="Browse", command=self.browse_pickle).grid(row=2, column=2, padx=5)
        
        # Load button
        ttk.Button(frame, text="Load Data", command=self.load_data, 
                   style='Accent.TButton').grid(row=3, column=0, columnspan=3, pady=15)
        
        # Status
        self.load_status = ttk.Label(frame, text="No data loaded", foreground="red")
        self.load_status.grid(row=4, column=0, columnspan=3, pady=10)
        
    def setup_crop_tab(self):
        """Setup data cropping tab"""
        frame = ttk.LabelFrame(self.tab_crop, text="Crop Data by Wavenumber Range", padding=15)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Info label
        info_label = ttk.Label(frame, text="Optional: Crop data to a specific wavenumber range", 
                               foreground="gray")
        info_label.grid(row=0, column=0, columnspan=3, pady=5, sticky='w')
        
        # Start wavenumber
        ttk.Label(frame, text="Start Wavenumber:").grid(row=1, column=0, sticky='w', pady=5)
        self.crop_start = ttk.Entry(frame, width=15)
        self.crop_start.grid(row=1, column=1, sticky='w', padx=5)
        self.start_auto_label = ttk.Label(frame, text="(Auto: min)", foreground="gray")
        self.start_auto_label.grid(row=1, column=2, sticky='w', padx=5)
        
        # End wavenumber
        ttk.Label(frame, text="End Wavenumber:").grid(row=2, column=0, sticky='w', pady=5)
        self.crop_end = ttk.Entry(frame, width=15)
        self.crop_end.grid(row=2, column=1, sticky='w', padx=5)
        self.end_auto_label = ttk.Label(frame, text="(Auto: max)", foreground="gray")
        self.end_auto_label.grid(row=2, column=2, sticky='w', padx=5)
        
        # Separator
        ttk.Separator(frame, orient='horizontal').grid(row=3, column=0, columnspan=3, sticky='ew', pady=10)
        
        # Data info
        self.crop_info = ttk.Label(frame, text="No data loaded", foreground="gray")
        self.crop_info.grid(row=4, column=0, columnspan=3, pady=5, sticky='w')
        
        # Crop button
        ttk.Button(frame, text="Crop Data", command=self.crop_data_action, 
                   style='Accent.TButton').grid(row=5, column=0, columnspan=3, pady=15)
        
        # Status
        self.crop_status = ttk.Label(frame, text="Ready", foreground="blue")
        self.crop_status.grid(row=6, column=0, columnspan=3, pady=10)
        
    def setup_pca_tab(self):
        """Setup PCA analysis tab"""
        frame = ttk.LabelFrame(self.tab_pca, text="PCA Settings", padding=15)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Number of components
        ttk.Label(frame, text="Number of Components:").grid(row=0, column=0, sticky='w', pady=5)
        self.n_components = ttk.Spinbox(frame, from_=2, to=50, width=10)
        self.n_components.set(10)
        self.n_components.grid(row=0, column=1, sticky='w', padx=5)
        
        # Scale data
        self.scale_data = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Scale Data (Standardize)", 
                        variable=self.scale_data).grid(row=1, column=0, columnspan=2, pady=5, sticky='w')
        
        # Outlier detection
        self.detect_outliers = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Detect Outliers", 
                        variable=self.detect_outliers).grid(row=2, column=0, columnspan=2, pady=5, sticky='w')
        
        # Outlier alpha
        ttk.Label(frame, text="Outlier Significance Level (α):").grid(row=3, column=0, sticky='w', pady=5)
        self.outlier_alpha = ttk.Combobox(frame, values=['0.01', '0.05', '0.1'], width=10, state='readonly')
        self.outlier_alpha.set('0.05')
        self.outlier_alpha.grid(row=3, column=1, sticky='w', padx=5)
        
        # Outlier action
        ttk.Label(frame, text="Outlier Action:").grid(row=4, column=0, sticky='w', pady=5)
        self.outlier_action = ttk.Combobox(frame, values=['show', 'hide', 'remove'], 
                                           width=10, state='readonly')
        self.outlier_action.set('show')
        self.outlier_action.grid(row=4, column=1, sticky='w', padx=5)
        
        # Color by selection
        ttk.Label(frame, text="Color by (optional):").grid(row=5, column=0, sticky='w', pady=5)
        self.color_by = ttk.Combobox(frame, width=20, state='readonly')
        self.color_by.grid(row=5, column=1, sticky='w', padx=5)
        
        # PC selection
        ttk.Label(frame, text="X-axis PC:").grid(row=6, column=0, sticky='w', pady=5)
        self.pc_x = ttk.Spinbox(frame, from_=1, to=50, width=10)
        self.pc_x.set(1)
        self.pc_x.grid(row=6, column=1, sticky='w', padx=5)
        
        ttk.Label(frame, text="Y-axis PC:").grid(row=7, column=0, sticky='w', pady=5)
        self.pc_y = ttk.Spinbox(frame, from_=1, to=50, width=10)
        self.pc_y.set(2)
        self.pc_y.grid(row=7, column=1, sticky='w', padx=5)
        
        # Run button
        ttk.Button(frame, text="Run PCA Analysis", command=self.run_pca, 
                   style='Accent.TButton').grid(row=8, column=0, columnspan=2, pady=15)
        
        # Status
        self.pca_status = ttk.Label(frame, text="Ready to analyze", foreground="blue")
        self.pca_status.grid(row=9, column=0, columnspan=2, pady=10)
        
    def setup_viz_tab(self):
        """Setup visualization tab"""
        frame = ttk.LabelFrame(self.tab_viz, text="Visualization Options", padding=15)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # PC selection for loadings
        ttk.Label(frame, text="Loadings - PC 1:").grid(row=0, column=0, sticky='w', pady=5)
        self.load_pc1 = ttk.Spinbox(frame, from_=1, to=50, width=10)
        self.load_pc1.set(1)
        self.load_pc1.grid(row=0, column=1, sticky='w', padx=5)
        
        ttk.Label(frame, text="Loadings - PC 2:").grid(row=1, column=0, sticky='w', pady=5)
        self.load_pc2 = ttk.Spinbox(frame, from_=1, to=50, width=10)
        self.load_pc2.set(2)
        self.load_pc2.grid(row=1, column=1, sticky='w', padx=5)
        
        ttk.Button(frame, text="Plot Loadings", 
                   command=self.plot_loadings_viz).grid(row=2, column=0, columnspan=2, pady=10)
        
        # Separator
        ttk.Separator(frame, orient='horizontal').grid(row=3, column=0, columnspan=2, sticky='ew', pady=10)
        
        # Multiple loadings
        ttk.Label(frame, text="Max PC for Multiple Loadings:").grid(row=4, column=0, sticky='w', pady=5)
        self.max_load_pc = ttk.Spinbox(frame, from_=2, to=20, width=10)
        self.max_load_pc.set(5)
        self.max_load_pc.grid(row=4, column=1, sticky='w', padx=5)
        
        ttk.Button(frame, text="Plot Multiple Loadings", 
                   command=self.multi_loadings_viz).grid(row=5, column=0, columnspan=2, pady=10)
        
        # Status
        self.viz_status = ttk.Label(frame, text="Ready to visualize", foreground="blue")
        self.viz_status.grid(row=6, column=0, columnspan=2, pady=10)
        
    def setup_stats_tab(self):
        """Setup statistics tab"""
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
        
        ttk.Button(frame, text="Test Single PC", 
                   command=self.test_single_pc).grid(row=3, column=0, columnspan=2, pady=10)
        
        # Separator
        ttk.Separator(frame, orient='horizontal').grid(row=4, column=0, columnspan=2, sticky='ew', pady=10)
        
        ttk.Label(frame, text="Max PC for Multiple Tests:").grid(row=5, column=0, sticky='w', pady=5)
        self.max_stat_pc = ttk.Spinbox(frame, from_=2, to=20, width=10)
        self.max_stat_pc.set(10)
        self.max_stat_pc.grid(row=5, column=1, sticky='w', padx=5)
        
        ttk.Button(frame, text="Test Multiple PCs", 
                   command=self.test_multi_pc).grid(row=6, column=0, columnspan=2, pady=10)
        
        # Status
        self.stats_status = ttk.Label(frame, text="Ready to analyze", foreground="blue")
        self.stats_status.grid(row=7, column=0, columnspan=2, pady=10)
        
    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Excel file",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if filename:
            self.file_path.set(filename)
            
    def browse_pickle(self):
        filename = filedialog.askopenfilename(
            title="Select Pickle file",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if filename:
            self.pickle_path.set(filename)
            
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
            
            # Update color_by and group_by dropdowns
            cat_cols = list(self.categoricals.columns)
            self.color_by['values'] = cat_cols
            self.group_by['values'] = cat_cols
            
            # Update crop info
            data_min = self.data.columns.min()
            data_max = self.data.columns.max()
            self.crop_info.config(
                text=f"Data range: {data_min:.1f} to {data_max:.1f} | Samples: {self.data.shape[0]}, Features: {self.data.shape[1]}",
                foreground="black"
            )
            self.start_auto_label.config(text=f"(Auto: {data_min:.1f})")
            self.end_auto_label.config(text=f"(Auto: {data_max:.1f})")
            
            self.load_status.config(
                text=f"✓ Loaded: {self.data.shape[0]} samples, {self.data.shape[1]} features",
                foreground="green"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data:\n{str(e)}")
            self.load_status.config(text="Error loading data", foreground="red")
            
    def run_pca(self):
        if self.data is None:
            messagebox.showerror("Error", "Please load data first")
            return
        
        try:
            self.pca_status.config(text="Running PCA...", foreground="orange")
            self.root.update()
            
            color_by = self.color_by.get() if self.color_by.get() else None
            
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
            pc1 = int(self.load_pc1.get())
            pc2 = int(self.load_pc2.get())
            plot_loadings(self.pca_results, n_components=[pc1, pc2])
            self.viz_status.config(text="✓ Loadings plot displayed", foreground="green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot loadings:\n{str(e)}")
            
    def multi_loadings_viz(self):
        if self.pca_results is None:
            messagebox.showerror("Error", "Please run PCA analysis first")
            return
        
        try:
            max_pc = int(self.max_load_pc.get())
            multi_loadings(self.pca_results, max_PC=max_pc)
            self.viz_status.config(text="✓ Multiple loadings plots displayed", foreground="green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot loadings:\n{str(e)}")
            
    def test_single_pc(self):
        if self.pca_results is None or self.group_by.get() == '':
            messagebox.showerror("Error", "Please run PCA and select a grouping variable")
            return
        
        try:
            test_pc_significance(
                self.pca_results,
                self.categoricals,
                group_by=self.group_by.get(),
                pc_number=int(self.stat_pc.get()),
                alpha=float(self.stat_alpha.get())
            )
            self.stats_status.config(text="✓ Statistical test completed", foreground="green")
        except Exception as e:
            messagebox.showerror("Error", f"Statistical test failed:\n{str(e)}")
            
    def test_multi_pc(self):
        if self.pca_results is None or self.group_by.get() == '':
            messagebox.showerror("Error", "Please run PCA and select a grouping variable")
            return
        
        try:
            multi_stat(
                self.pca_results,
                self.categoricals,
                group_by=self.group_by.get(),
                max_PC=int(self.max_stat_pc.get())
            )
            self.stats_status.config(text="✓ Multiple tests completed", foreground="green")
        except Exception as e:
            messagebox.showerror("Error", f"Multiple tests failed:\n{str(e)}")

    def crop_data_action(self):
        """Crop the data based on start and end wavenumbers"""
        if self.data is None:
            messagebox.showerror("Error", "Please load data first")
            return
        
        try:
            # Get start and end values, None if empty
            start = None
            end = None
            
            if self.crop_start.get().strip():
                start = float(self.crop_start.get().strip())
            
            if self.crop_end.get().strip():
                end = float(self.crop_end.get().strip())
            
            # Crop the data
            self.data = crop_data(self.data, start=start, end=end)
            
            # Update info
            data_min = self.data.columns.min()
            data_max = self.data.columns.max()
            self.crop_info.config(
                text=f"✓ Cropped range: {data_min:.1f} to {data_max:.1f} | Samples: {self.data.shape[0]}, Features: {self.data.shape[1]}",
                foreground="green"
            )
            self.crop_status.config(text="✓ Data cropped successfully", foreground="green")
            
        except ValueError as e:
            messagebox.showerror("Error", "Start and End values must be numbers")
            self.crop_status.config(text="Error: Invalid input", foreground="red")
        except Exception as e:
            messagebox.showerror("Error", f"Cropping failed:\n{str(e)}")
            self.crop_status.config(text="Error during cropping", foreground="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = PCAAnalysisGUI(root)
    root.mainloop()