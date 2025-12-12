# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 11:44:56 2025

@author: User
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats
# from scipy.stats import ttest_ind, f_oneway
# from statsmodels.formula.api import ols
# from statsmodels.stats.anova import anova_lm
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

#%%
def excel_to_pickle(path, output_path):
    '''
    Transforms excel sheet into pickle file that is needed for following operations

    Parameters
    ----------
    path : str
        Path to excel
    output_path : str
        Desired output path. Make sure to change .xlsx to .pkl

    Returns
    -------
    None.

    '''
    
    df= pd.read_excel(path)
    df.to_pickle(output_path)
    
#%%   
def extract_data(path: str= None, df: pd.DataFrame= None):
    '''
    
    Reads .pkl or df file and splits it into data and categorical data for further analysis.

    Parameters
    ----------
    path : str
        Path to pickle file.

    Returns
    -------
    categoricals : pd.DataFRame
        DataFrame containing your categorical data (non-numerical headers)
    data : pd.DataFrame
        Data frame containing your measrued spectra (numerical headers)

    
    '''
    if path:
        df = pd.read_pickle(path)
    else:
        df= df
    
    # Convert column names to numeric where possible
    new_columns = []
    for col in df.columns:
        try:
            new_columns.append(float(col))
        except (ValueError, TypeError):
            new_columns.append(col)
    
    df.columns = new_columns
    
    # Create boolean mask for numeric column names
    cols = df.columns
    idx = cols.map(lambda x: isinstance(x, (int, float)))
    
    # Split into numeric and categorical columns
    data = df.loc[:, idx]           # Numeric columns
    categoricals = df.loc[:, ~idx]  # Non-numeric columns
    
    #Reset index for subsets (fixes bug after outlier removal)
    data.reset_index(drop= True, inplace= True)
    categoricals.reset_index(drop= True, inplace= True)
    
    return categoricals, data

#%%Crop data
def crop_data(data: pd.DataFrame, start: int=None, end: int=None):
    '''
    Crop data at given wavenumbers

    Parameters
    ----------
    data : pd.DataFrame
        DESCRIPTION.
    start : int, optional
        DESCRIPTION. The default is None.
    end : int, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    cropped_df : TYPE
        DESCRIPTION.

    '''
    # Set default values after data is available
    if start is None:
        start = data.columns.min()
    if end is None:
        end = data.columns.max()
    
    cropped_df = data.loc[:, (data.columns >= start) & (data.columns <= end)]
    return cropped_df


#%%
def perform_pca(data, categoricals=None, color_by=None, n_components=10, 
                interactive=False, figsize=(12, 8),
                pc_x=1, pc_y=2, html_path=None, detect_outliers=False,
                outlier_alpha=0.05, outlier_components=None, outlier_action='show',
                _random_state=42):  # NEW: Add random state for reproducibility
    """
    Perform PCA analysis on numeric data with optional categorical coloring and outlier detection.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Numeric data for PCA (samples x features)
    categoricals : pd.DataFrame, optional
        Categorical data for coloring points
    color_by : str or list, optional
        Column name(s) from categoricals to color by
    n_components : int, default=10
        Number of principal components to compute
    interactive : bool, default=False
        Whether to create interactive plotly plots
    figsize : tuple, default=(12, 8)
        Figure size for matplotlib plots
    pc_x : int, default=1
        Principal component to plot on x-axis (1-based indexing)
    pc_y : int, default=2
        Principal component to plot on y-axis (1-based indexing)
    html_path : str, optional
        Path to save interactive HTML file. If None and interactive=True,
        saves as 'pca_plot.html' in current directory
    detect_outliers : bool, default=False
        Whether to detect outliers using Hotelling's T² statistic
    outlier_alpha : float, default=0.05
        Significance level for outlier detection (typically 0.01 or 0.05)
    outlier_components : int, optional
        Number of PCs to use for outlier detection. If None, uses all n_components
    outlier_action : str, default='show'
        What to do with detected outliers:
        - 'show': Display outliers with different markers (default)
        - 'hide': Don't display outliers in plots
        - 'remove': Remove outliers from all results (single-pass, no recursion)
    _random_state : int, default=42
        Random state for reproducibility (internal parameter)
        
    Returns:
    --------
    dict: Contains PCA results, scores, loadings, explained variance, and outlier info
    """
    
    # Validate parameters
    if detect_outliers and outlier_action not in ['show', 'hide', 'remove']:
        raise ValueError("outlier_action must be 'show', 'hide', or 'remove'")
    
    # Validate PC indices
    if pc_x < 1 or pc_y < 1:
        raise ValueError("PC indices must be >= 1 (1-based indexing)")
    if pc_x > n_components or pc_y > n_components:
        raise ValueError(f"PC indices cannot exceed n_components ({n_components})")
    if pc_x == pc_y:
        raise ValueError("pc_x and pc_y must be different")
    
    # CRITICAL FIX: Ensure data is sorted by index for reproducibility
    data = data.sort_index()
    if categoricals is not None:
        categoricals = categoricals.sort_index()
    
    # Handle missing data
    if data.isnull().sum().sum() > 0:
        print(f"Warning: Found {data.isnull().sum().sum()} missing values. Filling with column means.")
        data = data.fillna(data.mean())
    
    # Prepare the pipeline - NO SCALING for Raman data
    pipeline = Pipeline([
        ('pca', PCA(n_components=n_components, random_state=_random_state))
    ])
    
    # Fit PCA
    pca_scores = pipeline.fit_transform(data)
    pca_model = pipeline.named_steps['pca']
    
    # Create scores DataFrame
    pc_columns = [f'PC{i+1}' for i in range(n_components)]
    scores_df = pd.DataFrame(pca_scores, columns=pc_columns, index=data.index)
    
    # Get loadings
    loadings = pd.DataFrame(
        pca_model.components_.T,
        columns=pc_columns,
        index=data.columns
    )
    
    # Explained variance
    explained_var = pca_model.explained_variance_ratio_
    
    # Outlier detection using Hotelling's T²
    outlier_info = None
    is_outlier = None
    
    if detect_outliers:
        
        # Determine number of components to use for outlier detection
        n_outlier_components = outlier_components if outlier_components is not None else n_components
        n_outlier_components = min(n_outlier_components, n_components, data.shape[0] - 1)
        
        # Use subset of PCs for outlier detection
        scores_for_outliers = pca_scores[:, :n_outlier_components]
        
        # Calculate Hotelling's T² statistic
        n_samples = scores_for_outliers.shape[0]
        p = scores_for_outliers.shape[1]
        
        # Center the scores (they should already be centered, but ensure it)
        scores_centered = scores_for_outliers - np.mean(scores_for_outliers, axis=0)
        
        # Calculate covariance matrix
        cov_matrix = np.cov(scores_centered.T)
        
        # Calculate T² for each sample
        try:
            inv_cov = np.linalg.inv(cov_matrix)
            t_squared = np.array([
                scores_centered[i] @ inv_cov @ scores_centered[i].T 
                for i in range(n_samples)
            ])
        except np.linalg.LinAlgError:
            print("Warning: Covariance matrix is singular. Using pseudo-inverse.")
            inv_cov = np.linalg.pinv(cov_matrix)
            t_squared = np.array([
                scores_centered[i] @ inv_cov @ scores_centered[i].T 
                for i in range(n_samples)
            ])
        
        # Calculate critical value using F-distribution
        # T² follows (p(n-1)/(n-p)) * F(p, n-p) distribution
        f_crit = stats.f.ppf(1 - outlier_alpha, p, n_samples - p)
        t_squared_crit = (p * (n_samples - 1)) / (n_samples - p) * f_crit
        
        # Identify outliers
        is_outlier = t_squared > t_squared_crit
        n_outliers = np.sum(is_outlier)
        
        outlier_info = {
            't_squared': t_squared,
            'critical_value': t_squared_crit,
            'is_outlier': is_outlier,
            'outlier_indices': data.index[is_outlier].tolist(),
            'n_outliers': n_outliers,
            'n_components_used': n_outlier_components,
            'alpha': outlier_alpha
        }
        
        print("\nOutlier Detection (Hotelling's T²):")
        print(f"- Components used: {n_outlier_components}")
        print(f"- Significance level (α): {outlier_alpha}")
        print(f"- Critical value: {t_squared_crit:.2f}")
        print(f"- Outliers detected: {n_outliers} ({n_outliers/n_samples*100:.1f}%)")
        #print(f"- Outlier indices: {data.index[is_outlier].tolist()}")
        print(f"- Action: {outlier_action}")

        
        # CORRECTED: Handle outlier action with SINGLE-PASS removal (no recursion)
        if outlier_action == 'remove' and n_outliers > 0:
            print("\n⚠ SINGLE-PASS OUTLIER REMOVAL (no recursive PCA)")
            print(f"Removing {n_outliers} outliers identified in current PC space...")
            
            # Filter out outliers from scores for return
            scores_df_clean = scores_df[~is_outlier].copy()
            scores_df_clean.reset_index(drop=True, inplace=True)
            
            # Filter categoricals if provided
            categoricals_clean = None
            if categoricals is not None:
                categoricals_clean = categoricals[~is_outlier].copy()
                categoricals_clean.reset_index(drop=True, inplace=True)
            
            # Store information about removed outliers
            outlier_info['removed_indices'] = data.index[is_outlier].tolist()
            outlier_info['removed_count'] = n_outliers
            outlier_info['removal_method'] = 'single_pass'
            
            print(f"✓ Results now contain {len(scores_df_clean)} samples (removed {n_outliers})")
            print("✓ Loadings and explained variance unchanged (based on full dataset)")
            print("⚠ Note: Outliers identified from ORIGINAL PC space (stable)")
            
            # Update variables for plotting - use cleaned data
            scores_df = scores_df_clean
            categoricals = categoricals_clean
            
            # Update color variable with cleaned categoricals
            if color_by is not None and categoricals_clean is not None:
                if isinstance(color_by, str):
                    color_by = [color_by]
                
                if len(color_by) == 1:
                    color_var = categoricals_clean[color_by[0]].astype(str)
                else:
                    color_var = categoricals_clean[color_by].apply(
                        lambda x: ' | '.join(x.astype(str)), axis=1
                    )
            
            # Clear outlier detection flag to prevent showing empty T² plot
            detect_outliers = False
            is_outlier = None
            
            # Continue to plotting section below (don't return early)
    
    # Convert PC indices to 0-based for array access
    pc_x_idx = pc_x - 1
    pc_y_idx = pc_y - 1
    pc_x_col = f'PC{pc_x}'
    pc_y_col = f'PC{pc_y}'
    
    # Prepare color variable
    color_var = None
    color_label = 'Group'
    
    if color_by is not None and categoricals is not None:
        if isinstance(color_by, str):
            color_by = [color_by]
        
        if len(color_by) == 1:
            color_var = categoricals[color_by[0]].astype(str)
            color_label = color_by[0]
        else:
            # Combine multiple categorical variables
            color_var = categoricals[color_by].apply(
                lambda x: ' | '.join(x.astype(str)), axis=1
            )
            color_label = ' | '.join(color_by)
    
    if interactive:
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Set default path if not provided
            if html_path is None:
                html_path = 'pca_plot.html'
            
            # Create interactive plotly plots
            n_plots = 3 if detect_outliers else 2
            subplot_titles = ['PCA Score Plot', 'Explained Variance']
            if detect_outliers:
                subplot_titles.append("Hotelling's T² Statistics")
            
            specs = [[{"secondary_y": False}, {"secondary_y": True}]]
            if detect_outliers:
                specs[0].append({"secondary_y": False})
            
            fig = make_subplots(
                rows=1, cols=n_plots,
                subplot_titles=tuple(subplot_titles),
                specs=specs,
                horizontal_spacing=0.12
            )
            
            # Score plot with chosen PCs
            plot_df = pd.DataFrame({
                'x': scores_df[pc_x_col],
                'y': scores_df[pc_y_col],
                'index': data.index
            })
            
            if detect_outliers and outlier_action == 'show':
                plot_df['outlier'] = ['Outlier' if x else 'Normal' for x in is_outlier]
            elif detect_outliers and outlier_action == 'hide':
                # Filter out outliers from plot
                plot_df = plot_df[~is_outlier]
            
            if color_var is not None:
                plot_df['color'] = color_var if outlier_action != 'hide' else color_var[~is_outlier]
                if detect_outliers and outlier_action == 'show':
                    # Combine color and outlier status
                    plot_df['display'] = plot_df['color'] + ' (' + plot_df['outlier'] + ')'
                    scatter_fig = px.scatter(
                        plot_df, x='x', y='y', color='display',
                        symbol='outlier',
                        labels={'x': f'{pc_x_col} ({explained_var[pc_x_idx]:.1%})',
                               'y': f'{pc_y_col} ({explained_var[pc_y_idx]:.1%})'}
                    )
                else:
                    scatter_fig = px.scatter(
                        plot_df, x='x', y='y', color='color',
                        labels={'x': f'{pc_x_col} ({explained_var[pc_x_idx]:.1%})',
                               'y': f'{pc_y_col} ({explained_var[pc_y_idx]:.1%})',
                               'color': color_label}
                    )
            else:
                if detect_outliers and outlier_action == 'show':
                    scatter_fig = px.scatter(
                        plot_df, x='x', y='y', color='outlier',
                        symbol='outlier',
                        labels={'x': f'{pc_x_col} ({explained_var[pc_x_idx]:.1%})',
                               'y': f'{pc_y_col} ({explained_var[pc_y_idx]:.1%})'}
                    )
                else:
                    scatter_fig = px.scatter(
                        plot_df, x='x', y='y',
                        labels={'x': f'{pc_x_col} ({explained_var[pc_x_idx]:.1%})',
                               'y': f'{pc_y_col} ({explained_var[pc_y_idx]:.1%})'}
                    )
            
            # Add scatter traces to subplot
            for trace in scatter_fig.data:
                fig.add_trace(trace, row=1, col=1)
            
            # Explained variance bar plot
            pc_range = list(range(1, min(len(explained_var) + 1, 21)))
            fig.add_trace(
                go.Bar(x=pc_range, y=explained_var[:len(pc_range)], 
                       name='Individual', showlegend=False),
                row=1, col=2
            )
            
            # Cumulative explained variance line
            cumulative_var = np.cumsum(explained_var[:len(pc_range)])
            fig.add_trace(
                go.Scatter(x=pc_range, y=cumulative_var, 
                          mode='lines+markers', name='Cumulative',
                          yaxis='y2', showlegend=False),
                row=1, col=2, secondary_y=True
            )
            
            # Hotelling's T² plot if outlier detection is enabled
            if detect_outliers:
                colors_t2 = ['red' if x else 'blue' for x in is_outlier]
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(outlier_info['t_squared']))),
                        y=outlier_info['t_squared'],
                        mode='markers',
                        marker=dict(color=colors_t2, size=6),
                        name='T² values',
                        showlegend=False,
                        text=[f"Sample {i}" for i in data.index],
                        hovertemplate='%{text}<br>T²: %{y:.2f}<extra></extra>'
                    ),
                    row=1, col=3
                )
                
                # Add critical value line
                fig.add_trace(
                    go.Scatter(
                        x=[0, len(outlier_info['t_squared']) - 1],
                        y=[outlier_info['critical_value'], outlier_info['critical_value']],
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        name='Critical value',
                        showlegend=False
                    ),
                    row=1, col=3
                )
            
            # Update layout
            fig.update_xaxes(title_text=f"{pc_x_col} ({explained_var[pc_x_idx]:.1%})", row=1, col=1)
            fig.update_yaxes(title_text=f"{pc_y_col} ({explained_var[pc_y_idx]:.1%})", row=1, col=1)
            fig.update_xaxes(title_text="Principal Component", row=1, col=2)
            fig.update_yaxes(title_text="Explained Variance Ratio", row=1, col=2)
            fig.update_yaxes(title_text="Cumulative Explained Variance", row=1, col=2, secondary_y=True)
            
            if detect_outliers:
                fig.update_xaxes(title_text="Sample Index", row=1, col=3)
                fig.update_yaxes(title_text="Hotelling's T²", row=1, col=3)
            
            fig.update_layout(
                height=600,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                )
            )
            
            # Save interactive plot to specified path
            try:
                fig.write_html(html_path)
                print(f"Interactive plot saved as '{html_path}'")
                
                # Try to show plot as well
                try:
                    fig.show()
                except Exception:
                    print(f"Could not display interactive plot directly, but saved to {html_path}")
                    
            except Exception as e:
                print(f"Could not save HTML file to '{html_path}'. Error: {e}")
                print("Falling back to matplotlib plots...")
                interactive = False
        
        except ImportError:
            print("Plotly not installed. Install with 'pip install plotly' for interactive plots.")
            print("Using matplotlib instead...")
            interactive = False
    
    if not interactive:
        # Create matplotlib plots with chosen PCs
        n_plots = 3 if detect_outliers else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(figsize[0] * n_plots / 2, figsize[1]))
        if n_plots == 2:
            axes = [axes[0], axes[1]]
        
        # Plot 1: Score plot
        ax1 = axes[0]
        
        if detect_outliers and outlier_action == 'show':
            # Plot normal points
            normal_mask = ~is_outlier
            if color_var is not None:
                unique_cats = color_var.unique()
                colors = plt.cm.Set1(np.linspace(0, 1, len(unique_cats)))
                
                for i, cat in enumerate(unique_cats):
                    mask = (color_var == cat) & normal_mask
                    if mask.sum() > 0:
                        ax1.scatter(scores_df.loc[mask, pc_x_col], scores_df.loc[mask, pc_y_col], 
                                   c=[colors[i]], label=str(cat), alpha=0.7, s=50)
                
                # Plot outliers with different marker
                outlier_mask = is_outlier
                for i, cat in enumerate(unique_cats):
                    mask = (color_var == cat) & outlier_mask
                    if mask.sum() > 0:
                        ax1.scatter(scores_df.loc[mask, pc_x_col], scores_df.loc[mask, pc_y_col], 
                                   c=[colors[i]], marker='x', s=100, linewidths=2, 
                                   label=f'{cat} (outlier)')
            else:
                ax1.scatter(scores_df.loc[normal_mask, pc_x_col], scores_df.loc[normal_mask, pc_y_col], 
                           alpha=0.7, s=50, label='Normal')
                ax1.scatter(scores_df.loc[is_outlier, pc_x_col], scores_df.loc[is_outlier, pc_y_col], 
                           c='red', marker='x', s=100, linewidths=2, label='Outlier')
            
            ax1.legend(title=color_label if color_var is not None else None, loc='best')
        elif detect_outliers and outlier_action == 'hide':
            # Hide outliers - only plot non-outliers
            normal_mask = ~is_outlier
            if color_var is not None:
                unique_cats = color_var[normal_mask].unique()
                colors = plt.cm.Set1(np.linspace(0, 1, len(unique_cats)))
                
                for i, cat in enumerate(unique_cats):
                    mask = (color_var == cat) & normal_mask
                    if mask.sum() > 0:
                        ax1.scatter(scores_df.loc[mask, pc_x_col], scores_df.loc[mask, pc_y_col], 
                                   c=[colors[i]], label=str(cat), alpha=0.7, s=50)
                
                ax1.legend(title=color_label, loc='best')
            else:
                ax1.scatter(scores_df.loc[normal_mask, pc_x_col], scores_df.loc[normal_mask, pc_y_col], 
                           alpha=0.7, s=50)
        else:
            # No outlier detection or outlier_action is something else
            if color_var is not None:
                unique_cats = color_var.unique()
                colors = plt.cm.Set1(np.linspace(0, 1, len(unique_cats)))
                
                for i, cat in enumerate(unique_cats):
                    mask = color_var == cat
                    ax1.scatter(scores_df.loc[mask, pc_x_col], scores_df.loc[mask, pc_y_col], 
                               c=[colors[i]], label=str(cat), alpha=0.7, s=50)
                
                ax1.legend(title=color_label, loc='best')
            else:
                ax1.scatter(scores_df[pc_x_col], scores_df[pc_y_col], alpha=0.7, s=50)
        
        ax1.set_xlabel(f'{pc_x_col} ({explained_var[pc_x_idx]:.1%} variance)')
        ax1.set_ylabel(f'{pc_y_col} ({explained_var[pc_y_idx]:.1%} variance)')
        ax1.set_title(f'PCA Score Plot ({pc_x_col} vs {pc_y_col})')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Explained variance
        ax2 = axes[1]
        pc_range = range(1, min(len(explained_var) + 1, 21))
        ax2.bar(pc_range, explained_var[:len(pc_range)], alpha=0.7)
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Explained Variance Ratio')
        ax2.set_title('Explained Variance by Component')
        ax2.grid(True, alpha=0.3)
        
        # Add cumulative explained variance line
        cumulative_var = np.cumsum(explained_var[:len(pc_range)])
        ax2_twin = ax2.twinx()
        ax2_twin.plot(pc_range, cumulative_var, 'ro-', alpha=0.7)
        ax2_twin.set_ylabel('Cumulative Explained Variance', color='red')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        
        # Plot 3: Hotelling's T² if enabled
        if detect_outliers:
            ax3 = axes[2]
            colors_t2 = ['red' if x else 'blue' for x in is_outlier]
            ax3.scatter(range(len(outlier_info['t_squared'])), outlier_info['t_squared'], 
                       c=colors_t2, alpha=0.7, s=50)
            ax3.axhline(y=outlier_info['critical_value'], color='red', linestyle='--', 
                       label=f"Critical value (α={outlier_alpha})")
            ax3.set_xlabel('Sample Index')
            ax3.set_ylabel("Hotelling's T²")
            ax3.set_title("Hotelling's T² Statistics")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Print summary statistics
    print("\nPCA Summary:")
    print(f"- Data shape: {data.shape}")
    print(f"- Number of components: {n_components}")
    print(f"- Plotted components: {pc_x_col} vs {pc_y_col}")
    print(f"- Explained variance ({pc_x_col}): {explained_var[pc_x_idx]:.1%}")
    print(f"- Explained variance ({pc_y_col}): {explained_var[pc_y_idx]:.1%}")
    print(f"- Total explained variance ({pc_x_col}+{pc_y_col}): {explained_var[pc_x_idx] + explained_var[pc_y_idx]:.1%}")
    
    if len(explained_var) > max(pc_x_idx, pc_y_idx):
        remaining_pcs = [i for i in range(min(3, len(explained_var))) if i not in [pc_x_idx, pc_y_idx]]
        if remaining_pcs:
            next_pc = remaining_pcs[0] + 1
            print(f"- Explained variance (PC{next_pc}): {explained_var[remaining_pcs[0]]:.1%}")
    
    # Return results
    results = {
        'scores': scores_df,
        'loadings': loadings,
        'explained_variance_ratio': explained_var,
        'pca_model': pca_model,
        'pipeline': pipeline,
        'color_variable': color_var if color_var is not None else None,
        'plotted_components': (pc_x, pc_y),
        'outlier_info': outlier_info
    }

    # Outlier removal metadata
    if outlier_action == 'remove' and outlier_info and outlier_info.get('n_outliers', 0) > 0:
        results['outlier_cats'] = categoricals  # already filtered earlier
        results['removed_outliers'] = outlier_info.get('removed_indices', [])
        results['n_outliers_removed'] = outlier_info.get('removed_count', 0)
        results['original_n_samples'] = outlier_info.get('removed_count', 0) + len(scores_df)

    # Always include categoricals (cleaned if outliers removed)
    if outlier_action == 'remove' and outlier_info and outlier_info.get('n_outliers', 0) > 0:
        results['categoricals'] = categoricals  # cleaned version
    else:
        results['categoricals'] = categoricals  # original version

    return results


#%%    
def plot_loadings(pca_results, n_components=[1, 2], figsize=(12, 6)): 
    """
    Plot loadings for specified principal components as line plots.
    
    Parameters:
    -----------
    pca_results : dict
        Results from perform_pca function
    n_components : list, default=[1, 2]
        List of component numbers to plot (1-indexed, e.g., [1, 2, 3])
    figsize : tuple, default=(12, 6)
        Figure size for the plot
    """
    loadings = pca_results['loadings']
    explained_var = pca_results['explained_variance_ratio']
    
    # Convert to 0-indexed and validate components
    pc_indices = [pc - 1 for pc in n_components]
    max_available = loadings.shape[1]
    
    # Filter out invalid components
    valid_indices = [i for i in pc_indices if 0 <= i < max_available]
    valid_components = [i + 1 for i in valid_indices]
    
    if not valid_indices:
        raise ValueError(f"No valid components found. Available components: 1 to {max_available}")
    
    if len(valid_indices) != len(pc_indices):
        print(f"Warning: Some components not available. Using components: {valid_components}")
    
    plt.figure(figsize=figsize)
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, pc_idx in enumerate(valid_indices):
        pc_name = f'PC{pc_idx + 1}'
        color = colors[i % len(colors)]
        
        plt.plot(range(len(loadings)), loadings.iloc[:, pc_idx], 
                'o-', color=color, 
                label=f'{pc_name} ({explained_var[pc_idx]:.1%})', 
                linewidth=2, markersize=3)
    
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.axhline(y=0, color= 'k', linestyle= '--')
    plt.xlabel('Wavenumber (1/cm)')
    plt.ylabel('Loading Value')
    plt.title('PCA Loadings Plot')
    
    # Only show legend if more than one component
    if len(valid_indices) > 1:
        plt.legend()
    
    plt.grid(True, alpha=0.3)
    
    # Use rounded numeric column names from data.columns as x-axis
    numeric_columns = [col for col in loadings.index if isinstance(col, (int, float))]
    if numeric_columns:
        # Select evenly spaced columns for clean x-axis
        n_ticks = min(20, len(loadings))  # Max 20 ticks
        step = max(1, len(loadings) // n_ticks)
        tick_positions = range(0, len(loadings), step)
        
        tick_labels = []
        for pos in tick_positions:
            if pos < len(loadings):
                col_name = loadings.index[pos]
                if isinstance(col_name, (int, float)):
                    tick_labels.append(f'{col_name:.0f}')
                else:
                    tick_labels.append(str(col_name)[:8])
        
        plt.xticks(tick_positions[:len(tick_labels)], tick_labels, rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Showing loadings for {len(loadings)} features across components: {valid_components}")
    return loadings.iloc[:, valid_indices]

#%%
def multi_loadings(pca_results, max_PC=5):
    for n,i in combinations(range(1, max_PC+1), r=2):
        plot_loadings(pca_results, n_components=[n, i], figsize=(12, 6))
        

#%%
def test_pca_statistics(pca_results, group_by=None, mean_by=None, 
                        pc_number=1, max_pc=None):
    """
    Create boxplots showing PC scores by groups with means and standard deviations.
    
    Parameters:
    -----------
    pca_results : dict
        Results from perform_pca function
    categoricals : pd.DataFrame, optional
        Categorical data for grouping. If None, uses pca_results['outlier_cats']
    group_by : str
        Primary factor to compare (e.g., 'Treatment', 'Day')
    mean_by : str, optional
        Factor to group by before calculating means (e.g., 'Chip', 'Subject').
        Creates pivot table structure with mean_by as rows, group_by as columns.
        Individual means are shown as points on the boxplot.
    pc_number : int, default=1
        Which PC to test (1-indexed). Ignored if max_pc is specified.
    max_pc : int, optional
        If specified, creates plots for all PCs from 1 to max_pc.
    
    Returns:
    --------
    dict or list of dicts containing:
        - group_means: Mean PC scores per group
        - group_std: Standard deviation per group
        - pivot_table: Pivot table if mean_by was used (None otherwise)
        - pc_number: Which PC was analyzed
    """
    
    if "outlier_cats" in pca_results and isinstance(pca_results["outlier_cats"], pd.DataFrame):
        categoricals = pca_results["outlier_cats"]
        print("✓ Using cleaned categoricals from pca_results['outlier_cats']")
    
    elif "categoricals" in pca_results and isinstance(pca_results["categoricals"], pd.DataFrame):
        categoricals = pca_results["categoricals"]
        print("✓ Using original categoricals from pca_results['categoricals']")
    
    else:
        raise ValueError(
            "pca_results does not contain 'outlier_cats' or 'categoricals'. "
            
        )
    
    # Validate row alignment with PCA scores
    if len(categoricals) != len(pca_results["scores"]):
        raise ValueError(
            f"Categoricals length ({len(categoricals)}) does not match "
            f"PCA score length ({len(pca_results['scores'])})."
    )
    
    # Multi-PC mode
    if max_pc is not None:
        results_list = []
        for i in range(1, max_pc + 1):
            print(f"\n{'='*70}")
            print(f"PC{i} Visualization")
            print(f"{'='*70}")
            result = test_pca_statistics(
                pca_results, group_by=group_by, mean_by=mean_by,
                pc_number=i, max_pc=None
            )
            results_list.append(result)
        
        return results_list
    
    # Single PC mode
    scores = pca_results['scores']
    pc_col = f'PC{pc_number}'
    
    if pc_col not in scores.columns:
        raise ValueError(f"{pc_col} not found in scores. Available: {list(scores.columns)}")
    
    if group_by not in categoricals.columns:
        raise ValueError(f"{group_by} not found in categoricals")
    
    # Prepare data
    if mean_by is not None:
        # PIVOT TABLE LOGIC
        if mean_by not in categoricals.columns:
            raise ValueError(f"{mean_by} not found in categoricals")
        
        test_data = pd.DataFrame({
            'PC_scores': scores[pc_col],
            'group_by': categoricals[group_by].astype(str),
            'mean_by': categoricals[mean_by].astype(str)
        })
        
        test_data = test_data.dropna()
        
        # Create pivot table structure
        pivot_table = test_data.pivot_table(
            values='PC_scores',
            index='mean_by',      # Rows: mean_by (e.g., Chips)
            columns='group_by',   # Columns: group_by (e.g., Days)
            aggfunc='mean'
        )
        
        print(f"\nPivot table structure ({mean_by} as rows, {group_by} as columns):")
        print(pivot_table)
        
        # Reshape for plotting: each column becomes a group
        test_data_final = pivot_table.melt(ignore_index=False, var_name='Group', value_name='PC_scores')
        test_data_final = test_data_final.reset_index()
        test_data_final = test_data_final.dropna()
        
        grouped_means = pivot_table.copy()
        
    else:
        # Individual observations mode (no pivot)
        test_data_final = pd.DataFrame({
            'PC_scores': scores[pc_col],
            'Group': categoricals[group_by].astype(str)
        })
        
        test_data_final = test_data_final.dropna()
        grouped_means = None
    
    # Calculate statistics
    groups = sorted(test_data_final['Group'].unique())
    group_data = [test_data_final[test_data_final['Group'] == group]['PC_scores'].values 
                  for group in groups]
    
    means = [np.mean(data) for data in group_data]
    stds = [np.std(data, ddof=1) for data in group_data]
    medians = [np.median(data) for data in group_data]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Track outlier counts for groups with many points
    max_points_threshold = 30
    outlier_counts = {}
    
    for i, group in enumerate(groups):
        x_pos = i + 1
        mean = means[i]
        std = stds[i]
        median = medians[i]
        group_vals = test_data_final[test_data_final['Group'] == group]['PC_scores'].values
        n_points = len(group_vals)
    
        # --- NEW STYLE: mean = red horizontal line ---
        ax.plot([x_pos - 0.2, x_pos + 0.2], [mean, mean],
                color='red', linewidth=3, zorder=5,
                label='Mean' if i == 0 else '')
    
        # --- NEW STYLE: median = diamond marker ---
        ax.plot(x_pos, median, marker='X', markersize=8,
                color='darkblue', markeredgecolor='darkblue', zorder=6,
                label='Median' if i == 0 else '')
    
        # Whiskers still represent mean ± SD
        ax.plot([x_pos, x_pos], [mean - std, mean + std],
                color='black', linewidth=1.5, zorder=4,
                label='Mean±SD' if i == 0 else '')
        ax.plot([x_pos - 0.1, x_pos + 0.1], [mean - std, mean - std],
                color='black', linewidth=1.5, zorder=4)
        ax.plot([x_pos - 0.1, x_pos + 0.1], [mean + std, mean + std],
                color='black', linewidth=1.5, zorder=4)
    
        # Points overlay
        if n_points <= max_points_threshold:
            x_positions = np.random.normal(x_pos, 0.04, size=n_points)
            ax.scatter(x_positions, group_vals, alpha=0.6, s=50,
                       color='darkblue', edgecolors='black', linewidth=0.5, zorder=3,
                       label='Data points' if i == 0 else '')
        else:
            outliers = np.sum((group_vals < mean - std) | (group_vals > mean + std))
            outlier_counts[group] = {'total': n_points, 'outside_whiskers': outliers}
            print(f"  Note: {group} has {n_points} points (>{max_points_threshold}), hiding individual points")
            print(f"        {outliers} points outside Mean±SD whiskers")
    
    # Set labels and title
    ax.set_ylabel(f'{pc_col} Scores', fontsize=12)
    ax.set_xlabel(group_by, fontsize=12)
    
    # Set x-axis tick positions and labels
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(groups, rotation=45, ha='right')
    
    title = f'{pc_col} Scores by {group_by}'
    if mean_by:
        title += f'\n(Pivot: {mean_by} as rows, points show individual means)'
    ax.set_title(title, fontsize=12, pad=20)
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicate labels
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', framealpha=0.9)
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n{pc_col} Summary Statistics:")
    print(f"Comparing: {group_by}")
    if mean_by:
        print(f"Structure: Pivot with {mean_by} as rows")
        print("⚠ IMPORTANT: Statistics below are calculated on the PIVOT TABLE VALUES")
        print(f"  - Mean = average of {mean_by} means within each {group_by}")
        print(f"  - SD = standard deviation of {mean_by} means within each {group_by}")
        print(f"  - n per group = number of {mean_by} levels")
        print(f"n per group: {[len(test_data_final[test_data_final['Group'] == g]) for g in groups]}")
    else:
        print(f"n per group: {[len(test_data_final[test_data_final['Group'] == g]) for g in groups]}")
    
    print("\nGroup Statistics:")
    for group, mean, std in zip(groups, means, stds):
        if mean_by:
            print(f"  {group}: Mean of {mean_by} means={mean:.4f}, SD of {mean_by} means={std:.4f}")
        else:
            print(f"  {group}: Mean={mean:.4f}, SD={std:.4f}")
    
    # Print outlier information if any groups were hidden
    if outlier_counts:
        print("\n⚠ Groups with hidden points (n > 30):")
        for group, counts in outlier_counts.items():
            print(f"  {group}: {counts['total']} total points, {counts['outside_whiskers']} outside Mean±SD")
    
    return {
        'group_means': pd.Series(means, index=groups, name=f'{pc_col}_mean'),
        'group_std': pd.Series(stds, index=groups, name=f'{pc_col}_std'),
        'pivot_table': grouped_means if mean_by else None,
        'pc_number': pc_number,
        'group_by': group_by,
        'mean_by': mean_by,
        'outlier_counts': outlier_counts if outlier_counts else None  # Track hidden groups
    }


#%%  
def export_data(pca_results: dict, 
                output_directory: str, scores: bool = True, 
                loadings: bool = True, ex_variance: bool = True, 
                filetype: str = "xlsx", filename: str = "pca_results"):
    """
    Export PCA results and associated data to Excel or Pickle files.
    
    Parameters
    ----------
    pca_results : dict
        Dictionary returned by perform_pca() (must contain 'categoricals')
    output_directory : str
        Folder where exported files will be saved
    scores : bool, default=True
        Export PCA score matrix
    loadings : bool, default=True
        Export PCA loadings
    ex_variance : bool, default=True
        Export explained variance ratios
    filetype : {'xlsx', 'pkl'}, default='xlsx'
        Output file type (Excel or Pickle)
    
    Notes
    -----
    - Categoricals are automatically extracted from pca_results
    - If outliers were removed, the cleaned categoricals are used
    - The function prioritizes 'outlier_cats' over 'categoricals' if both exist
    """
    
    # --- Safety and setup ---
    os.makedirs(output_directory, exist_ok=True)
    
    # Auto-extract categoricals from pca_results (prioritize cleaned version)
    if "outlier_cats" in pca_results and isinstance(pca_results["outlier_cats"], pd.DataFrame):
        cat_df = pca_results["outlier_cats"]
        print("✓ Using cleaned categoricals from pca_results['outlier_cats']")
    elif "categoricals" in pca_results and isinstance(pca_results["categoricals"], pd.DataFrame):
        cat_df = pca_results["categoricals"]
        print("✓ Using original categoricals from pca_results['categoricals']")
    else:
        print("Warning: No valid categoricals found in pca_results. Skipping categorical export.")
        cat_df = None
    
    # Validate that categoricals match scores length
    if cat_df is not None and "scores" in pca_results:
        if len(cat_df) != len(pca_results["scores"]):
            print(f"Warning: Categoricals length ({len(cat_df)}) doesn't match scores length ({len(pca_results['scores'])})")
            print("This shouldn't happen - categoricals should be auto-aligned by perform_pca()")
    
    # --- Prepare export content ---
    dfs_to_export = {}
    
    if scores and "scores" in pca_results:
        dfs_to_export["scores"] = pca_results["scores"]
    
    if loadings and "loadings" in pca_results:
        dfs_to_export["loadings"] = pca_results["loadings"]
    
    if ex_variance and "explained_variance_ratio" in pca_results:
        dfs_to_export["explained_variance"] = pd.DataFrame(
            pca_results["explained_variance_ratio"],
            columns=["explained_variance_ratio"]
        )
    
    if cat_df is not None:
        dfs_to_export["categoricals"] = cat_df
    
    if not dfs_to_export:
        print("No data selected for export. Nothing saved.")
        return
    
    # --- Export logic ---
    if filetype.lower() == "xlsx":
        output_path = os.path.join(output_directory, f"{filename}.xlsx")  # Use filename parameter
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for name, df in dfs_to_export.items():
                df.to_excel(writer, sheet_name=name[:31])
        print(f"✅ PCA results successfully exported to '{output_path}'")
    
    elif filetype.lower() in ["pkl", "pickle"]:
        for name, df in dfs_to_export.items():
            output_path = os.path.join(output_directory, f"{filename}_{name}.pkl")  # Use filename parameter
            df.to_pickle(output_path)
        print(f"✅ PCA results successfully exported as pickle files in '{output_directory}'")
    
    else:
        raise ValueError("filetype must be either 'xlsx' or 'pkl'")
    
    # Print summary of what was exported
    print("\nExported data sheets:")
    for name, df in dfs_to_export.items():
        print(f"  - {name}: {df.shape[0]} rows × {df.shape[1]} columns")
        
#%%
def export_stats(stats_results, output_directory: str, filename: str = "stats_results", 
                filetype: str = "xlsx"):
    """
    Export statistical summary results to Excel or Pickle files.
    
    IMPORTANT: When mean_by is used, the exported statistics are calculated on 
    the pivot table values (mean of means, SD of means), NOT the raw data.
    
    For example, if mean_by='Chip' and group_by='Day':
    - The 'Mean' column shows the average of all Chip means within each Day
    - The 'Std' column shows the SD of Chip means within each Day
    - This is the same as taking the column means/SDs from the pivot table
    
    Parameters
    ----------
    stats_results : dict or list
        Statistical results from test_pca_statistics
    output_directory : str
        Folder where exported files will be saved
    filename : str, default="stats_results"
        Base name for output file(s)
    filetype : {'xlsx', 'pkl'}, default='xlsx'
        Output file type (Excel or Pickle)
    """
    
    # Setup
    os.makedirs(output_directory, exist_ok=True)
    
    # Determine if single or multiple results
    is_multi = isinstance(stats_results, list)
    results_list = stats_results if is_multi else [stats_results]
    
    # Prepare summary DataFrame
    summary_data = []
    
    for result in results_list:
        pc_num = result.get('pc_number', 1)
        group_by = result.get('group_by', 'Unknown')
        mean_by = result.get('mean_by', None)
        
        # Add means and stds
        means = result.get('group_means', pd.Series())
        stds = result.get('group_std', pd.Series())
        
        for group in means.index:
            row = {
                'PC': f'PC{pc_num}',
                'Group_By': group_by,
                'Mean_By': mean_by if mean_by else 'N/A',
                'Group': group,
                'Mean': means.loc[group],
                'Std': stds.loc[group]
            }
            
            # Add note if mean_by was used
            if mean_by:
                row['Note'] = f'Mean/SD of {mean_by} means'
            else:
                row['Note'] = 'Raw data statistics'
            
            summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Prepare pivot tables (if available)
    pivot_data = {}
    for i, result in enumerate(results_list):
        if result.get('pivot_table') is not None:
            pc_num = result.get('pc_number', i + 1)
            pivot_data[f'PC{pc_num}_pivot'] = result['pivot_table']
    
    # Export logic
    dfs_to_export = {'summary': summary_df}
    dfs_to_export.update(pivot_data)
    
    if filetype.lower() == "xlsx":
        output_path = os.path.join(output_directory, f"{filename}.xlsx")
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for sheet_name, df in dfs_to_export.items():
                df.to_excel(writer, sheet_name=sheet_name[:31], index=(sheet_name != 'summary'))
        
        print(f"✅ Statistical results successfully exported to '{output_path}'")
        print(f"   Sheets included: {list(dfs_to_export.keys())}")
    
    elif filetype.lower() in ["pkl", "pickle"]:
        for name, df in dfs_to_export.items():
            output_path = os.path.join(output_directory, f"{filename}_{name}.pkl")
            df.to_pickle(output_path)
        
        print(f"✅ Statistical results successfully exported as pickle files in '{output_directory}'")
        print(f"   Files created: {[f'{filename}_{name}.pkl' for name in dfs_to_export.keys()]}")
    
    else:
        raise ValueError("filetype must be either 'xlsx' or 'pkl'")
    
    # Print summary to console
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(summary_df.to_string(index=False))
    
    # Print explanation if mean_by was used
    if any(r.get('mean_by') is not None for r in results_list):
        print("\n" + "="*70)
        print("⚠ IMPORTANT NOTE ON STATISTICS WITH mean_by:")
        print("="*70)
        print("When mean_by is specified, the statistics are calculated on pivot table values:")
        print("  - Mean = average of the means across mean_by levels")
        print("  - Std = standard deviation of the means across mean_by levels")
        print("\nExample: If mean_by='Chip' and group_by='Day':")
        print("  Day1 Mean = average of (Chip1_Day1_mean, Chip2_Day1_mean, ...)")
        print("  Day1 Std = SD of (Chip1_Day1_mean, Chip2_Day1_mean, ...)")
        print("\nTo verify: Check the pivot table sheets and calculate column means/SDs")
    
#%%    
# Example usage:
'''
# Extract data
categoricals, data = extract_data('your_file.pkl')

#crop data
data= crop_data(data, start= 400, end= 1800)

# Non-interactive PCA (default)
pca_results = perform_pca(data, categoricals, color_by='Component')

# Plot loadings for specific components
loadings_plot = plot_loadings(pca_results, n_components=[2, 3])

# Test PC1 significance
significance_results = test_pc_significance(pca_results, categoricals, 
                                          group_by='Component', pc_number=1)
'''