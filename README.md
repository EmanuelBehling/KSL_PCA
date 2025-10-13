# 🧠 KSL_PCA

A Principal Component Analysis (PCA) tool tailored for **Raman** and other spectral data.  
It can be run in any Python IDE (e.g. Spyder, Jupyter Notebook) or via the included GUI.

> **Note:**  
> - **Categorical data** → non-numerical headers  
> - **Spectral (Raman) data** → numerical headers (wavenumbers)

---

## ⚙️ Installation

### 1. Install Anaconda  
Download here: [https://www.anaconda.com/download](https://www.anaconda.com/download)

### 2. Create a Virtual Environment
```bash
conda create -n PCA_env python=3.10.11
conda activate PCA_env

# 🧠 KSL_PCA

A Principal Component Analysis (PCA) tool tailored for **Raman** and other spectral data.  
You can run it directly in a Python IDE (e.g., Spyder, Jupyter Notebook) or through its built-in GUI.

> **Note:**  
> - **Categorical data** → non-numerical headers (text)  
> - **Spectral (Raman) data** → numerical headers (wavenumbers)

---

## ⚙️ Installation

### 1. Install Anaconda  
Download here: [https://www.anaconda.com/download](https://www.anaconda.com/download)

### 2. Create a Virtual Environment
```bash
conda create -n PCA_env python=3.10.11
conda activate PCA_env
```

### 3. Install Requirements
```bash
conda install pip
pip install -r requirements.txt
```

---

## ⬇️ Clone the Repository
```bash
cd path/to/desired/directory
git clone https://github.com/EmanuelBehling/KSL_PCA.git
cd KSL_PCA
```

---

## 🖥️ (Optional) Create a Desktop Shortcut

1. Right-click on your desktop → **New → Shortcut**  
2. In the *location* field, paste and adjust paths:
   ```
   C:\Windows\System32\cmd.exe /k "C:\Users\<USER>\anaconda3\Scripts\activate.bat C:\Users\<USER>\anaconda3 & conda activate PCA_env & cd /d C:\Projects\KSL_PCA"
   ```
3. Name it **Anaconda Prompt (KSL_PCA)**.  
4. You can now open your project and environment with one click.

---

## ▶️ Start the Program

### Option 1 – Run via IDE
Open the file in Spyder, Jupyter, or another IDE:
```bash
# Run GUI (recommended)
python PCA_GUI.py

# OR run the code-based version
python PCA_claude.py
```

### Option 2 – Run via the Shortcut  
Open your **Anaconda Prompt (KSL_PCA)** and type:
```bash
python PCA_GUI.py
```

---

## 🔁 Update the Program
To get the latest version:
```bash
git pull origin main
```

---

## 🧩 Program Overview

### `excel_to_pickle()`
Converts an Excel file into a faster `.pkl` (pickle) file.  
Large files may take a few minutes, but you only need to do this once.

---

### `extract_data()`
Splits your dataset into:
- **Categorical data** (non-numerical headers)
- **Spectral data** (numerical headers, e.g., wavenumbers)

---

### `crop_data()` *(optional)*
Crops your spectra between specified wavenumbers.  
If not specified, the full spectrum is used.

```python
crop_data(data, lower_bound=600, upper_bound=1800)
```

---

### `perform_pca()`
Performs PCA and visualizes the results.

**Main options:**

| Argument | Description |
|-----------|-------------|
| `n_components` | Number of components to compute (default = 10) |
| `scale_data` | Standardize data (`True` recommended) |
| `color_by` | Column name to color the plot by |
| `pc_x`, `pc_y` | Which PCs to plot |
| `outlier_detection` | `'show'`, `'hide'`, or `'remove'` |
| `outlier_alpha` | Alpha value for outlier detection |
| `html_path` | Save an interactive plot as HTML |

**Example:**
```python
perform_pca(
    data=df,
    n_components=10,
    scale_data=True,
    color_by='Condition',
    pc_x=1,
    pc_y=2,
    outlier_detection='show',
    outlier_alpha=0.05,
    html_path='results/PCA_plot.html'
)
```

> 💡 Launching via Anaconda Prompt provides interactive pop-ups that you can save directly.

---

### `plot_loadings()`
Plots loadings for specified PCs.

```python
plot_loadings(pca_model, pc_x=1, pc_y=2)
```

Plot multiple loading combinations:
```python
multi_loading(pca_model, max_pc=5)
```

---

### `test_pc_significance()`
Performs statistical tests on PCs.

- **t-test** → 2 groups  
- **ANOVA** → more than 2 groups

```python
test_pc_significance(df, pc=1, group='Condition', alpha=0.05)
multi_stat(df, group='Condition', max_pc=5, alpha=0.05)
```

---

## 🧠 Function Summary

| Function | Purpose |
|-----------|----------|
| `excel_to_pickle()` | Convert Excel → Pickle for faster loading |
| `extract_data()` | Split Raman and categorical data |
| `crop_data()` | Crop spectra by wavenumber |
| `perform_pca()` | Run PCA and visualize |
| `plot_loadings()` | Plot loadings |
| `test_pc_significance()` | Run t-tests / ANOVA |

---

## 💡 Tips
- Categorical headers = **text**  
- Raman wavenumbers = **numbers**  
- Use `.pkl` files for speed  
- Use the **Anaconda Prompt** for interactive GUI  
- Don’t commit large output files to GitHub

---

## 🧭 Update the Repository
```bash
git pull origin main
```

