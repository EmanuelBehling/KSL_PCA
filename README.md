🧠 KSL_PCA

A Principal Component Analysis (PCA) tool tailored for Raman and other spectral data.
Run it in any Python IDE (Spyder, Jupyter) or via the included GUI.

Important: Make sure your Excel headers follow this rule:

Categorical columns → non-numeric headers (text)

Spectral / Raman columns → numeric headers (wavenumbers)

🔖 Table of Contents

Quick start

Prerequisites

Clone & install

Run the GUI or script

Usage examples

Convert Excel to pickle

Perform PCA

Plot loadings

Test PC significance

Functions overview

Tips & best practices

Development & updating

License & citation

Contact / Issues

Quick start
Prerequisites

Anaconda (recommended)

Python 3.10+ (environment example uses 3.10.11)

requirements.txt — make sure it sits at the repository root.

If your repo does not contain requirements.txt, create one listing the Python packages your code uses (e.g. numpy, pandas, scikit-learn, plotly, etc.).

Clone & install
# choose a working directory
cd /path/to/desired/directory

# clone the repo
git clone https://github.com/EmanuelBehling/KSL_PCA.git
cd KSL_PCA

# create and activate conda environment (recommended)
conda create -n PCA_env python=3.10.11 -y
conda activate PCA_env

# install pip (if needed) and requirements
conda install pip -y
pip install -r requirements.txt

Run the GUI or script

Via Anaconda Prompt / Terminal:

# launch GUI (recommended for non-coders)
python PCA_GUI.py

# OR launch the script version (for code users)
python PCA_claude.py


Via IDE (Spyder / Jupyter):

Open PCA_GUI.py or PCA_claude.py and run inside the IDE.

Usage examples
Convert Excel to pickle
from ksl_pca import excel_to_pickle

# Example usage
excel_to_pickle('data/my_raman_data.xlsx')
# This will write data/my_raman_data.pkl (or similar), used for faster loading later

Perform PCA
from ksl_pca import perform_pca

# Example usage
perform_pca(
    df,                     # DataFrame with spectral and categorical columns
    n_components=10,        # number of PCs to compute (default 10)
    scale_data=True,        # whether to standardize spectral data
    color_by='Condition',   # categorical column to color score plot by
    pc_x=1,                 # x-axis PC number (1-indexed)
    pc_y=2,                 # y-axis PC number (1-indexed)
    outlier_detection='show',
    outlier_alpha=0.05,
    html_path='results/PCA_interactive.html'  # optional interactive plot
)


Tip: Use html_path to save an interactive HTML plot (Plotly). Opening GUI via Anaconda Prompt yields interactive pop-ups you can save directly.

Plot loadings
from ksl_pca import plot_loadings, multi_loading

# Single loading plot
plot_loadings(pca_model, pc_x=1, pc_y=2)

# Multiple loading combinations up to PC 5
multi_loading(pca_model, max_pc=5)

Test PC significance
from ksl_pca import test_pc_significance, multi_stat

# Test a single PC across categories (t-test if 2 groups, ANOVA otherwise)
test_pc_significance(df, pc=1, group='Condition', alpha=0.05)

# Test multiple PCs
multi_stat(df, group='Condition', max_pc=5, alpha=0.05)

Functions overview

excel_to_pickle(path, output_path=None) — Convert Excel → Pickle for faster loads.

extract_data(df) — Split DataFrame into spectral (numeric headers) and categorical (non-numeric headers).

crop_data(data, lower_bound=None, upper_bound=None) — Crop spectral range (wavenumbers).

perform_pca(...) — Run PCA, produce score plots; supports scaling, outlier detection, interactive HTML export.

plot_loadings(pca_model, ...) — Visualize PC loadings.

multi_loading(pca_model, max_pc=...) — Plot multiple loading pairs.

test_pc_significance(df, pc, group, alpha=0.05) — t-test or ANOVA for a PC vs group.

multi_stat(df, group, max_pc, alpha=0.05) — Run significance tests across multiple PCs.

(See inline docstrings in each function for parameter details.)

Tips & best practices

Keep categorical headers text-only (e.g., Condition, Day, SampleID).

Use numerical headers for the Raman wavenumbers (e.g., 600, 601, ..., 1800).

Save heavy Excel inputs to .pkl using excel_to_pickle() to speed up repeated analyses.

Use Anaconda Prompt to run the GUI for interactive (Plotly) pop-ups.

If you wish to version-control processed outputs, commit only small files — don't commit large .pkl or result folders unless necessary.

Development & updating
# when working on the repo and want latest changes
git pull origin main


If you plan to contribute, please open a pull request. Add tests and update requirements.txt when adding dependencies.

License & citation

KSL_PCA is distributed under the MIT License
.
If you use KSL_PCA in a publication, please cite the repository.

Contact & Issues

Open an issue on GitHub:
https://github.com/EmanuelBehling/KSL_PCA/issues


# KSL_PCA
🧠 KSL_PCA
A PCA tool tailored for Raman and other spectral data.

KSL_PCA can be run either directly in a Python IDE (such as Spyder or Jupyter Notebook) or through its built-in GUI.
To ensure smooth performance, make sure that your Excel headers follow this rule:

Categorical data: Non-numerical headers

Spectral (Raman) data: Numerical headers (wavenumbers)

⚙️ Python Setup
1. Install Anaconda

Download and install from 👉 https://www.anaconda.com/download

2. Create a Virtual Environment
conda create -n PCA_env python=3.10.11
conda activate PCA_env

3. Install Requirements
conda install pip
pip install -r /path/to/requirements.txt

⬇️ Clone the Repository

Open Anaconda Prompt (in your virtual environment)

Navigate to your target directory:

cd path/to/desired/directory


Clone the repository:

git clone https://github.com/EmanuelBehling/KSL_PCA.git


A folder named KSL_PCA will be created.

🖥️ (Optional) Create a Desktop Shortcut

Right-click on your desktop → New → Shortcut

In the location field, paste:

C:\Windows\System32\cmd.exe /k "C:\Users\<USER>\anaconda3\Scripts\activate.bat C:\Users\<USER>\anaconda3 & conda activate PCA_env & cd /d C:\Projects\KSL_PCA"


Name it something like Anaconda Prompt (KSL_PCA)

This allows you to open the environment and project folder with one click.

🚀 Start the Program

You can start KSL_PCA in two ways:

Option 1 – Run via IDE

Open one of the following in Spyder or Jupyter Notebook:

PCA_claude.py → for users comfortable with code

PCA_GUI.py → recommended for beginners

Option 2 – Run via GUI

Open your custom Anaconda Prompt (KSL_PCA) shortcut

Type:

python PCA_GUI.py

🔁 Update the Program

To get the latest version:

git pull origin main

🧩 What the Program Does
🧾 excel_to_pickle()

Converts an Excel file into a faster .pkl (pickle) file

Large Excel files may take minutes to process — but this step is only needed once

💡 In the GUI, the pickle file keeps the same name as your Excel file.

🔍 extract_data()

Splits data into:

Categorical (e.g., Day, Condition…) → Non-numerical headers

Spectral (Raman measurements) → Numerical headers (wavenumbers)

✂️ crop_data() (optional)

Crops spectra to a specified wavenumber range.
If not specified, the entire spectrum is used.

📊 perform_pca()

Performs Principal Component Analysis (PCA) on your data and visualizes results.

Options:
Parameter	Description
n_components	Number of principal components (default: 10)
scale_data	Standardize data before PCA (True recommended)
color_by	Color the score plot by a chosen categorical variable
pc_x, pc_y	Choose which PCs to display in the score plot
outlier_detection	Detect, show, or remove outliers with chosen alpha
html_path	Save an interactive HTML plot for exploration

💡 GUI users launched via Anaconda Prompt get interactive pop-ups that can be saved directly.

📈 plot_loadings()

Plots the loadings for chosen PCs.
Use multi_loading() to visualize multiple loading combinations up to a chosen PC.

🧮 test_pc_significance()

Performs statistical tests (t-test or ANOVA) for a given categorical variable, PC, and α value.

t-test: used when there are two categories

ANOVA: used when there are more than two categories

Use multi_stat() to test multiple PCs at once.

🧠 Summary
Function	Purpose
excel_to_pickle()	Convert Excel to Pickle format
extract_data()	Separate Raman and categorical data
crop_data()	Crop data by wavenumber
perform_pca()	Run PCA and visualize results
plot_loadings()	Visualize PC loadings
test_pc_significance()	Statistical testing of PCs
📚 Tips

Always ensure Excel headers follow the categorical = text, spectral = numbers rule.

Use Anaconda Prompt (KSL_PCA) for interactive GUI features.

Save and reuse .pkl files for faster loading.




