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




