# KSL_PCA
A PCA python code that is tailored for Raman or other spectral data. The program can be run either directly in any python IDE such as spyder or jupyter notebook or via the GUI. In order for the program to run smoothly, please make sure that the headers in your excel file are non-numerical for categorical data and numerical (wavenumbers) for your Raman data.

#Python setup
1. Download anaconda (https://www.anaconda.com/download)
2. Open a command-line interface (e.g. CMD.exe Prompt ("Anaconda Prompt") or Poweshell Prompt)
3. If needed, create a new virtual environment by typing "conda create -n PCA_env python=3.10.11"
4. Activate the environment "conda activate PCA_env" (should then change from (base) to (PCA_env))
5. Install pip ("conda install pip")
6. Download and install the requirements.txt from the main page above (pip install -r /path/to/requirements.txt)

#Pull the program from github
1. Open Anaconda prompt in your virtual environment
2. Type cd path/to/desired/directory
3. Type "git clone https://github.com/EmanuelBehling/KSL_PCA.git"
4. A folder called KSL_PCA will be created at the given directory

#Recommended: create desktop shortcut for anaconda prompt
1. Right click on your desktop "New --> Shortcut"
2. Enter in the location field:
   C:\Windows\System32\cmd.exe /k "C:\Users\path_to_anaconda_3(<USER>)\anaconda3\Scripts\activate.bat C:\Users\path_to_anaconda\anaconda3 & conda activate my_env & cd /d C:\Projects\KSL_PCA"
3. Name your shortcut Anaconda Prompt (KSL_PCA)

#Start the program
1. Either open PCA_claude.py or PCA_GUI.py in your IDE and run it. PCA_claude is the coded version, so only recommended if you understand python basics.
2. Alternatively (recommended for python newcomers): open the desktop shortcut and just type "python PCA_GUI.py"

#Update the program
1. Run "git pull origin main" in your anaconda prompt
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#What the program does

#excel_to_pickle()
1. The program converts an excel file into a much faster .pkl file. This can take minutes if your excel file is big, but you only have to do it once. At the GUI the pickle file hast the same name as your excel file, while you can specify the name in the code.

#extract_data()
1. This function splits your data into numerical data (Raman measurements) and categorical data (e.g. Day, Condition...) by identifiying the data type of the headers. It should be non-numerical for categorical data and numerical (so the wavenumbers) for the Raman data.

#crop_data() (optional)
1. Crops the data at the given wavenumbers. If not stated, the whole spectra are taken for further analysis.

#perform_pca()
Performs a PCA with the given data and also colors the data point by a given categorical variable. You can choose the following options:
1. The number of components to calculate (standard is 10)
2. Whether you want to standardize the data (set scale_data= True)
3. For code users: wheter you want to create and save an interactive plot. The standard plots in spyder are not interactive so you can save it via the html_path input and open it in your webbrowser. GUI users that open the GUI with the prompt window will get an interactive pop-up image that they can directly safe. If you open the GUI with spyder you only can save the non-interative plot from spydre directly, so I recommend the Prompt option described above.
4. Wheter you want to detect outliers. If True, you can choose the alpha value (outlier_alpha) and three options: 'show' only shows the outliers in the graph, 'hide' shows the graph with outliers removed and 'remove' deletes the outliers from further analysis. In the code you can also specify until which PC outliers shall be removed.
5. You can color the scoreplot by a categorical variable of your choice via color_by.
6. You can also choose the PCs of your scoreplot via pc_x and pc_y

#plot_loadings()
Plots the loadings of the given PCs. You can plot all loading combinations until PC of your choice via the multi_loading() function (Plot multiple loadings at the GUI).

#test_pc_significance
Performs statistical tests for a given categorical variable, PC and alpha. The statistical test is either a t-test if the chosen categorical data has only two categories or an ANOVA if there are more. multi_stat does the same but for multiple PCs at once.


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




