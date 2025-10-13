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







