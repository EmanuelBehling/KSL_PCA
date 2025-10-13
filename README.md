# KSL_PCA
A PCA that is tailored for Raman Data. The PCA code includes a number of functions that will be described later. The program can be run either directly in any python IDE such as spyder or jupyter notebook or via the GUI. In oreder for the program to run smoothly, please make sure that the headers in your excel file are non-numerical for categorical data and numerical (wavenumbers) for your Raman data.

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





