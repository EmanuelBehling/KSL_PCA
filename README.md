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
3. Install Requirements
bash
Copy code
conda install pip
pip install -r requirements.txt
⬇️ Clone the Repository
bash
Copy code
cd path/to/desired/directory
git clone https://github.com/EmanuelBehling/KSL_PCA.git
cd KSL_PCA
▶️ Start the Program
Option 1 – Run via IDE
bash
Copy code
# Run GUI (recommended)
python PCA_GUI.py

# OR run the script version
python PCA_claude.py
Option 2 – Run via Desktop Shortcut
Create a Windows shortcut with the following target (adjust paths):

bash
Copy code
C:\Windows\System32\cmd.exe /k "C:\Users\<USER>\anaconda3\Scripts\activate.bat C:\Users\<USER>\anaconda3 & conda activate PCA_env & cd /d C:\Projects\KSL_PCA"
Then open the shortcut and type:

bash
Copy code
python PCA_GUI.py
🔁 Update the Program
bash
Copy code
git pull origin main
