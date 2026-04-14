## Project Structure

This submission contains three Jupyter notebooks required for Assignment 3:

1. **sam_lung_xrays_1.ipynb** (Task 2)
2. **fine-tuning-language-model.ipynb** (Task 3)
3. **cot.ipynb** (Task 4)

The notebooks can be opened and executed independently, but each requires different setup configurations as described below.

---

## How to Run the Notebooks

### Task 2: sam_lung_xrays_1.ipynb

#### 1. Create and activate a Python environment

```bash
python3 -m venv dl_env
source dl_env/bin/activate        # macOS / Linux
dl_env\Scripts\activate           # Windows
```

#### 2. Install required libraries

```bash
pip install numpy
pip install matplotlib
pip install torch
pip install torchvision
pip install scikit-learn
pip install tqdm
pip install segment-anything-py
```

---

### Task 3: fine-tuning-language-model.ipynb

**⚠️ IMPORTANT: This notebook requires significant GPU resources and should be run on a cloud platform with GPU support such as Kaggle, Google Colab, or another virtual machine with substantial GPU power. Running this notebook locally on a standard machine may not be feasible due to computational requirements.**

#### 1. Upload notebook to cloud platform

Upload the `fine-tuning-language-model.ipynb` notebook to one of the following platforms:
- **Kaggle**: Create a new notebook and upload the file
- **Google Colab**: Upload the notebook to Google Drive and open with Colab
- **Other GPU-enabled VM**: Transfer the notebook to your virtual machine

#### 2. Install required libraries

On the cloud platform, install the following dependencies:

```bash
pip install numpy
pip install matplotlib
pip install torch
pip install transformers
pip install datasets
pip install pandas
pip install nltk
pip install wandb
pip install tqdm
```

#### 3. Enable GPU acceleration

- **Kaggle**: Enable GPU in the notebook settings (Settings → Accelerator → GPU)
- **Google Colab**: Runtime → Change runtime type → Hardware accelerator → GPU
- **VM**: Ensure CUDA-enabled PyTorch is installed and GPU drivers are configured

---

### Task 4: cot.ipynb

#### 1. Create and activate a Python environment

```bash
python3 -m venv dl_env
source dl_env/bin/activate        # macOS / Linux
dl_env\Scripts\activate           # Windows
```

#### 2. Install required libraries

Install the following packages with the specified versions (compatible with Python 3.11.11):

```bash
pip install torch==2.7.1
pip install transformers==4.31.0
pip install datasets==3.6.0
pip install tqdm==4.67.1
pip install matplotlib==3.10.3
pip install numpy==1.25.2
```

**Note:** If you encounter compatibility issues, you may need to adjust the versions slightly, but the above versions have been tested and verified to work together.

---