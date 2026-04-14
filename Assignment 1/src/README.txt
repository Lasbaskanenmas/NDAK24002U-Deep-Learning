## Project Structure

This submission contains three Jupyter notebooks required for Assignment 1:

1. **DL_Backpropagation_pen_and_paper.ipynb**
2. **DL_AutoDiff_Nanograd.ipynb**
3. **feedforwardAssignment.ipynb**

Each notebook can be opened and executed independently inside a standard Python environment with Jupyter support.

---

## How to Run the Notebooks

### 1. Create and activate a Python environment

```bash
python3 -m venv dl_env
source dl_env/bin/activate        # macOS / Linux
dl_env\Scripts\activate           # Windows
```

### 2. Install required libraries

Install all dependencies used across the three notebooks:

```bash
pip install numpy
pip install matplotlib
pip install torch
pip install torchvision
pip install
pip install tqdm
pip install notebook
```

---

## Notebook-specific notes

### **1. DL_Backpropagation_pen_and_paper.ipynb**

This notebook contains our computations carried out manually, but also uses 'Image' from IPython.display to visualize some of the results.
**No special runtime requirements** beyond NumPy.

### **2. DL_AutoDiff_Nanograd.ipynb**

This notebook implements a tiny automatic differentiation engine (“Nanograd”).
Core libraries are needed:

* numpy
* matplotlib (for simple visualization)
* networkx

Note that the results in Exercise 2.l will change slightly due to the nature of the data-generating function if you decide to re-run it.

### **3. feedforwardAssignment.ipynb**

This is the main PyTorch implementation used for:

* noisy-xor data generation
* multiple neural network configurations (depth/width sweep)
* training loop with Adam optimizer
* visualization of decision boundaries
* MLOps logging (W&B)

To reproduce all results:

1. Ensure PyTorch and Matplotlib are installed.
2. Run all cells sequentially in the notebook.
3. If MLOps logging is enabled, log in to W&B before execution:

   ```bash
   wandb login
   ```

Again; note that the results in Exercise 3.4 might change slightly due to the nature of the data-generating function if you decide to re-run it.
However, the convergence should result in the same as the gif in src file. 

Furthermore the results saved in Weights & Biases might also differ slightly when running this, but it should result to the same as in the notebook.

---

## Summary of Required Libraries

The full list of libraries used across the project:

* numpy
* matplotlib
* torch
* torchvision
* tqdm
* wandb
* notebook / jupyter

All libraries are available through `pip` and require no system-specific compilation.
