Assignment 4 – Notebook Setup and Task Notes

Environment Setup
- All notebooks were run using the conda environment specified in environment.yml.
- On macOS and Linux, create and activate the environment with:
	conda env create -f environment.yml
	conda activate dl-assignment4
- On Windows, some packages may require alternatives or extra steps (e.g., torch + CUDA/MPS backends, plotting backends). If environment creation fails, install platform-compatible wheels for PyTorch and re-run conda install for the remaining dependencies.

Task 1 (MNIST – wandb logging)
- Training runs and generated results are logged in the wandb/ folder. Use these runs to review the best-performing configurations and sample outputs for MNIST.
- Each run directory contains its config, summary, and media artifacts; consult wandb-summary.json and media/ for quick inspection.

Task 2 & Task 3 (ODE/SDE Notebooks)
- The notebooks required minimal changes beyond the instructions in the assignment description; the core classes and functions were already implemented.
- Results were obtained by changing setup and parameters (e.g., network variants, integration choices, and hyperparameters). Not all intermediate results were stored.

Notebook Locations
- ODE: ODE.ipynb
- SDE: SDE.ipynb
- Hierarchical VAE: hierarchical_VAE.ipynb

Notes
- Ensure your GPU/MPS configuration aligns with the device selection code in each notebook (CUDA on Linux/Windows, MPS on macOS where available; otherwise CPU).
- For reproducibility, prefer fixed seeds and consistent batch sizes; refer to the wandb runs to mirror the best configurations.
