# Deep Learning Assignment 2 - MNIST Classification and Autoencoders

This repository contains the implementation and answers for Deep Learning Assignment 2, focusing on MNIST digit classification using feedforward networks, CNNs, autoencoders, and variational autoencoders (VAEs).

## Repository Structure

- **`feedForwardMNIST.ipynb`**: Feedforward neural network implementation for MNIST classification
- **`CNNMNIST.ipynb`**: Convolutional neural network implementation for MNIST classification
- **`autoencoders.ipynb`**: Autoencoder implementation and analysis
- **`autoencoders_test.ipynb`**: Additional autoencoder experiments
- **`vae.ipynb`**: Variational Autoencoder (VAE) implementation
- **`data/`**: Directory containing MNIST dataset
- **`wandb/`**: Weights & Biases experiment tracking logs

## Finding Answers to Assignment Questions

### Task 1: Feedforward Neural Networks

#### **Task 1.1: Feedforward Classification**
**Location**: `feedForwardMNIST.ipynb` - Section "**Answers to Task 1.1**"

This section contains answers to:
1. **Brief description of each part of the program** - Explains imports, device configuration, functions (`training_loop`, `evaluate_model`), and main program components
2. **What does the `transform` function achieve?** - Explains `ToTensor()` and `Normalize()` transformations
3. **How many parameters does this model use?** - Total: 109,386 parameters with detailed breakdown by layer
4. **How well is it able to correctly classify each digit class 0, 1, ..., 9?** - Overall accuracy: 94%, with per-class accuracies listed

#### **Task 1.2: Loss Function Analysis**
**Location**: `feedForwardMNIST.ipynb` - Section "**Answers to Task 1.2**"

This section discusses:
- Current loss function (`NLLLoss` with `LogSoftmax`)
- Alternative loss function with logits (`CrossEntropyLoss`)
- Numerical stability considerations and gradient flow advantages

#### **Task 1.3: Convolutional Neural Networks**
**Location**: `CNNMNIST.ipynb` - Section "**Answers to Task 1.3**"

This section contains answers to:
1. **Why must the Linear layer be preceded by `Flatten` and why does the Linear layer have 800 input features?** - Explains tensor reshaping and dimension calculation (32 channels × 5 × 5 = 800)
2. **Number of parameters** - Total: 12,810 parameters with detailed breakdown by layer
3. **How well is it able to correctly classify each digit class 0, 1, ..., 9?** - Overall accuracy: 97%, with per-class accuracies listed

### Task 2: Autoencoders

#### **Task 2.1: Autoencoder Implementation**
**Location**: `autoencoders.ipynb` - Section "**Answers to Task 2.1**"

This section contains:
- **2.1.1 Autoencoder Program Walkthrough** - Detailed explanation of the program structure, functions, and workflow
- **2.1.2 The transform function** - Explanation of data preprocessing, model definition, training & loss, and alternative likelihood (Beta distribution)

**Additional Analysis**:
- Loss curve visualization (saved as `loss_curve_*.png`)
- Latent space visualization (saved as `latent_*.pdf`)
- Reconstruction grid visualization (saved as `reconstruction_*.pdf`)
- PCA comparison plots (saved as `latent_pca*.png` and `reconstruction_pca*.png`)
- Reconstruction error vs. latent dimension analysis (saved as `reconstruction_error_vs_latent_dim.png`)

#### **Task 2.5: Variational Autoencoder (VAE)**
**Location**: `vae.ipynb` - Section "**Answers to questions 2.5**"

This section explains:
- VAE architecture and workflow
- Functions: `VAE` class, `encode`, `reparameterize`, `decode`, `train`
- Transform function explanation
- Model definition (probabilistic encoder vs. deterministic autoencoder)
- Loss function: ELBO derivation, BCE reconstruction loss, and KL divergence term

**Generated Outputs**:
- Latent space visualizations
- Generated samples from the VAE (saved as `generated_run_*.pdf`)

#### **Task 2.6: VAE Analysis and Comparison**
**Location**: `vae.ipynb` - Check for additional analysis sections or comparison with autoencoders

This task typically covers:
- Comparison between standard autoencoders and variational autoencoders
- Analysis of the latent space structure and continuity
- Discussion of generation capabilities and interpolation in latent space
- Comparison of reconstruction quality between autoencoders and VAEs
- Analysis of the KL divergence term's effect on the latent space

**Note**: If Task 2.6 answers are not explicitly labeled in the notebook, they may be found in:
- The VAE notebook after the Task 2.5 section
- Comparison discussions between `autoencoders.ipynb` and `vae.ipynb` results
- Analysis of the generated samples and latent space visualizations

## Running the Notebooks

1. **Environment Setup**: 
   - Create a conda environment: `conda create -n adl python=3.9`
   - Activate: `conda activate adl`
   - Install dependencies: `conda install ipykernel torch matplotlib torchmetrics scikit-image jpeg`

2. **Data**: The MNIST dataset will be automatically downloaded when running the notebooks (saved to `data/MNIST/` or `MNIST/` directory)

3. **Execution**: Run the notebooks in order:
   - Start with `feedForwardMNIST.ipynb` for Task 1.1 and 1.2
   - Then `CNNMNIST.ipynb` for Task 1.3
   - Then `autoencoders.ipynb` for Task 2.1
   - Finally `vae.ipynb` for Task 2.5 and 2.6

## Generated Files

The notebooks generate several output files:
- **Loss curves**: `loss_curve_1.png`, `loss_curve_2.png`, `loss_curve_3.png`
- **Latent space visualizations**: `latent_1.pdf`, `latent_2.pdf`, `latent_3.pdf`, `latent_pca.png`
- **Reconstruction visualizations**: `reconstruction_1.pdf`, `reconstruction_2.pdf`, `reconstruction_3.pdf`, `reconstruction_pca_1.png`, `reconstruction_pca_2.png`
- **VAE generated samples**: `generated_run_1.pdf`, `generated_run_2.pdf`, `generated_run_3.pdf`
- **Analysis plots**: `reconstruction_error_vs_latent_dim.png`

## Notes

- All notebooks use Weights & Biases (wandb) for experiment tracking. Make sure to log in with `wandb.login()` before running.
- The notebooks are designed to run on both CPU and GPU (automatically detects available device).
- Some experiments may take significant time depending on your hardware (especially autoencoder and VAE training).
