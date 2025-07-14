# Genome Organization Simulation Framework

A comprehensive framework for simulating and analyzing 3D genome organization using coarse-grained molecular dynamics. This repository provides tools for chromatin simulation, contact map analysis, and Hi-C data processing.

## 🚀 Quick Start

### Prerequisites

Before you begin, ensure you have the following installed:
- **Git** (for cloning the repository)
- **Mamba** (for package management) - recommended over conda for faster installation

### Installation Guide

#### Step 1: Clone the Repository

**For Mac/Linux:**
```bash
git clone https://github.com/yourusername/genome_organization.git
cd genome_organization
```

**For Windows:**
```cmd
git clone https://github.com/yourusername/genome_organization.git
cd genome_organization
```

#### Step 2: Install Mamba (if not already installed)

**For Mac/Linux:**
```bash
# Download and install Miniforge (includes mamba)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
# Restart your terminal or run: source ~/.bashrc
```

**For Windows:**
```cmd
# Download Miniforge from: https://github.com/conda-forge/miniforge/releases
# Run the installer and follow the prompts
# Restart your command prompt
```

#### Step 3: Create a Virtual Environment

**For Mac/Linux:**
```bash
mamba create -n genome_sim python=3.10
mamba activate genome_sim
```

**For Windows:**
```cmd
mamba create -n genome_sim python=3.10
mamba activate genome_sim
```

#### Step 4: Install Dependencies

**Install core dependencies:**
```bash
# Install basic scientific computing packages
mamba install -c conda-forge numpy scipy pandas matplotlib seaborn jupyter

# Install OpenMM for molecular dynamics
mamba install -c conda-forge openmm

# Install additional dependencies
mamba install -c conda-forge h5py tqdm requests
```

**Install the package in development mode:**
```bash
pip install -e .
```

#### Step 5: Verify Installation

```bash
python -c "import numpy, scipy, pandas, openmm; print('All packages installed successfully!')"
```

## 📁 Repository Structure

