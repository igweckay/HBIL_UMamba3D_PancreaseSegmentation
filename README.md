# HBIL_UMamba3D_PancreaseSegmentation

This repository requires a specific Python environment to run. Follow the steps below to install Miniconda and set up the environment.

## Prerequisites

- Ensure you have access to a terminal (Linux/Mac) or Command Prompt/PowerShell (Windows).
- Download Miniconda from [Miniconda's official website](https://docs.conda.io/en/latest/miniconda.html).

## Installation Steps

### 1. Install Miniconda

1. Download the appropriate Miniconda installer for your operating system.
2. Run the installer and follow the on-screen instructions.
3. Verify the installation by running:
    ```bash
    conda --version
    ```

### 2. Download Repository and create the Environment

1. Clone this repository
    ```bash
    git clone https://github.com/igweckay/HBIL_UMamba3D_PancreaseSegmentation.git
    ```
2. Navigate to the repository directory:
    ```bash
    cd HBIL_UMamba3D_PancreaseSegmentation
    ```
3. Create a new environment Named `RevMed`:
    ```bash
    conda env create -n ReMed python=3.12
    ```
4. Activate the environment:
    ```bash
    conda activate RevMed
    ```

### 3. Install Dependencies

To install dependencies, use:
```bash
pip install -r requirements.txt
```

### 4. Verify the Setup

Run the following command to ensure everything is working:
```bash
python inferer.py -i ./TestSample/Image.nii.gz -o ./TestSample/Segment.nii.gz
```
