# MRI-Unsup

# Unsupervised MRI Reconstruction

When System Model meets Image Prior: An Unsupervised Deep Learning Architecture for Accelerated Magnetic Resonance Imaging
- 2023 Ibsa Jalata, University of Arkansas (ikjalata@uark.edu)

## Setup

Make sure the python requirements are installed

    pip3 install -r requirements.txt

## Dataset preparation

To begin, we need to download data and create sampling masks. The volumetric knee scans we will be using are fully sampled datasets obtained from mridata [2]. To accomplish this, we will utilize the BART binary with the setup script. To proceed, run the provided script in a new folder.:

    python mri_util/setup_mri.py -v

## Training

Use the following script to train the network:
    python3 main.py dataset_dir model_dir

where dataset_dir is the folder where the knee datasets were saved to,
and model_dir will be the top directory where the models will be saved to.


## Testing

Use the following script to test the network:
    python3 test.py dataset_dir model_dir

where dataset_dir is the folder where the knee datasets were saved to,
and model_dir will be the top directory where the models will be saved to.


## Questions/Issues

If you have any questions or encounter any problems, please don't hesitate to open an issue on the Github repository or reach out to us directly (ikjalata@uark.edu)
