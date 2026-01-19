## PIAP-Mamba: a physics-informed two-stage Mamba model for robust reconstruction of intracellular action potentials

### 🚀 Installation
```bash
#!/bin/bash
git clone https://github.com/xidanl/PIAP-Mamba
cd PIAP-Mamba

# Create conda environment
conda create -n piap_mamba python=3.9
conda activate piap_mamba

# Install necessary packages
pip install torch
pip install mamba_ssm casual_conv1d
pip install numpy transformers triton 
pip install scipy scikit-learn seaborn
```

### 🧪 Download data
```bash
# download the raw data
python data_downloader.py
```

### 🔨 How to Run
PIAP-Mamba training is performed in two stages. The main.py script provides a unified Trainer class designed for both phases. To specify the training stage (or to run evaluation), simply set the `step` parameter to either 'one' or 'two' and configure the corresponding arguments.
```python
# main.py

# define the trainer
trainer = PI2MambaTrainer(
        step='', 
        train_data_loc="path to training data",
        test_data_loc="path to test data",
        args=args
)

# train
trainer.train()

# eval: uncomment the belowing lines to start evaluate for your trained model
# model_ckpt_path = "path to checkpoint"
# trainer.evaluate(model_ckpt_path=model_ckpt_path, save_pic=True)
```

For results visualization, after evaluation, you can simply call the functions in `plotters.py`, such as `plot_samples()`, to visualize the results.