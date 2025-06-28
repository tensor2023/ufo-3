# UFO-3: Unsupervised Three-Compartment Learning for fODF Estimation

📄 **Accepted at MICCAI 2025**  
🧠 _An unsupervised framework for estimating the fiber Orientation Distribution Function (fODF) from single-shell diffusion MRI._

This repository hosts the implementation of our MICCAI 2025 paper:  
[**“UFO-3: Unsupervised Three-Compartment Learning for Fiber Orientation Distribution Function Estimation”**](https://github.com/tensor2023/xueqinggao.github.io/blob/main/UFO-3_Paper.pdf), authored by Xueqing Gao<sup>☯️</sup>, Rizhong Lin<sup>☯️</sup>, Jianhui Feng, Yonggang Shi, and Yuchuan Qiao<sup>✉️</sup>.  
<small><sup>☯️</sup> *Equal contribution* <sup>✉️</sup> _Corresponding author_</small>

###### Xueqing Gao is an undergraduate student at Tongji University, preparing to apply for PhD programs in Spring/Fall 2026. Xueqing is particularly interested in computational imaging and physics-based vision. Her full CV is available here: [**XueqingGao_CV**](https://github.com/tensor2023/xueqinggao.github.io/blob/main/XueqingGao_CV.pdf)

## 🧠 Key Features

- **Unsupervised Learning**: Requires no ground-truth fODF for training, eliminating the need for extensive training data.
- **Single-Shell Data**: Accurately estimates fODF from single-shell dMRI data, making it suitable for clinical settings.
- **Physics-Informed**: Integrates a three-compartment biophysical model (intra-, extra-axonal, and trapped water) for interpretable results.
- **High Performance**: Achieves subject-specific training in ~30 minutes on a single GPU, with fast inference (<10 seconds per subject).
- **Accurate**: Outperforms existing methods on synthetic data (MAE < 10°, ACC > 91%) and produces reconstructions comparable to multi-shell methods on in-vivo data.
- **End-to-End**: A U-Net-based model learns to map the input dMRI signal to the fODF and compartment maps.

## 📖 Methodology

The UFO-3 framework combines a biophysical three-compartment model with a U-Net architecture. The input dMRI signal is projected onto a dense spherical graph. The U-Net then estimates the intra-axonal fODF, extra-axonal fraction, trapped water fraction, and isotropic diffusivity. The final reconstruction is obtained through a physics-constrained optimization that enforces sparsity and non-negativity, ensuring the results are physically plausible.

## 📁 Project Structure

```text
├── .gitignore            # Git ignore file
├── data/                 # Dataset and data loading
│   ├── __init__.py
│   ├── base_dataset.py
│   ├── chcp_dataset.py
│   ├── gen_simu.py       # Simulated data generator
│   ├── gen_Y_G.py        # For convenience, Y and G can be calculated in advance
│   └── ConstraintSet.mat # SH basis constraint
│
├── model/                # Model components
│   ├── deconvolution.py
│   ├── interpolation.py
│   ├── model.py
│   └── reconstruction.py
│
├── utils/                # Utility functions
│   └── loss.py
│
├── train.py              # Training entry
├── test.py               # Evaluation script
├── train.yaml            # Training config
├── test.yaml             # Test config
└── README.md             # You are here
```

## ⚙️ Installation

This codebase uses Python 3.8+, PyTorch >= 1.10, and a requirements.txt file is provided for convenience.

To install all dependencies, simply run:

```bash
pip install -r requirements.txt
```

## 🚀 Getting Started

### 1. Simulate or prepare your dMRI dataset

You can use:

```bash
python data/gen_simu.py
```

to generate synthetic diffusion MRI signals.

### 2. Train the model

```bash
python train.py
```

### 3. Test the model

```bash
python test.py
```

## 📄 Citation

If you find our work useful, please kindly cite:

```bibtex
@inproceedings{gao2025ufo3,
  title={{UFO}-3: Unsupervised Three-Compartment Learning for Fiber Orientation Distribution Function Estimation},
  author={Gao, Xueqing and Lin, Rizhong and Feng, Jianhui and Shi, Yonggang and Qiao, Yuchuan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention ({MICCAI})},
  year={2025},
  month={sep},
}
```
