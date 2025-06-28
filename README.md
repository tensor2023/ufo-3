# UFO-3: Unsupervised Fiber Orientation 3-Compartment Model

📄 **Accepted at MICCAI 2025**  
🧠 *A fully unsupervised learning framework for estimating fiber orientation distribution function (FOD) using only diffusion MRI data.*

This repository contains the official PyTorch implementation of our MICCAI 2025 paper:

### ✍️ Authors
**Xueqing Gao**<sup>†</sup> &nbsp;&nbsp;|&nbsp;&nbsp; **Rizhong Lin**<sup>†</sup> &nbsp;&nbsp;|&nbsp;&nbsp; **Jianhui Feng**  
**Yonggang Shi** &nbsp;&nbsp;|&nbsp;&nbsp; **Yuchuan Qiao**<sup>✉</sup>  

<sub>† Equal contribution    ✉ Corresponding author</sub>  

> 📄 [**Read the Paper (PDF)**](https://github.com/tensor2023/xueqinggao.github.io/blob/main/UFO-3_Paper.pdf)



###### Xueqing Gao is an undergraduate student at Tongji University, preparing to apply for PhD programs in Spring/Fall 2026. Xueqing is particularly interested in computational imaging and physics-based vision. Her full CV is here: [**XueqingGao_CV**](https://github.com/tensor2023/xueqinggao.github.io/blob/main/XueqingGao_CV.pdf)

---


## 🧠 Key Features

- Completely **unsupervised** learning: no ground-truth FOD required
- Physically-informed **three-compartment model** (intra-, extra-axonal, and trapped water)
- End-to-end training from **DTI signal → FOD + compartment map**
- Simple CNN-based model; no spherical convolutions required

---

## 📁 Project Structure

```
├── data/                 # Dataset and data loading
│   ├── chcp\_dataset.py
│   ├── gen\_simu.py       # Simulated data generator
│   └── ConstraintSet.mat # SH basis constraint
│
├── model/                # Model components
│   ├── deconvolution.py
│   ├── interpolation.py
│   ├── model.py
│   └── reconstruction.py
│
├── utils/                # Utility functions
│
├── train.py              # Training entry
├── test.py               # Evaluation script
├── train.yaml            # Training config
├── test.yaml             # Test config
├── README.md             # You are here


---
## ⚙️ Installation

This codebase uses **Python 3.8+**, **PyTorch >= 1.10**

*Optional:* If you use `.mat` files, install `scipy` and `h5py`.

---

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

---

<!-- ## 📄 Citation

If you use this codebase in your work, please cite:

```bibtex
@inproceedings{gao2025ufo3,
  title={UFO-3: Unsupervised Fiber Orientation 3-Compartment Model for dMRI FOD Estimation},
  author={Gao, Xueqing and Qiao, Yuchuan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year={2025}
}
``` -->

