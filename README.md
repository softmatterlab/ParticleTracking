# Tracking Soft Matter

**Tracking Soft Matter** provides a modular set of Jupyter notebooks to help experimentalists and researchers analyze microscopy data with minimal coding effort. The tutorials are designed to be accessible, with well-structured utility modules and clear workflows.

---

## Overview

This repository supports the full pipeline for particle tracking:

- **Detection** – Identify and localize particles using both classical and deep learning methods.
- **Linking** – Associate particles across time frames to reconstruct trajectories.
- **Simulation** - Generate realistic dataset for training and evaluation.
- **Evaluation** – Compare predictions to ground truth using tracking metrics.
- **Visualization** – Animate and inspect results interactively.

---

## Tutorials

The tutorials are organized into two main parts: **Detection & Localization** and **Linking**.

### Detection & Localization

You will apply and compare several detection strategies:

1. **Thresholding & Connected Components** – Simple and fast.
2. **Crocker–Grier (TrackPy)** – Classical approach to particle tracking.
3. **U-Net** – Supervised deep learning for segmentation.
4. **LodeSTAR** – Unsupervised deep learning for subpixel localization.

Each method is benchmarked on simulated data and then applied to experimental datasets.

Tutorials:

- [Detection of Spheres](tutorial/detection/spheres/detection_spheres.ipynb)  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmanzo/tracking-softmatter/blob/main/tutorial/detection/spheres/detection_spheres.ipynb)

- [Detection of Core-Shell Spheres](tutorial/detection/core-shell%20spheres/detection_core-shell.ipynb)  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmanzo/tracking-softmatter/blob/main/tutorial/detection/core-shell%20spheres/detection_core-shell.ipynb)

- [Detection of Ellipses](tutorial/detection/ellipses/detection_ellipses.ipynb)  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmanzo/tracking-softmatter/blob/main/tutorial/detection/ellipses/detection_ellipses.ipynb)

<p align="left">
  <img width="200" src="assets/fig1.png?raw=true">
  <img width="200" src="assets/fig2.png?raw=true">
  <img width="200" src="assets/fig3.png?raw=true">
</p>

---

### Linking

Associate localized particles across frames to reconstruct trajectories using:

1. **Nearest-neighbor linking (TrackPy)**.
2. **Linear Assignment Problem (LAP) using Hungarian algorithm (LapTrack)**.
3. **Graph-based deep learning linker (MAGIK, via `deeplay`)**.

Tutorial:

- [Linking Spheres](tutorial/linking/spheres/linking_spheres.ipynb)  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmanzo/tracking-softmatter/blob/main/tutorial/linking/spheres/linking_spheres.ipynb)

<p align="left">
  <img width="400" src="assets/track.gif?raw=true">
</p>

---

## Getting Started

1. **Clone the repository:**

```bash
git clone https://github.com/cmanzo/tracking-softmatter.git
cd tracking-softmatter
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Launch the tutorials:**

```bash
jupyter lab  # or jupyter notebook
```

---

## Using `utils/` in Google Colab

The tutorials rely on the utility modules stored in the `utils/` folder.

### Option 1: Clone the full repository (recommended)

```python
!git clone https://github.com/cmanzo/tracking-softmatter.git
%cd tracking-softmatter
```

### Option 2: Upload the `utils/` folder manually

Upload a zipped copy of `utils/`:

```python
from google.colab import files
uploaded = files.upload()  # Upload utils.zip
```

Unzip it:

```bash
!unzip utils.zip -d .
```

Then import as usual:

```python
from utils import detection_utils, tracking_utils, video_utils
```

---

## Dependencies

Core libraries:

- `numpy`, `scipy`, `matplotlib`
- `scikit-image`, `torch`
- `trackpy`, `laptrack`, `deeptrack`, `deeplay`

See `requirements.txt` for full details.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Citation

If you use this toolkit for your research, please cite:  
*(BibTeX and citation information coming soon)*
