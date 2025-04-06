# Particle Tracking Tutorial

> This tutorial is intended for experimentalists who wish to expand
> their toolset in analyzing experimental data without an excessive focus
> on writing code. The notebooks are designed with a plug-and-play approach in mind,
> where most of the utility functions and classes are declared in a separate file in order to keep a clear flow of
> information throughout, with a focus on results.


The tutorial is divided into two main parts: [**Particle Detection**](https://github.com/aarondomenzain/tracking-softmatter-aarond/tree/tracking-softmatter-aarond-dev/tutorial/detection) and [**Particle Tracking**](https://github.com/aarondomenzain/tracking-softmatter-aarond/tree/tracking-softmatter-aarond-dev/tutorial/tracking), with categories of different types of experimental data therein. 

## What to expect 

The tutorial showcases a handful of conventional methods of detection and tracking with varying complexities of implementations. Threshholding for example, is very easy to implement but requires the data to be clean and simple. 

We show how to generate synthetic data with `DeepTrack2` in order to train the Deep Learning models, but also included are pre-trained weights which lets you skip training.

## Getting started

1.  Download the repository or clone it directly by typing  ```git clone https://github.com/aarondomenzain/tracking-softmatter-aarond.git``` in a terminal. 

2. Ensure you are using ``Python version 3.9-3.12``.
   
3. Install the required packages with the following command: ```pip install deeptrack deeplay imageio ipykernel ipywidgets laptrack matplotlib numpy opencv-python scikit-image scipy torch trackpy```


### **Detection**

The **Detection** part of the tutorial guides the user through the process of applying four different object detection methods, they are ordered as:

1. **Thresholding** with `NumPy`
2. **Crocker-Grier** with `TrackPy`
3. **U-Net detection** with `Deeplay`
4. **LodeSTAR detection** with `Deeplay`

These object detection methods are applied to simulated data in order to obtain a performance metric and then in turn applied to experimental data.

There are three notebook variations of the **Detection** part, accounting for different particle geometries:

1. Detection of **Spheres**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aarondomenzain/tracking-softmatter-aarond/blob/tracking-softmatter-aarond-dev/tutorial/detection/spheres/detection_spheres.ipynb)

2. Detection of **Core-Shell Spheres**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aarondomenzain/tracking-softmatter-aarond/blob/tracking-softmatter-aarond-dev/tutorial/detection/core-shell%20spheres/detection_core-shell.ipynb)

3. Detection of **Ellipses**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aarondomenzain/tracking-softmatter-aarond/blob/tracking-softmatter-aarond-dev/tutorial/detection/ellipses/detection_ellipses.ipynb)
   
If you wish to run the detection notebooks in Google Colab, you will need to upload the [utils.py](https://github.com/aarondomenzain/tracking-softmatter-aarond/blob/tracking-softmatter-aarond-dev/tutorial/detection/utils.py) file to your session.

The different types of data can be previewed in the figures below.
<p align="left">
  <img width="200" src=/assets/fig1.png?raw=true>
  <img width="200" src=/assets/fig2.png?raw=true>
  <img width="200" src=/assets/fig3.png?raw=true>
<br/>

### **Tracking**[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aarondomenzain/tracking-tracking-softmatter-aarond-dev/tutorial/tracking/tracking_spheres.ipynb)

The **Tracking** part of the tutorial is to guide through the process of applying different tracking methods to simulated and experimental data.

If you wish to run the tracking notebook in Google Colab, you will need to upload the [utils_tracking.py](https://github.com/aarondomenzain/tracking-softmatter-aarond/blob/tracking-softmatter-aarond-dev/tutorial/tracking/utils_tracking.py) file to your session.

!pip install deeptrack deeplay laptrack trackpy -q 

The methods used in the tracking tutorial are:

1. **Modified Crocker-Grier** with `TrackPy`
2. **Hungarian algorithm** with `LapTrack`
3. **MAGIK** with `Deeplay`

The results from these methods are then evaluated with performance metrics and displayed as movies. 

<p align="left">
  <img width="400" src=/assets/track.gif?raw=true>
<br/>



