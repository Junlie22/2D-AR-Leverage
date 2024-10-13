# Efficient Approximation of Leverage Scores in Two-dimensional Autoregressive Models

This repository includes the implementation of our work **"Efficient Approximation of Leverage Scores in Two-dimensional Autoregressive Models"**

## Introduction

A brief introduction about the folders and files:

* `dataset/`: path for storing downloaded data;
* `LARTVAD/`: path for implementation code of LARTVAD and result from LARTVAD;
  * `main.m`: the main program of LARTVAD;
  * `LARTVAD_AS.mat`: anomaly detection results from LARTVAD;
  * `LARTVAD__hyper_skin.mat`: facial skin analysis results from LARTVAD;

* `functions.py`: containing functions used in this project;

* `simulation.ipynb`: comparison of UNIF, OLS, LEV-exact, LEV-appr w.r.t. MSE and time cost versus the number of parameters, subsample size, and time series size;
* `simulation_parallel.ipynb`: parallel version of `simulation.ipynb`, designed for to improve the running efficiency during repeated experiments;
* `anomaly detection.ipynb`: comparison of LEV-exact, LEV-appr, RX and LARTVAD on anomaly detection tasks;
* `Hyper skin.ipynb`: comparison of LEV-exact, LEV-appr, RX and LARTVAD on facial skin analysis using hyper-skin dataset.

## Reproducibility

For simulation studies in Section 5,

* you can run `simulation.ipynb` or `simulation_parallel.ipynb` to reproduce results in Figure 3 and Figure 4.

We recommend to run `simulation.ipynb` for Figure 4 and run `simulation_parallel.ipynb` for Figure 3. Because parallel can improve computing efficiency but may result in inaccurate time estimates due to resource allocation schemes. You can get time cost in one repeated experiment using `simulation.ipynb` and get MSE in 20 repeated experiments using `simulation_parallel.ipynb`.

For the anomaly detection task in Section 6.1,

* you can run `anomaly detection.ipynb` to reproduce results in Figure 5, Figure 6 and Table 2.

For the facial skin analysis in Section 6.2,

* first, request and download Hyper-Skin 2023 Data (.mat and .jpg files as follows) from the link [https://github.com/hyperspectral-skin/Hyper-Skin-2023](https://github.com/hyperspectral-skin/Hyper-Skin-2023) and store them in the `dataset/` path;

```python
"dataset/Hyper-skin/NIR/p011_neutral_right.mat"
"dataset/Hyper-skin/NIR/p011_smile_right.mat"
"dataset/Hyper-skin/NIR/p015_neutral_right.mat"
"dataset/Hyper-skin/RGB_CIE/p011_neutral_right.jpg"
"dataset/Hyper-skin/RGB_CIE/p011_smile_right.jpg"
"dataset/Hyper-skin/RGB_CIE/p015_neutral_right.jpg"
```

* then, run `Hyper skin.ipynb` to reproduce the result in Figure 7.



