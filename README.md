# Data Science Challenge 2023 - Cardiac Electrophysiology

This repository contains the notebooks and code for the 2023 Data Science Challenge (DSC) at Lawrence Livermore National Laboratory (LLNL).

- Link : https://data-science.llnl.gov/dsc
- Authors : 
    - Mikel Landajuela (landajuelala1@llnl.gov)
    - Cindy Gonzales (gonzales72@llnl.gov)


## Description
We attempted at using a Transformer to predict the ECG data.

To run the program run 
```
python main.py
```


The above code may or may not run. In this case, run the notebook 
```
Transformer.ipynb
```

## Contents
- [tutorials](./tutorials/)
    - [tutorials/image_classifier_tutorial_v1.2](./tutorials/image_classifier_tutorial_v1.2.ipynb) : Tutorial on image classification
    - [tutorials/DSC_regression-tutorial](./tutorials/DSC_regression-tutorial.ipynb) : Tutorial on regression
- [notebooks](./notebooks/)
    - [task_1_getting_started.ipynb](./notebooks/task_1_getting_started.ipynb) : Task 1 notebook
    - [task_2_getting_started.ipynb](./notebooks/task_2_getting_started.ipynb) : Task 2 notebook
    - [task_3_getting_started.ipynb](./notebooks/task_3_getting_started.ipynb) : Task 3 notebook
    - [task_4_getting_started.ipynb](./notebooks/task_4_getting_started.ipynb) : Task 4 notebook

## Roadmap
- If you are unfamiliar with the field of machine learning, have a look at the [tutorials](./tutorials/) folder, which contains a set of notebooks to get you started with machine learning.
- The challenge is divided into 4 tasks, each of which is described in detail in the corresponding notebook.
- Start by reading the `task_#_getting_started.ipynb` notebooks for each task, to get familiar with the data and the task.


## Task 1 : Heartbeat Classification
Get familiar working with ECG data by using the [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) to perform binary classification for healthy heartbeat vs. irregular heartbeat

Start by reading the [task_1_getting_started.ipynb](./notebooks/task_1_getting_started.ipynb) notebook.

## Task 2 : Irregular Heartbeat Classification
Diagnosing an irregular heartbeat by using the [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) to perform multiclass classification to diagnose the irregular heartbeats.

Start by reading the [task_2_getting_started.ipynb](./notebooks/task_2_getting_started.ipynb) notebook.

## Task 3 : Activation Map Reconstruction from ECG
Sequence-to-vector prediction using the [Dataset of Simulated Intracardiac Transmembrane Voltage Recordings and ECG Signals](https://library.ucsd.edu/dc/object/bb29449106)
to perform activation map reconstruction (i.e. transform a sequence of length 12x500 to 75x1 using a neural network)

Start by reading the [task_3_getting_started.ipynb](./notebooks/task_3_getting_started.ipynb) notebook.

## Task 4 : Transmembrane Potential Reconstruction from ECG
Sequence-to-sequence prediction using the [Dataset of Simulated Intracardiac Transmembrane Voltage Recordings and ECG Signals](https://library.ucsd.edu/dc/object/bb29449106) to perform transmembrane potential reconstruction (i.e. transform a sequence of length 12x500 to 75x500 using a neural network)

Start by reading the [task_4_getting_started.ipynb](./notebooks/task_4_getting_started.ipynb) notebook.


# Additional Information

### Working with the ECG Heartbeat Categorization Dataset

<details>
<summary>Download dataset</summary>

- Download the dataset from the [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
- Unzip the `archive.zip` file
- Rename the folder `archive` as `ecg_dataset` and place it in the root of the git repository

</details>

### Working with the Dataset of Simulated Intracardiac Transmembrane Voltage Recordings and ECG Signals

<details>
<summary>Download dataset</summary>

1. Download the dataset from the [Dataset of Simulated Intracardiac Transmembrane Voltage Recordings and ECG Signals](https://library.ucsd.edu/dc/object/bb29449106)
    - You will need to download all the components of the dataset one by one
2. Unzip the dataset

**Note** : For convenience, we have included a bash script to perform the above steps. To use the script, run the following command from the root of the repository:
```bash 
source download_intracardiac_dataset.sh
```
</details>

<details>
<summary>Further details</summary>

For further details, navigate to the `intracardiac_dataset` folder and read the `README.md` file.
- Look in `documentation/documentation.pdf` for a detailed description of the dataset, including the simulation process
- Look at the files `documentation/dataset_description.png` and `documentation/dataset_description.csv` for details on each simulation study
- Jupyter Notebook: Inspect the data using `notebooks/dataset_inspect.ipynb`
- Mathematica Notebook: Inspect the data using `notebooks/dataset_inspect.nb`
- The license documents can be found in `license`
</details>

### Additional Resources
- (Task 1) Paper : https://arxiv.org/pdf/1805.00794.pdf

    <em>Mohammad Kachuee, Shayan Fazeli, and Majid Sarrafzadeh. \
    "ECG Heartbeat Classification: A Deep Transferable Representation." arXiv preprint arXiv:1805.00794 (2018).</em>

- (Task 3 & 4) Paper : https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10081783

    <em>M. Landajuela, R. Anirudh, J. Loscazo and R. Blake, \
    "Intracardiac Electrical Imaging Using the 12-Lead ECG: A Machine Learning Approach Using Synthetic Data," \
    2022 Computing in Cardiology (CinC), Tampere, Finland, 2022, pp. 1-4, doi: 10.22489/CinC.2022.026.</em>

- (Task 3 & 4) Dataset: https://library.ucsd.edu/dc/object/bb29449106

    <em>Landajuela, Mikel; Anirudh, Rushil; Blake, Robert (2022).\
     Dataset of Simulated Intracardiac Transmembrane Voltage Recordings and ECG Signals. \
     In Lawrence Livermore National Laboratory (LLNL) Open Data Initiative. UC San Diego Library Digital Collections. https://doi.org/10.6075/J0SN094N</em>

- (Task 3 & 4) Medium Blog post : https://medium.com/p/a20661669937



License
----------------

Data Science Challenge 2023 is distributed under the terms of the MIT license. All new
contributions must be made under this license.

See [LICENSE](./LICENSE),
and
[NOTICE](./NOTICE) for details.

SPDX-License-Identifier: MIT

LLNL-CODE-849487
