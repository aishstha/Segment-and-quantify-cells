
This initiative supports researchers in automating nuclei segmentation through a deep learning model trained on our dataset, providing quantified results in an Excel file. It facilitates the identification and quantification of cell nuclei.
    
## Prerequisites

- [Python](https://www.python.org/downloads/) - v3.10.8
- [cv2](https://pypi.org/project/opencv-python/) - 4.6.0
- [matplotlib](https://matplotlib.org/stable/users/installing/index.html) - 3.6.2
- [pandas](https://pypi.org/project/pandas/) - 1.5.2
- [skimage](https://pypi.org/project/scikit-image/) - 0.19.3

The repository's scripts are organized into three sections: pre-processing, deep learning model, and post-processing

## Setup 
Clone the repository, install the dependencies and get started right away.

    $ git clone git@github.com:aishstha/Segment-and-quantify-cells.git](https://github.com/aishstha/Segment-and-quantify-cells.git)
    $ cd Segment-and-quantify-cells
  
## For post-processing  
Example :
```bash
$ ./segment_script.py  "Data/IXMtest_C05_s7_w1F71963FB-8F29-41CB-A5F5-07CB9584BBC5.tif"

```
 
## Usage

You can manually Install packages mentioned above (cv2, numpy, matplotlib, pandas, sys, skimage).

OR 

Use the conda environment file of this project. For more information, check `environment.yml` 
    
### Conda Environment 

Use the terminal or an Anaconda Prompt for the following steps:

1. Create the environment from the environment.yml file:

```bash
$ conda env create -f environment.yml

```
2. Activate the new environment: `conda activate segmentation`

3. Verify that the new environment was installed correctly:

```bash
$ conda env list

```
