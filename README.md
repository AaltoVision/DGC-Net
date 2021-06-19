# DGC-Net: Dense Geometric Correspondence Network
This is a PyTorch implementation of our work ["DGC-Net: Dense Geometric Correspondence Network"](https://arxiv.org/abs/1810.08393)

**TL;DR** A CNN-based approach to obtain dense pixel correspondences between two views.

## License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa], available only for non-commercial use.

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg


## Installation
- create and activate conda environment with Python 3.x
```
conda create -n my_fancy_env python=3.7
source activate my_fancy_env
```
- install Pytorch v1.0.0 and torchvision library
```
pip install torch torchvision
```
- install all dependencies by running the following command:
```
pip install -r requirements.txt
```

## Getting started
* ```eval.py``` demonstrates the results on the HPatches dataset
To be able to run ```eval.py``` script:
    * Download an archive with pre-trained models [click](https://drive.google.com/file/d/1p1FarlU5byWez_mQC68DZ_eRQKfF9IIf/view?usp=sharing) and extract it
to the project folder
    * Download HPatches dataset (Full image sequences). The dataset is available [here](https://github.com/hpatches/hpatches-dataset) at the end of the page
    * Run the following command:
    ```
    python eval.py --image-data-path /path/to/hpatches-geometry
    ```

* ```train.py``` is a script to train DGC-Net/DGCM-Net model from scratch. To run this script, please follow the next procedure:
    * Download the [TokyoTimeMachine dataset](https://www.di.ens.fr/willow/research/netvlad/)
    * Run the command:
    ```
    python train.py --image-data-path /path/to/TokyoTimeMachine
    ```

## Performance on [HPatches](https://github.com/hpatches/hpatches-dataset) dataset
Method / HPatches ID|Viewpoint 1|Viewpoint 2|Viewpoint 3|Viewpoint 4|Viewpoint 5
:---|:---:|:---:|:---:|:---:|:---:
[PWC-Net](https://arxiv.org/abs/1709.02371)| 4.43 | 11.44 | 15.47 | 20.17 | 28.30
[GM](https://arxiv.org/abs/1703.05593) best model | 9.59 | 18.55 | 21.15 | 27.83 | 35.19
DGC-Net (paper) | **1.55** | **5.53** | **8.98** | 11.66 | 16.70
DGCM-Net (paper) | 2.97 | 6.85 | 9.95 | 12.87 | 19.13
DGC-Net (repo) | 1.74 | 5.88 | 9.07 | 12.14 | 16.50
DGCM-Net (repo) | 2.33 | 5.62 | 9.55 | **11.59** | **16.48**

Note: There is a difference in numbers presented in the original paper and obtained by the models of this repo. It might be related to the fact that both models (DGC-Net and DGCM-Net) have been trained using ```Pytorch v0.3```.

More qualitative results are presented on the [project page](https://aaltovision.github.io/dgc-net-site/)

## How to cite
If you use this software in your own research, please cite our publication:

```
@inproceedings{Melekhov+Tiulpin+Sattler+Pollefeys+Rahtu+Kannala:2018,
      title = {{DGC-Net}: Dense geometric correspondence network},
      author = {Melekhov, Iaroslav and Tiulpin, Aleksei and 
               Sattler, Torsten, and 
               Pollefeys, Marc and 
               Rahtu, Esa and Kannala, Juho},
       year = {2019},
       booktitle = {Proceedings of the IEEE Winter Conference on 
                    Applications of Computer Vision (WACV)}
}
```
