# DGC-Net: Dense Geometric Correspondence Network
This is a PyTorch implementation of our work ["DGC-Net: Dense Geometric Correspondence Network"](https://arxiv.org/abs/1810.08393)

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