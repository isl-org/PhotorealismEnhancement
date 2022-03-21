## Setup

The code has been developed using a conda environment:
    
    conda create -n epe python=3.8
    conda activate epe

You'll need pytorch, scikit-image, imageio, tqdm:

    conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
    conda install scikit-image
    conda install imageio

    pip install tqdm
    
Training requires [MSeg](https://github.com/mseg-dataset/mseg-semantic) for generating robust labels of a dataset, [faiss](https://github.com/facebookresearch/faiss) for matching crops across datasets, and [LPIPS](https://github.com/richzhang/PerceptualSimilarity) for the perceptual loss. The code contains optional downsampling using [Kornia](https://github.com/kornia/kornia) for the perceptual loss and the discriminator. If you do not wish to use it (it is disabled by default), you may remove this dependency. Please refer to the respective websites for up-to-date installation instructions.

After installing all the prerequisites, install the epe package (editable) locally:

    pip install -e ./


## Preparing datasets

The pipeline is designed for an image resolution of roughly 960x540. Using images of other resolutions (e.g., FullHD) may require modification of the code and possibly even the network architecture for best results. The pipeline expects for each dataset a txt file containing paths to all images. Each line should contain paths to image, robust label map, gbuffer file, and ground truth label map, all separated by commas. Real datasets only require paths to images and robust label maps. For computing the label maps, see below.

### Synthetic footage (from games)

Unfortunately, we cannot publish any recorded data here. You will need to build your own dataset. This is less challenging than you might think. For more information, we refer to the many papers that focused on this topic. The reference implementation we provide here is compatible with the Playing for Data and VIPER datasets, but does not use any actual G-buffers. To nevertheless demonstrate the pipeline, we added a script for generating fake G-buffers via a VGG16 network (dataset/generate_fake_gbuffers.py).
For best results, (e.g., if you record your own dataset) using a comprehensive set of G-buffers is recommended.

A number of research papers focused on this topic:

* [Playing for Data](https://download.visinf.tu-darmstadt.de/data/from_games/)
* [Free Supervision from Video Games](http://www.philkr.net/fsv/)

### Real footage (from real photos)

* For training our model, label maps on real data are not required. Raw images are sufficient.
* Label maps are required if you want to evaluate the realism of enhanced images using our sKVD metric (more on that below).
* As in many machine learning problems, more data is better than less, more diverse is better than all of the same. The better your real-world footage matches the synthetic data (scale, perspective, layout of a scene, distribution of objects), the better the enhancements.

### Computing robust label maps

We use segmentations generated with [MSeg](https://github.com/mseg-dataset/mseg-semantic) as robust label maps for training the discriminator.
Please refer to MSeg's documentation for running their pretrained models on your data.

Tip for speeding up computation: The robust label maps used in the discriminator are scaled to the size of the feature maps. Thus, they are effectively used only at down-sampled resolution(s) and can be computed from images of lower resolution.

We also provide a discriminator without using those labels. In case you want to use it without actually generating robust labels, you will need to modify the dataset code as this loads robust labels by default.

### Matching patches across datasets

First, we need to sample crops from source and target datasets and compute VGG features on those crops. As a default, we sample 15 crops per image. But, depending on the size of your dataset, increasing the number of sampled crops may be beneficial. The csv files for the datasets contain per row the paths to the images of each dataset.
    
    python epe/matching/feature_based/collect_crops.py PfD pfd_files.csv 		# creates crop_PfD.csv, crop_Pfd.npz
    python epe/matching/feature_based/collect_crops.py Cityscapes cs_files.csv	# creates crop_Cityscapes.csv, crop_Cityscapes.npz

The generated csv files contain path and coordinates, the npz files contain the features.

Second, for each crop in source dataset (here PfD), find k (here 10) nearest neighbours in target dataset (here Cityscapes).
This step (and only this one) requires [faiss](https://github.com/facebookresearch/faiss).

    python epe/matching/feature_based/find_knn.py crop_PfD.npz crop_Cityscapes.npz knn_PfD-Cityscapes.npz -k 10

Third, we filter neighbours based on feature distance: 1.0 works well.

    python epe/matching/filter.py knn_PfD-Cityscapes.npz crop_PfD.csv crop_Cityscapes.csv 1.0 matched_crops_PfD-Cityscapes.csv

As a rough guidance, the csv with matched crops should contain at least 200K pairs (lines), more is better. If your datasets are smaller than the ones we used, or differ more, we strongly recommend increasing the number of sampled crops as well as the number of neighbours (the -k in the second step). We advise against increasing the threshold for the feature distance (third step) as this will ultimately decrease quality (by requiring a stricter VGG loss later to reduce artifacts). For visualization we provide a script that samples matches for a set of thresholds (sample_matches.py).

Fourth, we compute sampling weights such that all locations in the source dataset will be sampled at the same frequency. This is necessary for preventing oversampling only regions that are well matched between datasets (e.g. just cars for GTA/Cityscape). The magic numbers 526 and 957 are image height and width of the source dataset (here Playing for Data).
    
    python epe/matching/compute_weights.py matched_crops_PfD-Cityscapes.csv 526 957 crop_weights_PfD-Cityscapes.npz

## Training a model

After modifying the paths in the config (./config/train_pfd2cs.yaml), we are ready to go:

    python epe/EPEExperiment.py train ./config/train_pfd2cs.yaml --log=info
    
## Running a model

For testing the model, paths and checkpoint to load need to be specified in the config. Then we can run:

    python epe/EPEExperiment.py test ./config/test_pfd2cs.yaml

## Differences to the paper

We have updated the network architecture slightly. A version close to the paper, but with more efficient fusion of layers is the ienet.py (hr in config). For the ienet2.py (hr_new in config) we take inspiration from superresolution works: we remove all batch/group normalization and initialize the residual paths such they have a lower initial contribution. Further, we add 2 stages to the HRNet. This further improves final quality as well as training stability.

## Evaluation

Not yet available


