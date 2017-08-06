# mv3d
![alt tag](https://github.com/mtatarchenko/mv3d/blob/master/thumbnail.png)
Source code accompanying the ECCV'16 paper "Multi-view 3D Models from Single Images with a Convolutional Network" by M. Tatarchenko, A. Dosovitskiy and T. Brox http://lmb.informatik.uni-freiburg.de/people/tatarchm/mv3d/. The models implemented here are slightly different from those described in the paper, so we do not guarantee both quantitative and qualitative results to be exactly the same.

## Dependencies
- Tensorflow 0.12
- Panda3D
- NumPy
- SciPy

## Data
The networks were trained on a subset of the ShapeNet dataset containing 3D models of cars http://shapenet.cs.stanford.edu/. If you want to reproduce our results, you need to get the models. Follow the download instructions from the official website. Unzipped models should be placed in the 'data/obj_cars' folder in the following structure:
*data/obj_cars/model_id/{model.obj, model.mtl}*. By default our rendering engine uses the .bam format, so after downloading the models you need to convert them. This can be done by subsequently applying obj2egg and egg2bam utilities, which come as a part of Panda3D package.

For the background experiment we used a subset of the ImageNet validation set. Those should be placed in *data/bg_imagenet/val/ILSVRC2012_preprocessed_val_xxxxxxxx.JPEG*, where xxxxxxxx is an 8-digit image id (in the range 00000001 - 00050000).

## Usage
Run *download_data.py* first to download the pre-rendered version of the test set and the pre-trained network snapshots. We provide 3 pre-trained networks:
- *nobg_nodm.py* - RGB input without background -> RGB output without background
- *nobg_dm.py* - RGB input without background -> RGB-D output without background
- *bg_nodm.py* - RGB input with background -> RGB output without background

You can run every script in train/test mode by uncommenting the corresponding parts of the script. When you run the script in the train mode, the realtime renderer is used to generate training data on the fly. To speed up the training process, you might want to implement a rendering app with multiple instances of the renderer running in parallel.

## License and Citation
All code is provided for research purposes only and without any warranty. Any commercial use requires our consent. When using the code in your research work, please cite the following paper:

    @InProceedings{TDB16a,
    author       = "M. Tatarchenko and A. Dosovitskiy and T. Brox",
    title        = "Multi-view 3D Models from Single Images with a Convolutional Network",
    booktitle    = "European Conference on Computer Vision (ECCV)",
    year         = "2016"
    }
