import json
import glob
import os

config = dict()

model_config = dict()
model_config["name"] = "DynUNet"  # network model name from MONAI
# set the network hyper-parameters
model_config["in_channels"] = 3  # 1 input images
model_config["out_channels"] = 2   # Vessel
model_config["spatial_dims"] = 3   # 3D input images
model_config["deep_supervision"] = False  # do not check outputs of lower layers
model_config["strides"] = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]][:-1]  # number of downsampling convolutions
model_config["filters"] = [64, 96, 128, 192, 256, 384, 512, 768, 1024][:len(model_config["strides"])]  # number of filters per layer
model_config["kernel_size"] = [[3, 3, 3]] * len(model_config["strides"])  # size of the convolution kernels per layer
model_config["upsample_kernel_size"] = model_config["strides"][1:]  # should be the same as the strides

# put the model config in the main config
config["model"] = model_config

config["optimizer"] = {'name': 'Adam', 
                       'lr': 0.001}  # initial learning rate

# define the loss
config["loss"] = {'name': 'DiceLoss', # from Monai
                  'include_background': True,  # we do not have a label for the background, so this should be true (by "include background" monai means include channel 0)
                  'sigmoid': True}  # transform the model logits to activations

# set the cross validation parameters
#config["cross_validation"] = {'folds': 5,  # number of cross validation folds
#                              'seed': 25},  # seed to make the generation of cross validation folds consistent across different trials
# set the scheduler parameters
config["scheduler"] = {'name': 'ReduceLROnPlateau', 
                       'patience': 10,  # wait 10 epochs with no improvement before reducing the learning rate
                       'factor': 0.5,   # multiply the learning rate by 0.5
                       'min_lr': 1e-08}  # stop reducing the learning rate once it gets to 1e-8

# set the dataset parameters
config["dataset"] = {'name': 'SegmentationDatasetPersistent',  # 'Persistent' means that it will save the preprocessed outputs generated during the first epoch
# However, using 'Persistent', does also increase the time of the first epoch compared to the other epochs, which should run faster
  'desired_shape': [128, 128, 32],  # resize the images to this shape, increase this to get higher resolution images (increases computation time and memory usage)
  'labels': [0,1],  # 0: non-vessel, 1: vessel
  #'setup_label_hierarchy': False,  # changes the labels to whole tumor (2, 1, 4), tumor core (1, 4), and enhancing tumor (4) to be consistent with the challenge
  'normalization': 'NormalizeIntensityD',  # z score normalize the input images to zero mean unit standard deviation
  'normalization_kwargs': {'channel_wise': True, "nonzero": False},  # perform the normalization channel wise and include the background
  'resample': False,  # resample the images when resizing them, otherwise the resize could crop out regions of interest
  'crop_foreground': False,  # crop the foreground of the images
  'cache_dir': './cache/',  # cache
                    }
config["training"] = {'batch_size': 2,  # number of image/label pairs to read at a time during training
  'validation_batch_size': 1,  # number of image/label pairs to read at atime during validation
  'amp': False,  # don't set this to true unless the model you are using is setup to use automatic mixed precision (AMP)
  'early_stopping_patience': None,  # stop the model early if the validaiton loss stops improving
  'n_epochs': 500,  # number of training epochs, reduce this if you don't want training to run as long
  'save_every_n_epochs': None,  # save the model every n epochs (otherwise only the latest model will be saved)
  'save_last_n_models': None,  # save the last n models 
  'save_best': True}  # save the model that has the best validation loss

# get the training filenames
config["training_filenames"] = list()

# if your BraTS data is stored somewhere else, change this code to fetch that data
for subject_folder in sorted(glob.glob("./training_data/*")):
    if not os.path.isdir(subject_folder):
        continue
    image_filenames = sorted(glob.glob(os.path.join(subject_folder, "*.nrrd")))
    for i in range(len(image_filenames)):
        if "seg_" in image_filenames[i].lower():
            label = image_filenames[i]
            image_filename = label
            image_filename = image_filename.replace('seg_', 'image_')            
            config["training_filenames"].append({"image": image_filename, "label": label})


config["test_filenames"] = list()  # "validation_filenames" is reserved for the cross-validation, so we will call this bratsvalidation_filenames
for subject_folder in sorted(glob.glob("./validation_data/*")):
    if not os.path.isdir(subject_folder):
        continue
    image_filenames = sorted(glob.glob(os.path.join(subject_folder, "*.nrrd")))
    config["test_filenames"].append({"image": image_filenames})
    
    
# set the dataset parameters
config["predict"] = {'name': 'PredictSegmentation',  # 'Persistent' means that it will save the preprocessed outputs generated during the first epoch
# However, using 'Persistent', does also increase the time of the first epoch compared to the other epochs, which should run faster
  'desired_shape': [128, 128, 32],  # resize the images to this shape, increase this to get higher resolution images (increases computation time and memory usage)
  'labels': [0,1],  # 0: non-vessel, 1: vessel
  #'setup_label_hierarchy': False,  # changes the labels to whole tumor (2, 1, 4), tumor core (1, 4), and enhancing tumor (4) to be consistent with the challenge
  'normalization': 'NormalizeIntensityD',  # z score normalize the input images to zero mean unit standard deviation
  'normalization_kwargs': {'channel_wise': True, "nonzero": False},  # perform the normalization channel wise and include the background
  'resample': False,  # resample the images when resizing them, otherwise the resize could crop out regions of interest
  'crop_foreground': False,  # crop the foreground of the images
  'cache_dir': './cache/',  # cache
  'filename': './test',  # Test directory
    'batch_size': 8,  # number of image/label pairs to read at a time during training
                    }

with open("./pcmra_vessel.json", "w") as op:
    json.dump(config, op, indent=4)
    