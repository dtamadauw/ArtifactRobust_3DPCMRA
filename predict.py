import os
import logging
import argparse
import numpy as np
import nrrd
from data import VDataset_Test_3axes
import torch
from torch.utils.data import DataLoader
from Tools import depatch_img
from unet3d.utils.utils import load_json
from unet3d.predict.volumetric import volumetric_predictions
from unet3d.utils.filenames import load_dataset_class
from unet3d.scripts.segment import format_parser as format_segmentation_parser
from unet3d.scripts.script_utils import (get_machine_config, in_config,
                                         add_machine_config_to_parser,
                                         fetch_inference_dataset_kwargs_from_config,
                                         build_or_load_model_from_config,
                                         check_hierarchy, build_inference_loader, get_kwargs)


def format_parser(parser=argparse.ArgumentParser(), sub_command=False):
    parser.add_argument("--output_directory", required=True)
    parser.add_argument("--input_dir", required=True)
    if not sub_command:
        parser.add_argument("--config_filename", required=True)
        parser.add_argument("--model_filename", required=True)
        add_machine_config_to_parser(parser)

    parser.add_argument("--group", default="test",
                        help="Name of the group of filenames to make predictions on. The default is 'test'. "
                             "The script will look for a key in the configuration file that lists the filenames "
                             "to read in and make predictions on.")
    parser.add_argument("--activation",
                        help="Specify whether to apply an activation function to the outputs of the model before writing to "
                             "file.")

    format_segmentation_parser(parser, sub_command=True)
    return parser


def parse_args():
    return format_parser().parse_args()


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    namespace = parse_args()
    logging.info("Config filename: %s", namespace.config_filename)
    config = load_json(namespace.config_filename)
    run_inference(config=config,
                  output_directory=namespace.output_directory,
                  model_filename=namespace.model_filename,
                  input_dir=namespace.input_dir,
                  group=namespace.group,
                  activation=namespace.activation,
                  system_config=get_machine_config(namespace))


def run_inference(config, output_directory, model_filename, input_dir, group, activation, system_config):
    logging.info("Output directory: %s", output_directory)
    work_dir = os.path.abspath(output_directory)
    label_hierarchy = check_hierarchy(config)
    cache_dir = os.path.join(work_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    logging.info("Cache dir: %s", cache_dir)
    dataset_class = load_dataset_class(config["dataset"],
                                       cache_dir=cache_dir)
    key = f"{group}_filenames"
    logging.info("Filenames key: %s", key)
    if key not in config:
        raise ValueError("Could not find key {} in the configuration file. Use the change the group "
                         "('--group' on commandline) to the name of the group of filenames "
                         "(e.g., 'validation' to use 'validation_filenames') "
                         "that you want to predict.")

    inference_dataset_kwargs, batch_size, prefetch_factor = fetch_inference_dataset_kwargs_from_config(config)


    print('===> Loading datasets')
    data_dir = config["predict"]['filename']
    full_dataset = VDataset_Test_3axes(input_dir) # create dataloader
    test_data_loader = DataLoader(dataset=full_dataset, num_workers=1, batch_size=config["predict"]['batch_size'], shuffle=False)
    print(len(full_dataset))
    device = torch.device("cuda")


    print('===> Building models')
    logging.info("Model filename: %s", model_filename)
    model = build_or_load_model_from_config(config,
                                            model_filename,
                                            system_config["n_gpus"],
                                            strict=True)
    model = model.float()
    model.eval()



    print('===> Start prediction')
    preds = []
    for iteration, batch in enumerate(test_data_loader, 1):

        inputs = batch[0].to(device)
        outputs = model(inputs)
        outputs = torch.sigmoid(outputs)
        outputs = outputs[:,1].data.cpu().numpy()

        if iteration == 1:
            preds = outputs
        else:
            preds = np.concatenate((preds,outputs), axis=0)


    diml = preds.shape
    print(preds.shape)
    preds = np.reshape(preds, (diml[0],diml[1],diml[2],diml[3]))


    print(full_dataset.get_X().shape)
    recon = depatch_img(preds,[full_dataset.imdim[0],full_dataset.imdim[1],full_dataset.imdim[2]], full_dataset.get_X(),full_dataset.get_Y(),full_dataset.get_Z())
    print(recon.shape)
    recon[np.isnan(recon)] = 0

    print(model_filename)
    print('W/ PA Augmentation')
    nrrd.write('%s/segmentation_map.nrrd'%(input_dir),recon.astype(np.float32),header=full_dataset.header)  

    th_val = 0.0
    th_mask = recon
    print(np.max(recon))
    print(np.min(recon))
    th_mask = np.where(recon<0.4, 1, 0)
    print(np.max(th_mask))
    print(np.min(th_mask))


    nrrd.write('%s/segmentation_binarized.nrrd'%(input_dir),th_mask.astype(np.uint8),header=full_dataset.header)



if __name__ == '__main__':
    main()
