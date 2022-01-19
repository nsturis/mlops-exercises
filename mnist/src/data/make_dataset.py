# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import torch
import numpy as np
import glob2 as glob


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    train_files = glob.glob(input_filepath + "/train*.npz")
    # load each train_file and concatenate them into a single numpy array
    images = torch.Tensor(np.concatenate([np.load(f)["images"] for f in train_files]))
    labels = torch.Tensor(
        np.concatenate([np.load(f)["labels"] for f in train_files])
    ).long()
    # create train dataset
    torch.save(images, output_filepath + "/train_images.pt")
    torch.save(labels, output_filepath + "/train_labels.pt")
    test_file = np.load(input_filepath + "/test.npz")
    # load test images and labels
    test_images = torch.Tensor(test_file["images"])
    test_labels = torch.Tensor(test_file["labels"]).long()
    torch.save(test_images, output_filepath + "/test_images.pt")
    torch.save(test_labels, output_filepath + "/test_labels.pt")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
