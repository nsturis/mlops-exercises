import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import argparse
import sys

import torch

import torch.nn.functional as F

from matplotlib import pyplot as plt

from src.models.model import MyAwesomeModel
from src.data.load_data import loadTrain


@click.command()
@click.argument("data_filepath", type=click.Path(exists=True))
@click.argument("figure_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path(exists=True))
def main(data_filepath, figure_filepath, output_filepath):
    print("Training day and night")
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--lr", default=0.1)
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[4:])
    print(args)

    logger = logging.getLogger(__name__)
    logger.info("training model")

    model = MyAwesomeModel()
    train_set = loadTrain(data_filepath)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    epochs = 15
    loss_values = []
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            log_probs = model(images)
            loss = F.nll_loss(log_probs, labels)
            running_loss += loss.item() * images.size(0)
            model.zero_grad()
            loss.backward()
            optimizer.step()
        loss_values.append(running_loss / len(train_loader))
        print(f"Epoch: {e}/{epochs}")
        print(f"Loss: {loss}")
        print(f"Running loss: {running_loss}")
    plt.figure()
    plt.plot(loss_values)
    plt.savefig(figure_filepath + "/loss_values.png", bbox_inches="tight")
    torch.save(model, output_filepath + "/trained_model.pt")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
