import torch
from torch.utils.data import TensorDataset


def loadTrain(data_filepath):
    train_images = torch.load(data_filepath + "/train_images.pt")
    train_labels = torch.load(data_filepath + "/train_labels.pt")
    train_set = TensorDataset(train_images, train_labels)
    return train_set
