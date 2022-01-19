from tests import _PATH_DATA

from torchvision.datasets import MNIST

dataset = MNIST(_PATH_DATA)
assert len(dataset) == N_train for training and N_test for test
assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
assert that all labels are represented