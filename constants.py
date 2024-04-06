import pathlib
import torch

# ROOT = '/scratch/tri/noisy_labels/'
ROOT = pathlib.Path().resolve()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

global_config = {}
