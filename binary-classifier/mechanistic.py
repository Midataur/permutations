import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from utilities import *
from config import *
from dataloading import *
from tqdm import tqdm
from transformer import *
import transformer_lens
import os

print("Logging in...")
wandb.login()

# this is the training script, assumes you're using the transformer
# if you're using the MLP, you'll need to change the data pipeline and the final dimension
# also you can modify the transformer config in the transformer.py file

# setup the model
model = BigramLanguageModel()

# cuda? (gpu)
if torch.cuda.is_available():
  device = "cuda:0"
else:
  device = "cpu"

# send to gpu (maybe)
model = nn.DataParallel(model)
model = model.to(device)

# load the model
filename = PATH + "/model/" + MODELNAME + ".pth"
if os.path.isfile(filename):
    model.load_state_dict(torch.load(filename, map_location=torch.device(device)))
else:
   raise Exception("Model not found")

print(model)