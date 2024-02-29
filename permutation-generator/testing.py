from utilities import *
from config import *
from dataloading import *
from transformer import *
from tqdm import tqdm

SHOW_PLOTS = False

# get the model
model = BigramLanguageModel()

# send to gpu (maybe)
if torch.cuda.is_available():
  device = "cuda:0"
else:
  device = "cpu"

model = nn.DataParallel(model)
model = model.to(device)

# reload the model
filename = PATH + "/model/" + MODELNAME + ".pth"
model.load_state_dict(torch.load(filename, map_location=torch.device(device)))

# test for all sequences

results = []

for seq in tqdm(test_data, desc="Testing"):
  real_perm = tuple(get_permutation(seq))
  gen_perm = tuple(model.module.generate(seq))
  results.append(int(real_perm == gen_perm))

print(f"Accuracy: {sum(results) / len(results)}")


# write results to a file that r can read
print("Writing results to file")
with open(PATH + "/results/" + MODELNAME + ".csv", "w") as f:
  f.write("results\n")
  for r in results:
    f.write(f"{r}\n")
  f.write("\n")