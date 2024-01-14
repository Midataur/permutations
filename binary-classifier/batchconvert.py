from utilities import *
from config import *

# batch converts indices into sequences

with open("mistakes.txt", "r") as file:
    mistakes = file.readlines()

with open("converted.txt", "a") as file:
    for mistake in mistakes:
        mistake = mistake.strip()
        sequence = int_to_seq(int(mistake))
        file.write(str(sequence).join("") + "\n")