from utilities import *
from config import *

#testing

# batch converts indices into sequences

with open("mistakes.txt", "r") as file:
    mistakes = file.readlines()

non_fours = 0

with open("converted.txt", "w") as file:
    for mistake in mistakes:
        mistake = mistake.strip()
        sequence = int_to_seq(int(mistake))
        file.write("".join([str(x) for x in sequence]) + "\n")
        if 4 not in sequence:
            non_fours += 1

print("non fours:", non_fours)