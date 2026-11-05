import csv, random
from pathlib import Path

VOISINS_QWERTY = {
    "q": ["w", "a"],
    "w": ["q", "e", "a", "s"],
    "e": ["w", "r", "s", "d"],
    "r": ["e", "t", "d", "f"],
    "t": ["r", "y", "f", "g"],
    "y": ["t", "u", "g", "h"],
    "u": ["y", "i", "h", "j"],
    "i": ["u", "o", "j", "k"],
    "o": ["i", "p", "k", "l"],
    "p": ["o", "l"],
    "a": ["s", "q", "w", "z"],
    "s": ["a", "d", "w", "e", "z", "x"],
    "d": ["s", "f", "e", "r", "x", "c"],
    "f": ["d", "g", "r", "t", "c", "v"],
    "g": ["f", "h", "t", "y", "v", "b"],
    "h": ["g", "j", "y", "u", "b", "n"],
    "j": ["h", "k", "u", "i", "n", "m"],
    "k": ["j", "l", "i", "o", "m"],
    "l": ["k", "o", "p"],
    "z": ["x", "a", "s"],
    "x": ["z", "c", "s", "d"],
    "c": ["x", "v", "d", "f"],
    "v": ["c", "b", "f", "g"],
    "b": ["v", "n", "g", "h"],
    "n": ["b", "m", "h", "j"],
    "m": ["n", "j", "k"],
}

assets_folder = Path(__file__).resolve().parent / "assets"
french_dict_path = assets_folder / "mots_courants_fr.txt"

with open(french_dict_path, "r", encoding="utf-8") as f:
    text = f.read()

words = str.split(text, "\n")
data_len = 500
dataset = {
    "fault" : [],
    "correct" : random.sample(words, data_len)
}

def add_random_fault(word, index):

    fault_type = random.choice(["double", "forget", "addneighbor", "neightbor"])
    target_char = word[index]

    realised = False

    if fault_type == "double":
        word = word[:index] + target_char + word[index:]
        realised = True
    elif fault_type == "forget":
        word = word[:index] + word[index+1:]
        realised = True
    elif fault_type == "addneighbor" and target_char in VOISINS_QWERTY:
        random_neighbor = random.choice(VOISINS_QWERTY[target_char])
        word = word[:index] + random_neighbor + word[index:]
        realised = True
    elif fault_type == "neightbor" and target_char in VOISINS_QWERTY:
        random_neighbor = random.choice(VOISINS_QWERTY[target_char])
        word = word[:index] + random_neighbor + word[index+1:]
        realised = True

    return word, fault_type, realised

for i, corr_word in enumerate(dataset["correct"]):

    fault_word = corr_word

    max_fault = random.choices([0,1,2,3], weights=[1, 5, 3, 1], k=1)[0]
    nb_fault = 0
    index = 0

    while index < len(fault_word):
        p = random.random()
        if p < 0.3 and nb_fault < max_fault:
            nb_fault += 1
            fault_word, fault_type, realised = add_random_fault(fault_word, index)
            if fault_type == "double" or fault_type == "addneighbor" and realised:
                index += 1
        index += 1

    dataset["fault"].insert(i, fault_word)

folder_dir = Path(__file__).resolve().parent
with open(folder_dir / "datasets" / "test_2.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(dataset.keys())      
    writer.writerows(zip(*dataset.values()))