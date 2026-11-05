from pathlib import Path

alphabet = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"] + list("abcdefghijklmnopqrstuvwxyz êéàèçâîôûçïüë'-")
char_to_id = {char:i for i, char in enumerate(alphabet)}
id_to_char = {i: char for char, i in char_to_id.items()}

encode = lambda c: [char_to_id[s] for s in c]
decode = lambda l: ''.join([id_to_char[i] for i in l])

model_config = {
    "vocab_size" : len(alphabet),
    "embedding_dim": 16,
    "hidden_size": 64,
    "attention_dim": 32,
    "pad_id": char_to_id["<SOS>"]
}
B, T = 4, 10
BASE_DIR = Path(__file__).resolve().parent

LR = 1e-3
N_EPOCH = 100
BATCH_SIZE = 32

EVAL_ITERS = 2