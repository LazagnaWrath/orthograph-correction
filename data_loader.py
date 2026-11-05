import torch, config, csv
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, pad_id):
    src_list, tgt_list = zip(*batch)

    src = pad_sequence(
        src_list,
        batch_first=True,
        padding_value=pad_id
    )

    tgt = pad_sequence(
        tgt_list,
        batch_first=True,
        padding_value=pad_id
    )

    return src, tgt

def prepare_data(data):
    new_data = []
    for e1, e2 in data:
        new_data.append(
            (torch.tensor(config.encode(e1), dtype=torch.long),
             torch.tensor(config.encode(["<SOS>"] + list(e2) + ["<EOS>"]), dtype=torch.long))
        )
    return new_data

def load_data(file_name):
    full_path = config.BASE_DIR / "datasets" / file_name
    with open(full_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        pairs = [(row["fault"], row["correct"]) for row in reader]

    pairs = prepare_data(pairs)
    return pairs