import torch, data_loader, config, architecture
from functools import partial
from torch.nn import functional as F
from torch.utils.data import DataLoader

Model = architecture.Seq2Seq(**config.model_config)
optim = torch.optim.Adam(params= Model.parameters(), lr=config.LR)

pairs = data_loader.load_data('V2.csv')
test_data = data_loader.load_data('test_2.csv')

dataloader = DataLoader(
    dataset=pairs,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    collate_fn= partial(data_loader.collate_fn, pad_id = 0),
    pin_memory=True,
    pin_memory_device='cuda'
)
valid_data = DataLoader(
    dataset=test_data,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    collate_fn= partial(data_loader.collate_fn, pad_id = 0),
)
@torch.no_grad()
def estimate_loss():
    global Model, valid_data, test_data

    Model.eval()
    total_loss = 0

    for src, tgt in valid_data:

        logits = Model.forward_train(src, tgt, config.char_to_id["<SOS>"], config.char_to_id["<PAD>"]) # (B, T_tgt -1, vocab_size)
        loss = F.cross_entropy(
            logits.reshape(-1, config.model_config["vocab_size"]), 
            tgt[:, 1:].reshape(-1),ignore_index=config.char_to_id["<PAD>"]
        )
        total_loss += loss.item()

    Model.train()
    return total_loss/(len(test_data)//config.BATCH_SIZE)

for epoch in range(config.N_EPOCH):

    total_loss = 0

    for src, tgt in dataloader:

        optim.zero_grad()

        logits = Model.forward_train(src, tgt, config.char_to_id["<SOS>"], config.char_to_id["<PAD>"]) # (B, T_tgt -1, vocab_size)

        loss = F.cross_entropy(
            logits.reshape(-1, config.model_config["vocab_size"]), 
            tgt[:, 1:].reshape(-1),ignore_index=config.char_to_id["<PAD>"]
        )
        loss.backward()
        optim.step()

        total_loss += loss

    validation_loss = estimate_loss()
    print(f"epoch : {epoch}, Loss : {(total_loss/(len(pairs)//config.BATCH_SIZE)):.4f}, test : {validation_loss} ")

torch.save(Model.state_dict(), config.BASE_DIR / "weights" / 'M2.pth')
