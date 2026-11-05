import torch, config, architecture

Model = architecture.Seq2Seq(**config.model_config)
weights = torch.load(config.BASE_DIR / "weights" / "M2.pth")
Model.load_state_dict(weights)

while True :

    answer = input("Mot que l'on veut corriger : ")
    word = config.encode(answer)
    generated = Model.forward(
        src = torch.tensor(word, dtype=torch.long).unsqueeze(0),
        sos_id= config.char_to_id["<SOS>"],
        eos_id= config.char_to_id["<EOS>"],
        pad_id= config.char_to_id["<PAD>"],
        valid_mask= torch.ones((1,len(word)), dtype=torch.bool)
    )
    print(f'Resultat : {config.decode(generated[0].tolist())}')