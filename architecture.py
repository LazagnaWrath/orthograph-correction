import torch
import torch.nn as nn
from torch.nn import functional as F

class Seq2Seq(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, attention_dim, pad_id):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.encoder = Encoder(embedding_dim, hidden_size, self.embedding)
        self.decoder = Decoder(hidden_size, embedding_dim, attention_dim, vocab_size, self.embedding)

    def forward(self, src, sos_id, eos_id, pad_id, valid_mask, max_len=20): # (B, T_src), valid_mask : (B, T_src) -> bool

        encoder_outputs, encoder_hidden = self.encoder(src, valid_mask) # (B, T, H), (B, H) -> step context, -1 context
        generated_tokens = self.decoder.generate(
            encoder_outputs, encoder_hidden, 
            sos_id, eos_id, pad_id, 
            valid_mask, 
            max_len
        ) 
        return generated_tokens

    def forward_train(self, src, tgt, sos_id, pad_id): # (B, T_src), (B, T_tgt) -> (B,[<SOS>,true,<EOF>])

        valid_mask = src != pad_id # (B, T_src) -> bool
        B, T_tgt = tgt.shape

        encoder_outputs, encoder_hidden = self.encoder(src, valid_mask) # (B, T, H), (B, H) -> step context, -1 context
        
        decoder_hidden = encoder_hidden
        prev_token = torch.full((B,), sos_id, dtype=torch.long, device=encoder_outputs.device) # (B * <SOS>,) 

        predicted_logits = []

        for i in range(T_tgt-1):

            logits, decoder_hidden = self.decoder.forward_step(
                prev_token,
                encoder_outputs,
                decoder_hidden,
                valid_mask
            ) # (B, vocab_size), (B, H)
            predicted_logits.append(logits) 

            prev_token = tgt[:, i+1] # (B,) -> teacher forcing

        return torch.stack(predicted_logits, dim=1) # (B, T, vocab_size)

class GRU(nn.Module):

    def __init__(self, embedding_dim, hidden_size):
        super().__init__()

        self.Wz = nn.Linear(embedding_dim, hidden_size)                                                  
        self.Uz = nn.Linear(hidden_size, hidden_size)                                                  

        self.Wr = nn.Linear(embedding_dim, hidden_size)                                                  
        self.Ur = nn.Linear(hidden_size, hidden_size)                                                  

        self.Wx = nn.Linear(embedding_dim, hidden_size)                                                  
        self.Uh = nn.Linear(hidden_size, hidden_size)                                                  

        self.Sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def gru_cell(self, x_t, h_prev):

        r = self.Sig( self.Wr(x_t) + self.Ur(h_prev) ) # (B, C) @ (C, H) + (B, H) @ (H, H) -> (B, H) + (B, H)
        z = self.Sig( self.Wz(x_t) + self.Uz(h_prev) ) # (B, C) @ (C, H) + (B, H) @ (H, H) -> (B, H) + (B, H)

        candidate = self.tanh( self.Wx(x_t) + r * self.Uh(h_prev)) # (B, C) @ (C, H) + (B, H) * (B, H) @ (H, H) -> (B, H) + (B, H)
        h = (1-z) * h_prev + z * candidate # (1 - (B, H)) * (B, H) + (B, H) * (B, H) -> (B, H) + (B, H)

        return h

    def forward(self, x, valid_mask = None):
        B, T, _ = x.shape

        h = x.new_zeros(B, self.Uz.in_features) # (B, H)
        outputs = []

        for t in range(T):
            new_h = self.gru_cell(x[:, t, :], h) # (B, H)

            if valid_mask is not None:
                h = torch.where(
                    valid_mask[:, t].unsqueeze(1),
                    new_h,
                    h
                )
            else:
                h = new_h
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1) # (B, T, H)
        return outputs, h
    

class Cross_Attention(nn.Module):

    def __init__(
        self,
        hidden_size,
        attention_dim
    ):
        super().__init__()

        self.Wq = nn.Linear(hidden_size, attention_dim)
        self.Wk = nn.Linear(hidden_size, attention_dim)
        self.Wv = nn.Linear(hidden_size, attention_dim)

        self.scale = attention_dim ** -0.5

    def forward(self, encoder_output, decoder_hidden, valid_mask):

        Q = self.Wq(decoder_hidden) # (B, H) @ (H, D) -> (B, D)
        K = self.Wk(encoder_output) # (B, T, H) @ (H, D) -> (B, T, D)
        V = self.Wv(encoder_output) # (B, T, H) @ (H, D) -> (B, T, D)

        scores = Q.unsqueeze(1) @ K.transpose(-2, -1) * self.scale # (B, 1, D) @ (B, D, T) -> (B, 1, T)
        scores = scores.masked_fill(
            ~valid_mask.unsqueeze(1),
                float("-inf")
            )
        weights = F.softmax(scores, dim=-1) # (B, 1, T)
        context = (weights @ V).squeeze(1) # (B, 1, T) @ (B, T, D) -> (B, 1, D) and (B, D)

        return weights, context


class Encoder(nn.Module):

    def __init__(self, embedding_dim, hidden_size, embedding):
        super().__init__()

        self.embedding = embedding
        self.gru = GRU(embedding_dim, hidden_size)

    def forward(self, src, valid_mask):
        # src : (B, T)
        x = self.embedding(src) # (B, T, D)

        outputs, hidden = self.gru(x, valid_mask) # (B, T, H) and (B, H) -> -1
        return outputs, hidden

class Decoder(nn.Module):

    def __init__(self, hidden_size, embedding_dim, attention_dim, vocab_size, embedding):

        super().__init__()

        self.gru = GRU(embedding_dim + attention_dim, hidden_size)
        self.cross_attention = Cross_Attention(hidden_size, attention_dim)
        self.embedding = embedding
        self.output_proj = nn.Linear(
            hidden_size, vocab_size
        )

    def forward_step(self, prev_token, encoder_outputs, decoder_hidden, valid_mask): # -> (B,), (B, T, H), (B, H), (B, T) -> bool

        _, context = self.cross_attention(encoder_outputs, decoder_hidden, valid_mask) # (B, D) -> score relation
        prev_token = self.embedding(prev_token) # (B, 16)
        st = torch.cat((prev_token, context), dim=1) # cat((B, D), (B, 16)) -> (B, D+16)
        decoder_hidden = self.gru.gru_cell(st, decoder_hidden) # (B, H)
        logits = self.output_proj(decoder_hidden) # (B, vocab_size)

        return logits, decoder_hidden # (B, vocab_size), (B, H)

    def generate(self, encoder_outputs, encoder_hidden, sos_id, eos_id, pad_id, valid_mask, max_len=20): # -> (B, T, H), (B, H), valid_mask :(B, T) -> bool

        B = encoder_outputs.shape[0] # n batch

        prev_token = torch.full( # 1 sos * batch
            (B,),
            sos_id,
            dtype=torch.long,
            device=encoder_outputs.device
        )

        hidden = encoder_hidden # (B, H)
        generated = []
        finished = torch.zeros(B, dtype=torch.bool, device=encoder_outputs.device)

        for t in range(max_len):

            logits, hidden = self.forward_step(prev_token, encoder_outputs, hidden, valid_mask) # (B, vocab_size), (B, H)
            token = torch.argmax(logits, -1) # (B,) 

            token = token.masked_fill(finished, pad_id) # replace token by <PAD> when finished
            finished = finished | (token == eos_id) # check and set true if finished

            generated.append(token)
            prev_token = token
            
            if finished.all():
                break

        generated_tokens = torch.stack(generated, dim=1)

        return generated_tokens