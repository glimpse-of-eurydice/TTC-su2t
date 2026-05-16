import argparse
import torch
import torch.nn as nn
import math

from s2ut_config import add_num_clusters_arg, build_experiment_config

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)

class Conv1dSubsampler(nn.Module):
    """Reduce speech sequence length by a factor of four."""
    def __init__(self, input_dim=80, d_model=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, stride=2, padding=1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, stride=2, padding=1),
            nn.GLU(dim=1)
        )
        self.out_proj = nn.Linear(d_model // 2, d_model)

    def forward(self, x, src_key_padding_mask=None):
        # x: (batch, time, mel bins) -> (batch, channels, time)
        x = x.transpose(1, 2)
        x = self.conv(x)
        # Return to (batch, time / 4, d_model).
        x = x.transpose(1, 2)
        x = self.out_proj(x)
        
        if src_key_padding_mask is not None:
            mask = src_key_padding_mask[:, ::4]
            return x, mask
        return x, None

class S2UTModel(nn.Module):
    """Transformer speech-to-unit translation model."""
    def __init__(self, input_dim=80, vocab_size=103, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.d_model = d_model
        
        self.subsampler = Conv1dSubsampler(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.unit_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_decoder = PositionalEncoding(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1
        )
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz, device):
        """Generate a causal mask for autoregressive decoding."""
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src, src_key_padding_mask = self.subsampler(src, src_key_padding_mask)
        src = self.pos_encoder(src)
        
        tgt = self.unit_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_decoder(tgt)
        
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, tgt.device)
        
        out = self.transformer(
            src, tgt, 
            tgt_mask=tgt_mask, 
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        return self.fc_out(out)

def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test the S2UT model with a configurable vocabulary size.")
    add_num_clusters_arg(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = build_experiment_config(args.num_clusters)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Testing model on {device}...")
    
    dummy_src = torch.randn(4, 706, 80).to(device)
    dummy_tgt = torch.randint(0, config.vocab_size, (4, 152)).to(device)
    
    model = S2UTModel(vocab_size=config.vocab_size).to(device)
    
    tgt_input = dummy_tgt[:, :-1]
    output = model(dummy_src, tgt_input)
    
    print("=== Model forward pass succeeded ===")
    print(f"Output shape (batch, target sequence length, vocabulary size): {output.shape}")
    print(f"Expected shape: (4, 151, {config.vocab_size})")
