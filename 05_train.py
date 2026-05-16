import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import S2UTDataset, make_collate_fn
from model import S2UTModel
from s2ut_config import add_num_clusters_arg, build_experiment_config, ensure_parent_dir

TRAIN_CSV = "./data/train.csv"
DEV_CSV = "./data/dev.csv"
BATCH_SIZE = 16 
EPOCHS = 20
LEARNING_RATE = 5e-4


def parse_args():
    parser = argparse.ArgumentParser(description="Train the S2UT model for a configurable unit inventory.")
    add_num_clusters_arg(parser)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--extra-epochs", type=int, default=0,
                        help="Continue training for N more epochs on top of what's in the checkpoint.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the existing checkpoint (restores model, optimizer, epoch, best val loss).")
    return parser.parse_args()


def _fmt(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main():
    args = parse_args()
    config = build_experiment_config(args.num_clusters)
    device = get_device()
    print(f"Training device: {device}")
    print(f"Using K = {config.num_clusters}, vocab_size = {config.vocab_size}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    script_start = time.time()

    print("Loading training split...")
    train_dataset = S2UTDataset(
        TRAIN_CSV,
        config.units_json,
        config.bos_token,
        config.eos_token,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(config.pad_token),
    )

    print("Loading development split...")
    dev_dataset = S2UTDataset(
        DEV_CSV,
        config.units_json,
        config.bos_token,
        config.eos_token,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(config.pad_token),
    )

    model = S2UTModel(vocab_size=config.vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_token)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    best_val_loss = float('inf')
    start_epoch = 1
    total_epochs = args.epochs

    if args.resume and os.path.exists(config.checkpoint_path):
        ckpt = torch.load(config.checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float('inf'))
        total_epochs = start_epoch - 1 + args.extra_epochs if args.extra_epochs else ckpt.get("epoch", 0) + args.epochs
        print(f"Resumed from epoch {start_epoch - 1}, best val loss so far: {best_val_loss:.4f}")
        print(f"  Will train until epoch {total_epochs}")
    elif args.resume:
        print(f"--resume was set, but no checkpoint was found at {config.checkpoint_path}; starting from scratch.")

    for epoch in range(start_epoch, total_epochs + 1):
        t_epoch = time.time()
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]")
        
        for fbanks, units in train_pbar:
            fbanks = fbanks.to(device)
            units = units.to(device)

            tgt_input = units[:, :-1] 
            tgt_expect = units[:, 1:] 

            optimizer.zero_grad()
            outputs = model(fbanks, tgt_input)

            outputs_flatten = outputs.reshape(-1, outputs.shape[-1])
            tgt_expect_flatten = tgt_expect.reshape(-1)
            
            loss = criterion(outputs_flatten, tgt_expect_flatten)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(dev_loader, desc=f"Epoch {epoch}/{total_epochs} [Val]")
        
        with torch.no_grad():
            for fbanks, units in val_pbar:
                fbanks = fbanks.to(device)
                units = units.to(device)

                tgt_input = units[:, :-1]
                tgt_expect = units[:, 1:]

                outputs = model(fbanks, tgt_input)
                outputs_flatten = outputs.reshape(-1, outputs.shape[-1])
                tgt_expect_flatten = tgt_expect.reshape(-1)
                
                loss = criterion(outputs_flatten, tgt_expect_flatten)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(dev_loader)
        
        print(f"Epoch {epoch} summary | Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f} | Elapsed: {_fmt(time.time() - t_epoch)}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ensure_parent_dir(config.checkpoint_path)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "num_clusters": config.num_clusters,
                    "vocab_size": config.vocab_size,
                },
                config.checkpoint_path,
            )
            print(f"New best validation loss. Saved checkpoint to {config.checkpoint_path}")
            
    print(f"Training finished. Total time: {_fmt(time.time() - script_start)}")

if __name__ == "__main__":
    main()
