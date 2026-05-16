import numpy as np
import soundfile as sf
import torch


def load_audio(path):
    """Load audio as a torch tensor with shape (channels, frames)."""
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    samples = np.asarray(samples).T
    return torch.from_numpy(samples), sample_rate
