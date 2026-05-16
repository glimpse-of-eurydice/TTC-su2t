import _repo_path  # noqa: F401
import torch
import torchaudio
import transformers

print(f"PyTorch Version: {torch.__version__}")
print(f"Torchaudio Version: {torchaudio.__version__}")
print(f"Transformers Version: {transformers.__version__}")

# Check whether Apple Silicon MPS acceleration is available.
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(f"MPS acceleration is available. Test tensor: {x}")
else:
    print("MPS acceleration is not available in this PyTorch environment.")
