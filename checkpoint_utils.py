import torch


def load_checkpoint_into_model(model, checkpoint_path, device, expected_num_clusters=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint_num_clusters = checkpoint.get("num_clusters")
        if (
            expected_num_clusters is not None
            and checkpoint_num_clusters is not None
            and checkpoint_num_clusters != expected_num_clusters
        ):
            raise ValueError(
                f"Checkpoint num_clusters={checkpoint_num_clusters} does not match "
                f"requested num_clusters={expected_num_clusters}."
            )
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return checkpoint
