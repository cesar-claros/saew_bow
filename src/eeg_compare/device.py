"""Device selection shared by the retained models."""

import torch


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve a device spec (None/"cpu"/"cuda"/"cuda:N"/"auto") to a torch.device."""
    if device is None or device == "cpu":
        return torch.device("cpu")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
