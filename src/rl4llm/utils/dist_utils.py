import torch
import torch.distributed as dist


def gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Safely gather tensors of variable lengths using padding."""
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()

    tensor = tensor.contiguous().to(tensor.device)

    # Get sizes from all ranks
    local_size = torch.tensor(tensor.numel(), device=tensor.device, dtype=torch.long)
    sizes = [torch.empty_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    max_size = max(s.item() for s in sizes)

    # Pad tensor to max size
    if tensor.numel() < max_size:
        pad_size = max_size - tensor.numel()
        padded_tensor = torch.cat([tensor, torch.full((pad_size,), float('nan'), device=tensor.device)])
    else:
        padded_tensor = tensor

    # Gather padded tensors
    padded_tensors = [torch.empty_like(padded_tensor) for _ in range(world_size)]
    dist.all_gather(padded_tensors, padded_tensor)

    # Remove padding and combine
    gathered = []
    for t, s in zip(padded_tensors, sizes):
        gathered.append(t[: s.item()])
    return torch.cat(gathered)
