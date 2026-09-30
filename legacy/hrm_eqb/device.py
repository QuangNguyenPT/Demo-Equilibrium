"""Device selection with a hard VRAM cap.

On Windows the NVIDIA driver's sysmem fallback lets CUDA allocations spill into system RAM
instead of failing; under sustained training that paging stalled the driver and the machine
crashed (DPC_WATCHDOG_VIOLATION, 2026-09-27). Capping PyTorch's allocator turns the same
situation into an ordinary, recoverable out-of-memory error.
"""

import torch

MEMORY_FRACTION = 0.7


def select_device(prefer_cuda: bool = True, memory_fraction: float = MEMORY_FRACTION) -> torch.device:
    """CUDA with the allocator capped at memory_fraction of VRAM, else CPU."""
    if prefer_cuda and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        return torch.device("cuda")
    return torch.device("cpu")
