"""Distributed Data Parallel utilities for mapping networks.

Design notes
------------
``MappingModel`` wraps a frozen target model plus a ``ParameterGenerator``.
The generator holds two kinds of tensors:

* **Trainable latent vectors** (``requires_grad=True``) — these *must* be
  synchronized across DDP ranks so every worker computes gradients w.r.t. the
  same latent state.
* **Fixed mapper buffers** (``requires_grad=False``, registered via
  ``register_buffer``) — these are initialised identically on every rank (same
  seed, same architecture) and never receive gradients, so they do **not** need
  gradient synchronisation.

``wrap_ddp`` passes ``find_unused_parameters=False`` because the target model
parameters are not ``nn.Parameter`` objects and the generator always exercises
all of its latent vectors during ``generate_parameters()``.  If you attach a
custom generator that may skip some latent vectors, pass
``find_unused_parameters=True`` explicitly.

Example::

    import torch.distributed as dist
    from marn.distributed import setup_ddp, wrap_ddp, cleanup_ddp

    def train(rank: int, world_size: int) -> None:
        setup_ddp(rank, world_size)
        model = MappingModel(target, latent_dim=64)
        ddp_model = wrap_ddp(model, device_ids=[rank])
        # ... training loop ...
        cleanup_ddp()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from marn.models.mapping_model import MappingModel

logger = logging.getLogger("marn.distributed")


def is_ddp_available() -> bool:
    """Return *True* if ``torch.distributed`` is available on this system.

    This checks both the compiled-in backend support and whether the runtime
    can be initialized (it does **not** call ``init_process_group``).

    Returns:
        ``True`` when distributed training is supported.
    """
    return dist.is_available()


def setup_ddp(
    rank: int,
    world_size: int,
    backend: str = "gloo",
    init_method: str = "env://",
    **kwargs: Any,
) -> None:
    """Initialize the default distributed process group.

    Call this once at the start of each worker process before constructing the
    model.  Use :func:`cleanup_ddp` at the end of training to destroy the group.

    Args:
        rank: Global rank of this process (0-based).
        world_size: Total number of processes.
        backend: Distributed backend — ``"gloo"`` (CPU/cross-platform),
            ``"nccl"`` (CUDA multi-GPU, recommended for GPU training), or
            ``"mpi"``.  Defaults to ``"gloo"``.
        init_method: URL for rendezvous.  Defaults to ``"env://"`` which reads
            ``MASTER_ADDR`` / ``MASTER_PORT`` environment variables.
        **kwargs: Additional keyword arguments forwarded to
            :func:`torch.distributed.init_process_group`.

    Raises:
        RuntimeError: If ``torch.distributed`` is not available.
    """
    if not dist.is_available():
        raise RuntimeError(
            "torch.distributed is not available on this system. "
            "Rebuild PyTorch with distributed support or use single-process training."
        )
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
        **kwargs,
    )
    logger.info("DDP initialized: rank=%d/%d backend=%s", rank, world_size, backend)


def cleanup_ddp() -> None:
    """Destroy the default distributed process group.

    Call this at the end of each worker process to cleanly release resources.
    Safe to call even if the process group was never initialized.
    """
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
        logger.info("DDP process group destroyed")


def wrap_ddp(
    model: MappingModel,
    device_ids: list[int] | None = None,
    output_device: int | None = None,
    find_unused_parameters: bool = False,
    **kwargs: Any,
) -> torch.nn.parallel.DistributedDataParallel:
    """Wrap a :class:`~marn.models.MappingModel` for DDP training.

    **Synchronization scope**: DDP synchronizes gradients for all
    ``nn.Parameter`` tensors in the wrapped module.  In a ``MappingModel``:

    * Trainable latent vectors are ``nn.Parameter`` objects and *will* be
      synchronized — this is correct and desired.
    * Frozen mapper weights are registered as *buffers* (``requires_grad=False``)
      and are **not** synchronized — they are byte-for-byte identical on every
      rank because they share the same random seed and initialization, so no
      communication is needed.
    * The frozen target model's parameters are also registered as buffers and
      excluded from synchronization for the same reason.

    Args:
        model: A :class:`~marn.models.MappingModel` instance that
            has already been moved to the target device (e.g. ``model.cuda()``).
        device_ids: List of GPU device IDs for this process.  Pass ``None`` for
            CPU-only (gloo) training.
        output_device: Device where output will be gathered.  Defaults to
            ``device_ids[0]`` when ``device_ids`` is set.
        find_unused_parameters: Set to ``True`` only if your generator may skip
            some latent vectors during certain forward passes.  The default
            ``False`` avoids the overhead of the unused-parameter scan.
        **kwargs: Additional keyword arguments forwarded to
            :class:`torch.nn.parallel.DistributedDataParallel`.

    Returns:
        A ``DistributedDataParallel``-wrapped model.  Access the underlying
        ``MappingModel`` via ``ddp_model.module``.

    Raises:
        RuntimeError: If the process group has not been initialized.

    Example::

        setup_ddp(rank=0, world_size=1)
        model = MappingModel(nn.Linear(4, 2), latent_dim=8).cpu()
        ddp_model = wrap_ddp(model)
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available; cannot wrap with DDP.")
    if not dist.is_initialized():
        raise RuntimeError("Process group is not initialized. Call setup_ddp() before wrap_ddp().")

    ddp = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=output_device,
        find_unused_parameters=find_unused_parameters,
        **kwargs,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())
    logger.info(
        "Model wrapped with DDP (trainable=%d  buffers=%d  find_unused=%s)",
        trainable,
        buffers,
        find_unused_parameters,
    )
    return ddp
