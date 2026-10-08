"""Learning Rate Finder for mapping networks."""

from __future__ import annotations

import copy
import logging
import math
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
import torch.distributed as dist

if TYPE_CHECKING:
    from marn.trainers.trainer import MappingTrainer

logger = logging.getLogger("marn.trainer.lr_finder")


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DDP and torch.compile wrappers to get the underlying MappingModel."""
    curr = model
    while True:
        if hasattr(curr, "_orig_mod"):
            orig = getattr(curr, "_orig_mod")
            if isinstance(orig, nn.Module):
                curr = orig
            else:
                break
        elif hasattr(curr, "module"):
            mod = getattr(curr, "module")
            if isinstance(mod, nn.Module):
                curr = mod
            else:
                break
        else:
            break
    return curr


def _suggest_lr_steep(lrs: list[float], smoothed_losses: list[float]) -> float | None:
    if len(lrs) < 5:
        return lrs[0] if lrs else None

    # Use log-scale for learning rates
    log_lrs = [math.log10(lr) for lr in lrs]

    # Find the index of the minimum smoothed loss
    min_loss_idx = min(range(len(smoothed_losses)), key=lambda i: smoothed_losses[i])

    # If the minimum loss is too early, we fall back to searching the entire sweep
    # (minus the last 5 steps which are likely diverging) to find a steep descent region.
    if min_loss_idx < 3:
        end_idx = max(3, len(smoothed_losses) - 5)
    else:
        end_idx = min_loss_idx

    end_idx = min(end_idx, len(smoothed_losses) - 1)

    # Calculate gradients of smoothed losses w.r.t log10(lr)
    grads = []
    for i in range(end_idx):
        dx = log_lrs[i + 1] - log_lrs[i]
        dy = smoothed_losses[i + 1] - smoothed_losses[i]
        grads.append(dy / dx if dx != 0 else 0.0)

    # Smooth the gradients to avoid noise
    window = 3
    if len(grads) >= window:
        smoothed_grads = []
        for i in range(len(grads)):
            start = max(0, i - window // 2)
            end = min(len(grads), i + window // 2 + 1)
            smoothed_grads.append(sum(grads[start:end]) / (end - start))
    else:
        smoothed_grads = grads

    if not smoothed_grads:
        return lrs[0]

    # Find index of the minimum (steepest descent) gradient
    steepest_idx = min(range(len(smoothed_grads)), key=lambda i: smoothed_grads[i])

    return lrs[steepest_idx]


def _suggest_lr_valley(lrs: list[float], smoothed_losses: list[float]) -> float | None:
    n = len(smoothed_losses)
    if n < 5:
        return lrs[0] if lrs else None

    # Longest Decreasing Subsequence (LDS) to find the longest valley of learning
    lds = [1] * n
    for i in range(1, n):
        for j in range(0, i):
            if (smoothed_losses[i] < smoothed_losses[j]) and (lds[i] < lds[j] + 1):
                lds[i] = lds[j] + 1

    max_idx = max(range(n), key=lambda i: lds[i])
    max_len = lds[max_idx]
    max_start = max_idx - max_len

    # Suggest learning rate 2/3 of the way through the valley
    suggest_idx = max_start + 2 * max_len // 3
    suggest_idx = max(0, min(suggest_idx, n - 1))

    return lrs[suggest_idx]


def _suggest_lr_minimum(lrs: list[float], smoothed_losses: list[float]) -> float | None:
    if not lrs:
        return None
    min_loss_idx = min(range(len(smoothed_losses)), key=lambda i: smoothed_losses[i])
    # Suggest 1/10th of the learning rate at minimum loss
    return lrs[min_loss_idx] / 10.0


def _suggest_lr(
    lrs: list[float],
    losses: list[float],
    smoothed_losses: list[float],
    method: str = "steep",
) -> float | None:
    """Find a good learning rate suggestion based on the chosen heuristic."""
    method = method.lower()
    if method == "steep":
        return _suggest_lr_steep(lrs, smoothed_losses)
    elif method == "valley":
        return _suggest_lr_valley(lrs, smoothed_losses)
    elif method == "minimum":
        return _suggest_lr_minimum(lrs, smoothed_losses)
    else:
        raise ValueError(
            f"Unknown suggestion method: {method!r}. Choose from: 'steep', 'valley', 'minimum'"
        )


class LRFinderResult:
    """Result of the learning rate range test.

    Attributes:
        lrs: List of learning rates swept during the finder.
        losses: List of raw batch losses.
        smoothed_losses: List of exponentially smoothed batch losses.
    """

    def __init__(
        self,
        lrs: list[float],
        losses: list[float],
        smoothed_losses: list[float],
        suggested_lr: float | None,
    ) -> None:
        self.lrs = lrs
        self.losses = losses
        self.smoothed_losses = smoothed_losses
        self._suggested_lr = suggested_lr

    def suggestion(self, method: str = "steep") -> float:
        """Get the suggested learning rate.

        Args:
            method: The heuristic to use for the suggestion. Choose from
                ``"steep"`` (steepest gradient), ``"valley"`` (longest valley),
                or ``"minimum"`` (minimum loss / 10).

        Returns:
            The recommended learning rate.

        Raises:
            ValueError: If no suggestion was found or an unknown method is requested.
        """
        # Return pre-configured suggested_lr for backward compatibility in mock tests
        if self._suggested_lr is not None and method == "steep":
            return self._suggested_lr

        suggested = _suggest_lr(self.lrs, self.losses, self.smoothed_losses, method=method)
        if suggested is None:
            raise ValueError(f"No suggested learning rate found using method {method!r}.")
        return suggested

    def plot(self, show: bool = True, suggest: bool = True, method: str = "steep") -> Any:
        """Plot the loss versus learning rate.

        Args:
            show: Whether to show the plot using ``plt.show()``.
            suggest: Whether to highlight the suggested learning rate in the plot.
            method: The suggestion method to highlight. Choose from ``"steep"``,
                ``"valley"``, or ``"minimum"``.

        Returns:
            The matplotlib Figure object.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for plotting the learning rate finder results. "
                "Install it using `pip install matplotlib`."
            ) from e

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.lrs, self.smoothed_losses, label="Smoothed Loss", color="royalblue", lw=2)
        ax.plot(self.lrs, self.losses, label="Raw Loss", color="lightgray", alpha=0.6, lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Learning Rate Finder", fontsize=14, fontweight="bold")
        ax.grid(True, which="both", ls="--", alpha=0.5)

        suggested = None
        if suggest:
            try:
                suggested = self.suggestion(method=method)
            except ValueError:
                pass

        if suggest and suggested is not None:
            # Find closest learning rate index to place the point
            idx = min(range(len(self.lrs)), key=lambda i: abs(self.lrs[i] - suggested))
            ax.scatter(
                [suggested],
                [self.smoothed_losses[idx]],
                color="red",
                s=100,
                zorder=5,
                label=f"Suggested LR ({method}): {suggested:.2e}",
            )

        ax.legend(fontsize=10, loc="upper right")
        if show:
            plt.show()
        return fig


class LRFinder:
    """Executes Leslie Smith's Learning Rate Range Test."""

    def __init__(
        self,
        trainer: MappingTrainer,
        min_lr: float = 1e-7,
        max_lr: float = 10.0,
        num_iterations: int = 100,
        divergence_threshold: float = 4.0,
        beta: float = 0.95,
    ) -> None:
        self.trainer = trainer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_iterations = num_iterations
        self.divergence_threshold = divergence_threshold
        self.beta = beta

    def run(self) -> LRFinderResult:
        """Run the range sweep over iterations."""
        trainer = self.trainer

        # Save states
        unwrapped = _unwrap_model(trainer.model)
        model_state = {k: v.clone() for k, v in unwrapped.state_dict().items()}
        optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())

        scheduler_state = None
        if trainer.scheduler is not None:
            scheduler_state = copy.deepcopy(trainer.scheduler.state_dict())

        scaler_state = None
        if trainer.scaler is not None:
            scaler_state = copy.deepcopy(trainer.scaler.state_dict())

        orig_epoch = trainer.current_epoch
        orig_should_stop = trainer.should_stop
        orig_callbacks = trainer.callbacks
        orig_lr = trainer.config.learning_rate
        orig_accumulation_steps = trainer.config.accumulation_steps

        # RNG state
        rng_state = torch.get_rng_state()
        cuda_rng_state = None
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state()

        # Temporarily prepare trainer state for sweep
        trainer.callbacks = []
        trainer.config.accumulation_steps = 1

        lrs: list[float] = []
        losses: list[float] = []
        smoothed_losses: list[float] = []

        loader_iter = iter(trainer.train_loader)

        beta = self.beta
        best_loss = float("inf")
        smoothed_loss = 0.0

        trainer.optimizer.zero_grad()
        trainer.model.train()

        try:
            for step in range(self.num_iterations):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(trainer.train_loader)
                    batch = next(loader_iter)

                # Compute exponential learning rate
                current_lr = self.min_lr * (self.max_lr / self.min_lr) ** (
                    step / max(1, self.num_iterations - 1)
                )

                # Update optimizer learning rate
                for param_group in trainer.optimizer.param_groups:
                    param_group["lr"] = current_lr

                # Run step (incorporating AMP, device mapping, stable losses, mapper weight clear)
                loss_output = trainer._train_step(batch, step)
                loss_val = loss_output.total.item()

                # Distributed reduction
                if dist.is_available() and dist.is_initialized():
                    loss_tensor = torch.tensor(loss_val, device=trainer.device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    loss_val = loss_tensor.item() / dist.get_world_size()

                lrs.append(current_lr)
                losses.append(loss_val)

                if step == 0:
                    smoothed_loss = loss_val
                else:
                    smoothed_loss = beta * smoothed_loss + (1 - beta) * loss_val

                smoothed_losses.append(smoothed_loss)

                if smoothed_loss < best_loss or step == 0:
                    best_loss = smoothed_loss

                # Stop if loss diverges
                if step > 0 and smoothed_loss > self.divergence_threshold * best_loss:
                    logger.info(
                        "Learning rate finder reached divergence threshold; stopping early."
                    )
                    break

        finally:
            # Restore original trainer states
            unwrapped.load_state_dict(model_state)
            trainer.optimizer.load_state_dict(optimizer_state)

            if trainer.scheduler is not None and scheduler_state is not None:
                trainer.scheduler.load_state_dict(scheduler_state)

            if trainer.scaler is not None and scaler_state is not None:
                trainer.scaler.load_state_dict(scaler_state)

            trainer.current_epoch = orig_epoch
            trainer.should_stop = orig_should_stop
            trainer.callbacks = orig_callbacks
            trainer.config.learning_rate = orig_lr
            trainer.config.accumulation_steps = orig_accumulation_steps

            # Clear mapper weights
            trainer._clear_mapper_weights()

            # Restore RNG
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)

        suggested_lr = _suggest_lr(lrs, losses, smoothed_losses, method="steep")
        return LRFinderResult(lrs, losses, smoothed_losses, suggested_lr)
