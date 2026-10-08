"""Training loop for mapping networks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from marn.trainers.lr_finder import LRFinderResult

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]
from torch.optim.lr_scheduler import LRScheduler
from marn.callbacks.base import Callback
from marn.config.trainer import TrainerConfig
from marn.losses.base import BaseLoss
from marn.losses.context import TrainingContext
from marn.losses.mapping_loss import MappingLoss
from marn.losses.outputs import LossOutput
from marn.losses.task import RegressionLoss
from marn.models.mapping_model import MappingModel
from marn.mappers.base import BaseMapper
from marn.models.forward_result import ForwardResult
from marn.trainers.batch_adapter import BatchAdapter, TupleBatchAdapter


logger = logging.getLogger("marn.trainer")

_OPTIMIZER_MAP: dict[str, type[torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

_SCHEDULER_MAP: dict[str, type[LRScheduler]] = {
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    "step": torch.optim.lr_scheduler.StepLR,
    "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "exponential": torch.optim.lr_scheduler.ExponentialLR,
}


def _resolve_device(device_str: str) -> torch.device:
    """Resolve ``"auto"`` to CUDA if available, else CPU."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


class MappingTrainer:
    """Training loop for mapping networks.

    The trainer owns optimization only: forward, compute loss, backward,
    optimizer step, validation, and callbacks.  Mapping logic stays in the
    model/generator/loss subsystems.

    Args:
        model: A ``MappingModel`` instance.
        train_loader: Training ``DataLoader``.
        val_loader: Optional validation ``DataLoader``.
        loss_fn: A ``MappingLoss`` or ``BaseLoss`` instance.  If ``None``,
            defaults to ``MappingLoss(RegressionLoss())``.
        config: Trainer configuration.  If ``None``, uses defaults.
        callbacks: List of ``Callback`` instances.
        batch_adapter: Adapter for unpacking dataloader batches.  Defaults
            to ``TupleBatchAdapter``.

    Example::

        trainer = MappingTrainer(
            model=mapping_model,
            train_loader=train_loader,
            loss_fn=MappingLoss(ClassificationLoss()),
            config=TrainerConfig(max_epochs=50, learning_rate=3e-4),
            callbacks=[EarlyStopping(patience=5), MetricLogger()],
        )
        history = trainer.fit()
    """

    def __init__(
        self,
        model: MappingModel,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any] | None = None,
        loss_fn: MappingLoss | BaseLoss | None = None,
        config: TrainerConfig | None = None,
        callbacks: list[Callback] | None = None,
        batch_adapter: BatchAdapter | None = None,
    ) -> None:
        self.config = config or TrainerConfig()
        self.device = _resolve_device(self.config.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_adapter = batch_adapter or TupleBatchAdapter()
        self.callbacks = list(callbacks or [])
        self.should_stop = False
        self.current_epoch = 0

        # Resolve loss function
        if loss_fn is None:
            self.loss_fn: MappingLoss | BaseLoss = MappingLoss(task_loss=RegressionLoss())
        elif isinstance(loss_fn, MappingLoss):
            self.loss_fn = loss_fn
        elif isinstance(loss_fn, BaseLoss):
            self.loss_fn = MappingLoss(task_loss=loss_fn)
        else:
            raise TypeError(f"loss_fn must be a BaseLoss or MappingLoss, got {type(loss_fn)}")

        if isinstance(self.loss_fn, nn.Module):
            self.loss_fn = self.loss_fn.to(self.device)

        # Build optimizer
        self.optimizer = self._build_optimizer()

        # Build scheduler
        self.scheduler = self._build_scheduler()

        # AMP
        self.scaler: GradScaler | None = None
        if self.config.amp_enabled:
            self.scaler = GradScaler()

        # Reproducibility
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

    @property
    def learning_rate(self) -> float:
        """Get the current learning rate configured for training."""
        return self.config.learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        """Set the learning rate, updating the config and the optimizer in-place."""
        self.config.learning_rate = value
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = value

    def lr_find(
        self,
        min_lr: float = 1e-7,
        max_lr: float = 10.0,
        num_iterations: int = 100,
        divergence_threshold: float = 4.0,
        beta: float = 0.95,
    ) -> LRFinderResult:
        """Run a Learning Rate Finder range sweep to suggest a good learning rate.

        Args:
            min_lr: Lower bound of learning rate range to search.
            max_lr: Upper bound of learning rate range to search.
            num_iterations: Number of training steps to sweep over.
            divergence_threshold: Stop search early if loss exceeds this multiple
                of the best loss so far.
            beta: Exponential moving average smoothing factor for loss values.

        Returns:
            An ``LRFinderResult`` instance containing sweep stats and suggestion.
        """
        from marn.trainers.lr_finder import LRFinder

        finder = LRFinder(
            self,
            min_lr=min_lr,
            max_lr=max_lr,
            num_iterations=num_iterations,
            divergence_threshold=divergence_threshold,
            beta=beta,
        )
        return finder.run()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        opt_name = self.config.optimizer.lower()
        opt_cls = _OPTIMIZER_MAP.get(opt_name)
        if opt_cls is None:
            available = ", ".join(sorted(_OPTIMIZER_MAP))
            raise ValueError(f"Unknown optimizer {opt_name!r}; available: {available}")

        kwargs: dict[str, Any] = {
            "lr": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
        }
        params = list(self.model.parameters())
        if isinstance(self.loss_fn, MappingLoss) and self.loss_fn.trainable_coefficients:
            params.extend(self.loss_fn.parameters())
        optimizer = opt_cls(params, **kwargs)
        assert isinstance(optimizer, torch.optim.Optimizer)
        return optimizer

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        if self.config.scheduler is None:
            return None

        sched_name = self.config.scheduler.lower()
        sched_cls = _SCHEDULER_MAP.get(sched_name)
        if sched_cls is None:
            available = ", ".join(sorted(_SCHEDULER_MAP))
            raise ValueError(f"Unknown scheduler {sched_name!r}; available: {available}")

        kwargs = dict(self.config.scheduler_kwargs)
        if sched_name == "cosine" and "T_max" not in kwargs:
            kwargs["T_max"] = self.config.max_epochs
        if sched_name == "step" and "step_size" not in kwargs:
            kwargs["step_size"] = max(1, self.config.max_epochs // 3)
        if sched_name == "exponential" and "gamma" not in kwargs:
            kwargs["gamma"] = 0.95

        return sched_cls(self.optimizer, **kwargs)

    def fit(self, auto_lr_find: bool = False) -> dict[str, list[float]]:
        """Run the full training loop.

        Args:
            auto_lr_find: If *True*, runs the learning rate finder before
                training and automatically sets the suggested learning rate.

        Returns:
            History dict mapping metric names to lists of per-epoch values.
        """
        if auto_lr_find:
            logger.info("Running learning rate finder before training...")
            lr_finder = self.lr_find()
            suggested = lr_finder.suggestion()
            logger.info("Setting learning rate to suggested value: %.2e", suggested)
            self.learning_rate = suggested

        history: dict[str, list[float]] = {"train_loss": []}
        if self.val_loader is not None:
            history["val_loss"] = []

        self._fire_callbacks("on_fit_start")

        try:
            for epoch in range(self.config.max_epochs):
                self.current_epoch = epoch

                if self.should_stop:
                    break

                self._fire_callbacks("on_epoch_start", epoch=epoch)

                # Training
                train_metrics = self._train_one_epoch()
                history["train_loss"].append(train_metrics["train_loss"])

                # Validation
                val_metrics: dict[str, float] = {}
                if self.val_loader is not None:
                    self._fire_callbacks("on_validation_start")
                    val_metrics = self._validate()
                    history.setdefault("val_loss", []).append(val_metrics.get("val_loss", 0.0))
                    self._fire_callbacks("on_validation_end", metrics=val_metrics)

                # Epoch end
                epoch_metrics = {**train_metrics, **val_metrics}
                self._fire_callbacks("on_epoch_end", epoch=epoch, metrics=epoch_metrics)

                # Scheduler step
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        monitor_val = epoch_metrics.get("val_loss", epoch_metrics["train_loss"])
                        self.scheduler.step(monitor_val)
                    else:
                        self.scheduler.step()

        except KeyboardInterrupt:
            logger.info("Training interrupted at epoch %d", self.current_epoch + 1)
        finally:
            self._fire_callbacks("on_fit_end")

        return history

    def _train_one_epoch(self) -> dict[str, float]:
        """Run one training epoch and return metrics."""
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            self._fire_callbacks("on_batch_start", batch_idx=batch_idx)

            loss_output = self._train_step(batch, batch_idx)

            self._fire_callbacks("on_batch_end", batch_idx=batch_idx, loss_output=loss_output)

            running_loss += loss_output.metrics.get("task", loss_output.total.detach().item())
            num_batches += 1

            if self.should_stop:
                break

        avg_loss = running_loss / max(num_batches, 1)
        return {"train_loss": avg_loss}

    def _train_step(self, batch: Any, batch_idx: int) -> LossOutput:
        """Execute one training step with accumulation, AMP, and clipping."""
        inputs_tuple, targets = self.batch_adapter.unpack(batch)
        inputs_tuple = self._to_device(inputs_tuple)
        targets = self._to_device_single(targets)

        # Forward
        amp_device = "cuda" if self.device.type == "cuda" else "cpu"
        with autocast(device_type=amp_device, enabled=self.config.amp_enabled):
            result = self.model(*inputs_tuple)
            context = self._create_training_context(result, targets, inputs_tuple)
            loss_output = self.loss_fn(context)
            assert isinstance(loss_output, LossOutput)
            loss = loss_output.total / self.config.accumulation_steps

        # Backward
        if self.scaler is not None:
            self.scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        else:
            loss.backward()  # type: ignore[no-untyped-call]

        self._clear_mapper_weights()

        # Optimizer step (every accumulation_steps batches)
        if (batch_idx + 1) % self.config.accumulation_steps == 0:
            # Gradient clipping
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            if self.config.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            if self.config.gradient_clip_value is not None:
                nn.utils.clip_grad_value_(self.model.parameters(), self.config.gradient_clip_value)

            # Step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

        return loss_output

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        """Run validation and return metrics."""
        assert self.val_loader is not None
        self.model.eval()
        running_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            inputs_tuple, targets = self.batch_adapter.unpack(batch)
            inputs_tuple = self._to_device(inputs_tuple)
            targets = self._to_device_single(targets)

            result = self.model(*inputs_tuple)
            context = self._create_training_context(result, targets, inputs_tuple)
            loss_output = self.loss_fn(context)
            assert isinstance(loss_output, LossOutput)

            self._clear_mapper_weights()

            running_loss += loss_output.total.item()
            num_batches += 1

        avg_loss = running_loss / max(num_batches, 1)
        return {"val_loss": avg_loss}

    def _create_training_context(
        self,
        result: ForwardResult,
        targets: Any,
        inputs_tuple: tuple[Any, ...],
    ) -> TrainingContext:
        perturbed_predictions = None
        mapper_weights = {}

        if isinstance(self.loss_fn, MappingLoss):
            # 1. Stability Loss — populate perturbed_predictions for any
            #    non-None stability loss, not just specific subclasses.
            if self.loss_fn.stability_loss is not None:
                # Extract perturbation parameters from the loss if available,
                # otherwise use sensible defaults.
                stability_loss = self.loss_fn.stability_loss
                eps = getattr(stability_loss, "epsilon", 0.01)
                num_samples = getattr(stability_loss, "num_samples", 1)

                perturbed_outputs = []
                for _ in range(num_samples):
                    perturbed_latents = {}
                    for name, latent in result.latent_vectors.items():
                        noise = torch.randn_like(latent)
                        perturbed_latents[name] = latent + eps * noise

                    perturbed_res = self.model(*inputs_tuple, latents=perturbed_latents)
                    perturbed_outputs.append(perturbed_res.predictions)

                if num_samples == 1:
                    perturbed_predictions = perturbed_outputs[0]
                else:
                    if isinstance(perturbed_outputs[0], tuple):
                        tuple_len = len(perturbed_outputs[0])
                        stacked_tuples = []
                        for i in range(tuple_len):
                            stacked_tuples.append(
                                torch.stack([out[i] for out in perturbed_outputs], dim=0)
                            )
                        perturbed_predictions = tuple(stacked_tuples)
                    else:
                        perturbed_predictions = torch.stack(perturbed_outputs, dim=0)

            # 2. Alignment Loss — populate mapper_weights for any
            #    non-None alignment loss.
            if self.loss_fn.alignment_loss is not None:
                mapper_weights = self.model.modulated_mapper_weights()

        return TrainingContext(
            predictions=result.predictions,
            targets=targets,
            latent_vectors=result.latent_vectors,
            generated_parameters=result.generated_parameters,
            perturbed_predictions=perturbed_predictions,
            mapper_weights=mapper_weights,
        )

    def _clear_mapper_weights(self) -> None:
        for module in self.model.modules():
            if isinstance(module, BaseMapper):
                module._last_modulated_weight = None

    def _to_device(self, items: tuple[Any, ...]) -> tuple[Any, ...]:
        """Move tuple elements to the trainer's device."""
        return tuple(item.to(self.device) if isinstance(item, Tensor) else item for item in items)

    def _to_device_single(self, item: Any) -> Any:
        """Move a single item to the trainer's device."""
        if isinstance(item, Tensor):
            return item.to(self.device)
        return item

    def _fire_callbacks(self, hook_name: str, **kwargs: Any) -> None:
        """Call a hook on all callbacks with error wrapping."""
        for callback in self.callbacks:
            method = getattr(callback, hook_name, None)
            if method is None:
                continue
            try:
                if hook_name in ("on_fit_start", "on_fit_end", "on_validation_start"):
                    method(self)
                elif hook_name in ("on_epoch_start",):
                    method(self, kwargs["epoch"])
                elif hook_name in ("on_epoch_end",):
                    method(self, kwargs["epoch"], kwargs["metrics"])
                elif hook_name == "on_batch_start":
                    method(self, kwargs["batch_idx"])
                elif hook_name == "on_batch_end":
                    method(self, kwargs["batch_idx"], kwargs["loss_output"])
                elif hook_name == "on_validation_end":
                    method(self, kwargs["metrics"])
            except Exception as e:
                cb_name = type(callback).__name__
                raise RuntimeError(f"Callback {cb_name!r} failed during {hook_name}: {e}") from e
