import sys

from pytorch_lightning.callbacks import Callback

from .schedulers import WarmupPolyLRScheduler


class StatsCallback(Callback):
    """Callback to log and print training stats each epoch."""

    def on_validation_end(self, trainer, pl_module):
        """Print stats at end of each epoch."""
        metrics = trainer.callback_metrics

        # Extract metrics
        train_loss = metrics.get("train/loss", None)
        val_loss = metrics.get("val/loss", None)
        val_auroc = metrics.get("val/AUC_ROC", None)
        val_acc = metrics.get("val/ACC", None)

        # Get learning rate from optimizer
        lr = trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else None

        epoch = trainer.current_epoch + 1

        # Format and print
        parts = [f"Epoch {epoch:3d}"]
        if train_loss is not None:
            parts.append(f"train_loss: {train_loss:.6f}")
        if val_loss is not None:
            parts.append(f"val_loss: {val_loss:.6f}")
        if val_auroc is not None:
            parts.append(f"val_auroc: {val_auroc:.4f}")
        if val_acc is not None:
            parts.append(f"val_acc: {val_acc:.4f}")
        if lr is not None:
            parts.append(f"lr: {lr:.6f}")

        print(" | ".join(parts))
        sys.stdout.flush()


class LRSchedulerCallback(Callback):
    """Callback to step the warmup poly LR scheduler each epoch."""

    def __init__(self, config):
        """
        Args:
            config: Dict with keys initial_lr, max_lr, max_steps, warmup_steps, poly_exp
        """
        self.config = config
        self.scheduler = None

    def on_train_epoch_start(self, trainer, pl_module):
        """Initialize scheduler on first epoch."""
        if self.scheduler is None:
            self.scheduler = WarmupPolyLRScheduler(
                optimizer=trainer.optimizers[0],
                initial_lr=self.config["initial_lr"],
                max_lr=self.config["max_lr"],
                max_steps=self.config["max_steps"],
                warmup_steps=self.config["warmup_steps"],
                exponent=self.config["poly_exp"],
            )

    def on_train_epoch_end(self, trainer, pl_module):
        """Step the scheduler at the end of each epoch."""
        if self.scheduler is not None:
            self.scheduler.step()
