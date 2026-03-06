import warnings

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import config
from data import DummyMRIDataset, Random3DAugmentation
from models import ResidualEncoderClsLightning
from training import LRSchedulerCallback, StatsCallback

warnings.filterwarnings("ignore", "No positive samples in targets")


def main():
    torch.set_float32_matmul_precision("medium")

    # Create base dataset from three .b2nd cases
    base_dataset = DummyMRIDataset(
        num_classes=config.NUM_CLASSES,
        data_dir=config.DATA_DIR,
    )

    # Create augmentation pipeline
    train_augment = Random3DAugmentation(**config.AUG_CONFIG)

    # Create train and validation datasets
    train_dataset = DummyMRIDataset(
        num_classes=config.NUM_CLASSES,
        volumes=base_dataset.volumes,
        targets=base_dataset.targets,
        augment=train_augment,
    )
    val_dataset = DummyMRIDataset(
        num_classes=config.NUM_CLASSES,
        volumes=base_dataset.volumes,
        targets=base_dataset.targets,
        augment=None,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    # Create model
    model = ResidualEncoderClsLightning(
        in_ch=config.IN_CHANNELS,
        out_ch=config.NUM_CLASSES,
        spatial_dims=config.SPATIAL_DIMS,
        final_layer_dropout=config.FINAL_LAYER_DROPOUT,
        optimizer=torch.optim.SGD,
        optimizer_kwargs={
            "lr": config.INITIAL_LR,
            "weight_decay": config.WEIGHT_DECAY,
            "momentum": config.MOMENTUM,
            "nesterov": config.NESTEROV,
        },
    )

    # Load pretrained encoder weights
    model.load_pretrained_unet_encoder(
        checkpoint_path=config.PRETRAINED_CHECKPOINT,
        strict=False,
        map_location="cpu",
        verbose=False,
    )

    # Configure scheduler for callback
    model.warmup_poly_config = {
        "initial_lr": config.INITIAL_LR,
        "max_lr": config.MAX_LR,
        "max_steps": config.NUM_EPOCHS,
        "warmup_steps": config.WARMUP_EPOCHS,
        "poly_exp": config.POLY_EXPONENT,
    }

    # Create callbacks and trainer
    callbacks = [StatsCallback(), LRSchedulerCallback(model.warmup_poly_config)]
    trainer = pl.Trainer(
        max_epochs=config.NUM_EPOCHS,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=False,
        log_every_n_steps=1,
        callbacks=callbacks,
        logger=False,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    print("Training complete!")
    print("Final model state saved in trainer.model")

    return model, trainer


if __name__ == "__main__":
    model, trainer = main()
