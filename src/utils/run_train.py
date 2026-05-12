import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from .dataset_class import Multi_Modal_Dataset
from .trainer_class import LitMultiModal

torch.set_float32_matmul_precision("high")

def run_train(dataset, hparams, callbacks=[]):

    train_loader = DataLoader(
        Multi_Modal_Dataset(dataset["train"], k_patches=16, n_views=20),
        batch_size=hparams["batch_size"],
        shuffle=True, 
        num_workers=max(1, min(os.cpu_count() - 2, 4)), 
        prefetch_factor=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        Multi_Modal_Dataset(dataset["valid"], k_patches="all", n_views=1),
        batch_size=len(dataset["valid"]),
        num_workers=max(1, min(os.cpu_count() - 2, 4)),
        prefetch_factor=4,
        pin_memory=True
    )

    callbacks = callbacks + [
        EarlyStopping(
            monitor="val_loss", 
            patience=5, 
            mode="min", 
            min_delta=1e-4
        )
    ]

    ckpt = any(isinstance(cb, ModelCheckpoint) for cb in callbacks)

    trainer = pl.Trainer(
        max_epochs=hparams["max_epochs"],
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed", # Recommended solution for modern GPUs
        callbacks=callbacks,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
        enable_checkpointing=ckpt,
        enable_progress_bar=True,
        logger=False, # no log
        num_sanity_val_steps=1 # Minimal sanity check
    )

    model = LitMultiModal(hparams)

    trainer.fit(model, train_loader, val_loader)
    
    val_loss = trainer.callback_metrics["val_loss"].item()

    return val_loss
