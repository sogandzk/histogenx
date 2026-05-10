import os
import gc
import torch
import optuna
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback
from datasets import load_from_disk
from utils.dataset_class import Multi_Modal_Dataset
from utils.config import PROJECT_DIR
from utils.trainer_class import LitMultiModal

torch.set_float32_matmul_precision("high")

def objective(trial, dataset):
    # ------------------------------------------------------------------------------------------------------------------------------------------
    hparams = {
        "max_epochs": 100,
        "k_patches": trial.suggest_categorical("k_patches", [4, 8, 16]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True),
        "temperature": trial.suggest_float("temperature", 0.01, 0.3),
        "dropout1": 0.2, #trial.suggest_float("dropout1", 0, 0.6),
        "dropout2": 0.3, #trial.suggest_float("dropout2", 0, 0.6),
        # "img_depth": trial.suggest_categorical("img_depth", ["shallow", "deep"]),
    }
    hparams["img_hidden_dims"] = [512, 256]
    hparams["expr_hidden_dims"] = [512, 256]
    hparams["img_dropout_probs"] = [hparams["dropout1"]] + [hparams["dropout2"]] * (len(hparams["img_hidden_dims"]) - 1)
    hparams["expr_dropout_probs"] = [hparams["dropout1"]] + [hparams["dropout2"]] * (len(hparams["expr_hidden_dims"]) - 1)
    # ----------------------------------------------------------------------------------------------------------------------------------------

    # trial_dir = os.path.join(PROJECT_DIR, "experiments", "trials", f"trial_{trial.number}")

    train_loader = DataLoader(
        Multi_Modal_Dataset(dataset["train"], k_patches=hparams["k_patches"]),
        collate_fn=Multi_Modal_Dataset.collate_fn,
        batch_size=hparams["batch_size"],
        shuffle=True, 
        num_workers=max(1, min(os.cpu_count() - 2, 4)), 
        pin_memory=True
    )
    val_loader = DataLoader(
        Multi_Modal_Dataset(dataset["valid"], k_patches="all"),
        collate_fn=Multi_Modal_Dataset.collate_fn,
        batch_size=len(dataset["valid"]),
        num_workers=max(1, min(os.cpu_count() - 2, 4)),
        pin_memory=True
    )
    
    model = LitMultiModal(hparams)
    callbacks = [
        PyTorchLightningPruningCallback(trial, monitor="val_loss"),
        EarlyStopping(monitor="val_loss", patience=5, mode="min", min_delta=1e-4)
    ]

    # logger = CSVLogger(save_dir=os.path.join(trial_dir, "logs"), name="")

    trainer = pl.Trainer(
        max_epochs=hparams["max_epochs"],
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed", # Recommended solution for modern GPUs
        callbacks=callbacks,
        # gradient_clip_algorithm="norm",
        # gradient_clip_val=1.0,
        enable_checkpointing=False,
        enable_progress_bar=True,
        logger=False, # no log
        num_sanity_val_steps=1 # Minimal sanity check
    )

    trainer.fit(model, train_loader, val_loader)

    # Get the metric to return to Optuna
    val_loss = trainer.callback_metrics["val_loss"].item()

    # Clean up to prevent "Too many open files"
    del trainer, model, train_loader, val_loader
    gc.collect()

    return val_loss



def main():
    dataset = load_from_disk(os.path.join(PROJECT_DIR, "data", "dataset"))

    os.makedirs(os.path.join(PROJECT_DIR, "experiments"), exist_ok=True)

    for tag in ['A', 'B', 'C', 'D']:
        seed = ord(tag)
        study = optuna.create_study(
            study_name=f"May_10_2025_{tag}",
            direction="minimize",
            sampler=optuna.samplers.TPESampler(n_startup_trials=20, seed=seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=40, n_warmup_steps=20, interval_steps=5),
            storage=f"sqlite:///{PROJECT_DIR}/experiments/optuna.db",
            load_if_exists=True
        )
        study.optimize(lambda trial: objective(trial, dataset), n_trials=100)

    print("*** DONE ***")

if __name__ == "__main__":
    main()

