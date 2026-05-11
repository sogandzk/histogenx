import os
import gc
import torch
import optuna
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from optuna.integration import PyTorchLightningPruningCallback
from datasets import load_from_disk
from utils.dataset_class import Multi_Modal_Dataset
from utils.config import PROJECT_DIR
from utils.trainer_class import LitMultiModal

dataset = load_from_disk(os.path.join(PROJECT_DIR, "data", "dataset"))

optuna_db_path = os.path.join(
    PROJECT_DIR,
    "experiments",
    "optuna-V1.db"
)

study = optuna.load_study(
    study_name="May_9_2025_A",
    storage=f"sqlite:///{optuna_db_path}"
)


best_trial = study.best_trial
best_value = best_trial.value
best_params = best_trial.params

print("====================================")
print("Best Value:", best_value)
print("Best Params:", best_params)
print("====================================")

# ----------------------------------------------------------------------------------------------------------------------
hparams = {
    "max_epochs": 100,
    "k_patches": best_params["k_patches"],
    "batch_size": best_params["batch_size"],
    "learning_rate": best_params["learning_rate"],
    "weight_decay": best_params["weight_decay"],
    "temperature": best_params["temperature"],
    "dropout1": 0.2,
    "dropout2": 0.3
}
hparams["img_hidden_dims"] = [512, 256]
hparams["expr_hidden_dims"] = [512, 256]
hparams["img_dropout_probs"] = [hparams["dropout1"]] + [hparams["dropout2"]] * (len(hparams["img_hidden_dims"]) - 1)
hparams["expr_dropout_probs"] = [hparams["dropout1"]] + [hparams["dropout2"]] * (len(hparams["expr_hidden_dims"]) - 1)
# ----------------------------------------------------------------------------------------------------------------------


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
    EarlyStopping(monitor="val_loss", patience=5, mode="min", min_delta=1e-4),
    ModelCheckpoint(
        dirpath=os.path.join(PROJECT_DIR, "checkpoints"),
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,       # best model only
        save_last=True,     # also save last epoch
    )
]

trainer = pl.Trainer(
    max_epochs=hparams["max_epochs"],
    accelerator="gpu",
    devices=1,
    precision="bf16-mixed", # Recommended solution for modern GPUs
    callbacks=callbacks,
    # gradient_clip_algorithm="norm",
    # gradient_clip_val=1.0,
    enable_checkpointing=True,
    enable_progress_bar=True,
    logger=False, # no log
    num_sanity_val_steps=1 # Minimal sanity check
)

trainer.fit(model, train_loader, val_loader)
