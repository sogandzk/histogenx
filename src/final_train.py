from utils.trainer_class import LitMultiModal
import optuna
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from datasets import load_from_disk
from utils.dataset_class import Multi_Modal_Dataset
from utils.config import PROJECT_DIR
from utils.trainer_class import LitMultiModal
from pytorch_lightning.loggers import CSVLogger


dataset = load_from_disk(os.path.join(PROJECT_DIR, "data", "dataset"))

optuna_db_path = os.path.join(
    PROJECT_DIR,
    "experiments",
    "optuna.db"
)

study = optuna.load_study(
    study_name="Dec_26_2025_A",
    storage=f"sqlite:///{optuna_db_path}"
)

best_params = study.best_trial.params

best_trial = study.best_trial

print("Best Validation Score:", best_trial.value)
print("Best Hyperparameters:", best_trial.params)


# ------------------------------------------------------------------------------------------------------------------------------------------
hparams = {
    "max_epochs": 100,
    "batch_size": best_params['batch_size'],
    "learning_rate": best_params['learning_rate'],
    "weight_decay": best_params['weight_decay'],
    "temperature": best_params['temperature'],
    "dropout1": 0.2,
    "dropout2": 0.3,
}
hparams["img_hidden_dims"] = [1024, 512]
hparams["expr_hidden_dims"] = [1024, 512]
hparams["img_dropout_probs"] = [hparams["dropout1"]] + [hparams["dropout2"]] * (len(hparams["img_hidden_dims"]) - 1)
hparams["expr_dropout_probs"] = [hparams["dropout1"]] + [hparams["dropout2"]] * (len(hparams["expr_hidden_dims"]) - 1)
# ----------------------------------------------------------------------------------------------------------------------------------------



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

model = LitMultiModal(hparams)

callbacks = [
    EarlyStopping(
        monitor="val_loss", 
        patience=20,
        mode="min", 
        min_delta=1e-4
    ),
    ModelCheckpoint(
        dirpath=os.path.join(PROJECT_DIR, "checkpoints"),
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,       # best model only
        save_last=True,     # also save last epoch
    )
]

logger = CSVLogger(save_dir=os.path.join(PROJECT_DIR, "logs"), name="")

trainer = pl.Trainer(
    max_epochs=hparams["max_epochs"],
    accelerator="gpu",
    devices=1,
    precision="bf16-mixed", # Recommended solution for modern GPUs
    callbacks=callbacks,
    
    gradient_clip_algorithm="norm",
    gradient_clip_val=1.0,

    enable_checkpointing=True,
    enable_progress_bar=True,
    
    logger=logger,

    num_sanity_val_steps=1 # Minimal sanity check
)

trainer.fit(model, train_loader, val_loader)



