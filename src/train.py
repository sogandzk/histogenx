import os
import optuna
from pytorch_lightning.callbacks import ModelCheckpoint
from optuna.integration import PyTorchLightningPruningCallback
from datasets import load_from_disk
from utils.config import PROJECT_DIR
from utils.run_train import run_train

def objective(trial, dataset):
    # ------------------------------------------------------------------------------------------------------------------------------------------
    hparams = {
        "max_epochs": 100,
        "k_patches": trial.suggest_categorical("k_patches", [4, 8, 16]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]), # [32, 64, 128, 256]
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True), # 1e-6, 1e-4
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True), # 1e-6, 1e-4
        "temperature": trial.suggest_float("temperature", 0.05, 0.3), # 0.01, 0.3
        "dropout1": 0.2, #trial.suggest_float("dropout1", 0, 0.6),
        "dropout2": 0.3, #trial.suggest_float("dropout2", 0, 0.6),
    }
    hparams["img_hidden_dims"] = [512, 256]
    hparams["expr_hidden_dims"] = [512, 256]
    hparams["img_dropout_probs"] = [hparams["dropout1"]] + [hparams["dropout2"]] * (len(hparams["img_hidden_dims"]) - 1)
    hparams["expr_dropout_probs"] = [hparams["dropout1"]] + [hparams["dropout2"]] * (len(hparams["expr_hidden_dims"]) - 1)
    # ----------------------------------------------------------------------------------------------------------------------------------------

    trial.set_user_attr("hparams", hparams)

    ckpt_dir = os.path.join(PROJECT_DIR, "checkpoints", trial.study.study_name, str(trial.number))

    callbacks = [
        PyTorchLightningPruningCallback(
            trial, 
            monitor="val_loss"
        ),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,       # best model only
        )
    ]
    
    val_loss = run_train(dataset, hparams, callbacks)

    return val_loss

def main():
    dataset = load_from_disk(os.path.join(PROJECT_DIR, "data", "dataset"))

    os.makedirs(os.path.join(PROJECT_DIR, "experiments"), exist_ok=True)

    for tag in ['A', 'B', 'C', 'D']:
        seed = ord(tag)
        study = optuna.create_study(
            study_name=f"May_11_2026_{tag}",
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
