import os
import optuna
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import load_from_disk
from utils.config import PROJECT_DIR
from utils.run_train import run_train

dataset = load_from_disk(os.path.join(PROJECT_DIR, "data", "dataset"))

def main():
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


    ckpt_dir = os.path.join(PROJECT_DIR, "checkpoints", "selected")

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,       # best model only
        )
    ]

    val_loss = run_train(dataset, hparams, callbacks)



if __name__ == "__main__":
    main()

