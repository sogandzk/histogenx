import torch
import pytorch_lightning as pl
from .img_encoder import ImgEncoder
from .expr_encoder import ExprEncoder
from .loss_function import contrastive_loss
from .config import PROJECT_DIR
torch.set_float32_matmul_precision("high")

class LitMultiModal(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.img_encoder = ImgEncoder(
            in_dim=1843,
            hidden_dims=hparams["img_hidden_dims"],
            out_dim=128,
            dropout_probs=hparams["img_dropout_probs"],
            batchnorm=True # [*]
        )

        self.expr_encoder = ExprEncoder(
            in_dim=2000,
            hidden_dims=hparams["expr_hidden_dims"],
            out_dim=128,
            dropout_probs=hparams["expr_dropout_probs"],
            batchnorm=True # [*]
        )

    def forward(self, img_features, expr_features):
        return self.img_encoder(img_features), self.expr_encoder(expr_features)

    def training_step(self, batch, batch_idx):
        img_features, ks, expr_features, _ = batch
        img_emb, expr_emb = self(img_features, expr_features)
        loss = contrastive_loss(img_emb, ks, expr_emb, self.hparams.temperature)
        self.log("train_loss",
                 loss, 
                 on_step=False, # for GPU Efficiency, syncs only once per epoch.
                 on_epoch=True, 
                 prog_bar=True,
                 batch_size=len(ks))
        return loss

    def validation_step(self, batch, batch_idx):
        img_features, ks, expr_features, _ = batch
        img_emb, expr_emb = self(img_features, expr_features)
        loss = contrastive_loss(img_emb, ks, expr_emb, self.hparams.temperature)
        self.log("val_loss", 
                 loss, 
                 on_epoch=True, # "Early Stopping" checks the average value once the entire validation loop is finished.
                 prog_bar=True,
                 batch_size=len(ks))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch", # Update LR every epoch
            },
        }
