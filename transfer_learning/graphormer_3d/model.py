import torch
import bitsandbytes as bnb
import numpy as np
import pytorch_lightning as pl

from collections import defaultdict, namedtuple
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch

from utils.reporting import (
    get_cls_metrics_binary_pt,
    get_cls_metrics_multilabel_pt,
    get_cls_metrics_multiclass_pt,
    get_regr_metrics_pt,
)

from transfer_learning.graphormer_3d.graphormer import Graphormer3D


class Estimator(pl.LightningModule):
    def __init__(
        self,
        task_type: str,
        batch_size: int = 32,
        lr: float = 0.001,
        scaler=None,
        monitor_loss_name: str = "val_loss",
        regression_loss_fn: str = "mae",
        early_stopping_patience: int = 30,
        optimiser_weight_decay: float = 1e-3,
        blocks: int = 4,
        layers: int = 12,
        embed_dim: int = 768,
        ffn_embed_dim: int = 768,
        attention_heads: int = 48,
        input_dropout: float = 0.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        node_loss_weight: int = 15,
        min_node_loss_weight: int = 1,
        num_kernel: int = 32,
        proj_dim: int = 768,
        **kwargs,
    ):
        super().__init__()


        self.task_type = task_type
        self.lr = lr
        self.batch_size = batch_size
        self.scaler = scaler
        self.monitor_loss_name = monitor_loss_name
        self.regression_loss_fn = regression_loss_fn
        self.early_stopping_patience = early_stopping_patience
        self.optimiser_weight_decay = optimiser_weight_decay

        Args = namedtuple('Args', ['blocks', 'layers', 'embed_dim', 'ffn_embed_dim', 'attention_heads', 'input_dropout', 'dropout', 'attention_dropout', 'activation_dropout', 'node_loss_weight', 'min_node_loss_weight', 'num_kernel', 'proj_dim'])

        args_graphormer = Args(blocks=blocks, layers=layers, embed_dim=embed_dim, ffn_embed_dim=ffn_embed_dim, attention_heads=attention_heads, input_dropout=input_dropout, dropout=dropout, attention_dropout=attention_dropout, activation_dropout=activation_dropout, node_loss_weight=node_loss_weight, min_node_loss_weight=min_node_loss_weight, num_kernel=num_kernel, proj_dim=proj_dim)

        print("Graphormer 3D arguments = ", args_graphormer)

        self.graphormer = Graphormer3D(args_graphormer)

        # Store model outputs per epoch (for train, valid) or test run; used to compute the reporting metrics
        self.train_output = defaultdict(list)
        self.val_output = defaultdict(list)
        self.val_test_output = defaultdict(list)
        self.test_output = defaultdict(list)

        self.val_preds = defaultdict(list)

        self.test_true = defaultdict(list)
        self.val_true = defaultdict(list)

        # Keep track of how many times we called test
        self.num_called_test = 1

        # Metrics per epoch (for train, valid); for test use above variable to register metrics per test-run
        self.train_metrics = {}
        self.val_metrics = {}
        self.val_test_metrics = {}
        self.test_metrics = {}

        # Holds final graphs embeddings
        self.test_graph_embeddings = defaultdict(list)
        self.val_graph_embeddings = defaultdict(list)
        self.train_graph_embeddings = defaultdict(list)

        self.linear_output_size = 1

        self.output_mlp = nn.Sequential(
            nn.Linear(proj_dim, 64), nn.Mish(), nn.Linear(64, self.linear_output_size)
        )


    def forward(self, z, pos, batch_mapping):        
        z_unbatched, batch_mask = to_dense_batch(z, batch=batch_mapping, max_num_nodes=30, fill_value=0)
        pos_unbatched, _ = to_dense_batch(pos, batch=batch_mapping, max_num_nodes=30, fill_value=0)

        out = self.graphormer(atoms=z_unbatched, pos=pos_unbatched)
        out = out[:, -1, :]

        predictions = torch.flatten(self.output_mlp(out))

        return predictions


    def configure_optimizers(self):
        opt = bnb.optim.AdamW8bit(self.parameters(), lr=self.lr, weight_decay=self.optimiser_weight_decay)

        opt_dict = {
            "optimizer": opt,
            "monitor": self.monitor_loss_name,
        }

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=self.early_stopping_patience // 2, verbose=True
        )
        if self.monitor_loss_name != "train_loss":
            opt_dict["lr_scheduler"] = sched

        return opt_dict


    def _batch_loss(self, z, pos, y, batch_mapping):
        # Forward pass (graph_embeddings not used here so far, after forward)
        predictions = self.forward(z, pos, batch_mapping)

        if self.task_type == "multi_classification":
            predictions = predictions.reshape(-1, self.linear_output_size)
            task_loss = F.cross_entropy(predictions.squeeze().float(), y.squeeze().long())

        elif self.task_type == "binary_classification":
            y = y.view(predictions.shape)
            task_loss = F.binary_cross_entropy_with_logits(predictions.float(), y.float())

        else:
            if self.regression_loss_fn == "mse":
                task_loss = F.mse_loss(torch.flatten(predictions), torch.flatten(y.float()))
            elif self.regression_loss_fn == "mae":
                task_loss = F.l1_loss(torch.flatten(predictions), torch.flatten(y.float()))

        return task_loss, predictions


    def _step(self, batch: torch.Tensor, step_type: str):
        assert step_type in ["train", "validation", "test", "validation_test"]

        z, pos, y = batch.z, batch.pos, batch.y

        total_loss, predictions = self._batch_loss(z, pos, y, batch.batch)

        if self.task_type == "regression":
            output = (torch.flatten(predictions), torch.flatten(y))
        elif "classification" in self.task_type:
            output = (predictions, y)

        if step_type == "train":
            self.train_output[self.current_epoch].append(output)
        elif step_type == "validation":
            self.val_output[self.current_epoch].append(output)
        elif step_type == "validation_test":
            self.val_test_output[self.current_epoch].append(output)
        elif step_type == "test":
            self.test_output[self.num_called_test].append(output)

        return total_loss


    def training_step(self, batch: torch.Tensor, batch_idx: int):
        train_loss = self._step(batch, "train")

        self.log("train_loss", train_loss, prog_bar=True, batch_size=self.batch_size)

        return train_loss


    def validation_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0):
        if dataloader_idx == 0:
            val_loss = self._step(batch, "validation")

            self.log("val_loss", val_loss, batch_size=self.batch_size)

            return val_loss

        if dataloader_idx == 1:
            val_test_loss = self._step(batch, "validation_test")

            self.log("val_test_loss", val_test_loss, batch_size=self.batch_size)

            return val_test_loss


    def test_step(self, batch: torch.Tensor, batch_idx: int):
        test_loss = self._step(batch, "test")

        self.log("test_loss", test_loss, batch_size=self.batch_size)

        return test_loss


    def _epoch_end_report(self, epoch_outputs, epoch_type):
        assert epoch_type in ["Train", "Validation", "Test", "ValidationTest"]

        def flatten_list_of_tensors(lst):
            return np.array([item.item() for sublist in lst for item in sublist])

        if self.task_type == "regression":
            y_pred, y_true = flatten_list_of_tensors([item[0] for item in epoch_outputs]), flatten_list_of_tensors(
                [item[1] for item in epoch_outputs]
            )
        else:
            y_pred = torch.cat([item[0] for item in epoch_outputs], dim=0)
            y_true = torch.cat([item[1] for item in epoch_outputs], dim=0)

        if self.scaler:
            if self.linear_output_size > 1:
                y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, self.linear_output_size))
                y_true = self.scaler.inverse_transform(y_true.reshape(-1, self.linear_output_size))
            else:
                y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                y_true = self.scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

            y_pred = torch.from_numpy(y_pred)
            y_true = torch.from_numpy(y_true)

        if self.task_type == "binary_classification" and self.linear_output_size > 1:
            y_true = y_true.detach().cpu().reshape(-1, self.linear_output_size).long()
            y_pred = y_pred.detach().cpu().reshape(-1, self.linear_output_size)

            metrics = get_cls_metrics_multilabel_pt(y_true, y_pred, self.linear_output_size)

            self.log(f"{epoch_type} AUROC", metrics[0], batch_size=self.batch_size)
            self.log(f"{epoch_type} MCC", metrics[1], batch_size=self.batch_size)
            self.log(f"{epoch_type} Accuracy", metrics[2], batch_size=self.batch_size)
            self.log(f"{epoch_type} F1", metrics[3], batch_size=self.batch_size)

        elif self.task_type == "binary_classification" and self.linear_output_size == 1:
            metrics = get_cls_metrics_binary_pt(y_true, y_pred)

            self.log(f"{epoch_type} AUROC", metrics[0], batch_size=self.batch_size)
            self.log(f"{epoch_type} MCC", metrics[1], batch_size=self.batch_size)
            self.log(f"{epoch_type} Accuracy", metrics[2], batch_size=self.batch_size)
            self.log(f"{epoch_type} F1", metrics[3], batch_size=self.batch_size)

        elif self.task_type == "multi_classification" and self.linear_output_size > 1:
            metrics = get_cls_metrics_multiclass_pt(y_true, y_pred, self.linear_output_size)

            self.log(f"{epoch_type} AUROC", metrics[0], batch_size=self.batch_size)
            self.log(f"{epoch_type} MCC", metrics[1], batch_size=self.batch_size)
            self.log(f"{epoch_type} Accuracy", metrics[2], batch_size=self.batch_size)
            self.log(f"{epoch_type} F1", metrics[3], batch_size=self.batch_size)

        elif self.task_type == "regression":
            metrics = get_regr_metrics_pt(y_true.squeeze(), y_pred.squeeze())

            self.log(f"{epoch_type} R2", metrics["R2"], batch_size=self.batch_size)
            self.log(f"{epoch_type} MAE", metrics["MAE"], batch_size=self.batch_size)
            self.log(f"{epoch_type} RMSE", metrics["RMSE"], batch_size=self.batch_size)
            self.log(f"{epoch_type} SMAPE", metrics["SMAPE"], batch_size=self.batch_size)

        return metrics, y_pred, y_true


    def on_train_epoch_end(self):
        self.train_metrics[self.current_epoch], y_pred, y_true = self._epoch_end_report(
            self.train_output[self.current_epoch], epoch_type="Train"
        )

        del y_pred
        del y_true
        del self.train_output[self.current_epoch]


    def on_validation_epoch_end(self):
        if len(self.val_output[self.current_epoch]) > 0:
            self.val_metrics[self.current_epoch], y_pred, y_true = self._epoch_end_report(
                self.val_output[self.current_epoch], epoch_type="Validation"
            )

            del y_pred
            del y_true
            del self.val_output[self.current_epoch]

        if len(self.val_test_output[self.current_epoch]) > 0:
            self.val_test_metrics[self.current_epoch], y_pred, y_true = self._epoch_end_report(
                self.val_test_output[self.current_epoch], epoch_type="ValidationTest"
            )

            del y_pred
            del y_true
            del self.val_test_output[self.current_epoch]


    def on_test_epoch_end(self):
        test_outputs_per_epoch = self.test_output[self.num_called_test]
        self.test_metrics[self.num_called_test], y_pred, y_true = self._epoch_end_report(
            test_outputs_per_epoch, epoch_type="Test"
        )
        self.test_output[self.num_called_test] = y_pred
        self.test_true[self.num_called_test] = y_true

        self.num_called_test += 1
