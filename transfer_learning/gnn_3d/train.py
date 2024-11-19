import argparse
import copy
import os
import sys
import torch
import wandb
import numpy as np
import pytorch_lightning as pl

from pathlib import Path
from torch_geometric.seed import seed_everything
from torch_geometric.loader import DataLoader as GeometricDataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

# Imports from this project
sys.path.append(os.path.realpath("."))

from transfer_learning.gnn_3d.graph_models import Estimator
from data_loading.data_loading_transfer_learning_QM9_3D import load_qm9_gw_hq_chemprop_3D_PyG, load_qm9_dft_lq_chemprop_3D_PyG
from gnn.config import (
    save_gnn_arguments_to_json,
    load_gnn_arguments_from_json,
    validate_gnn_argparse_arguments,
    get_gnn_wandb_name,
)

os.environ["WANDB__SERVICE_WAIT"] = "500"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser()

    # Seed for seed_everything
    parser.add_argument("--seed", type=int)

    # Dataset arguments
    parser.add_argument("--dataset-download-dir", type=str)
    parser.add_argument("--dataset-target-name", type=str)

    # GNN arguments
    parser.add_argument("--output-node-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--conv-type", choices=["GCN", "GIN", "PNA", "GAT", "GATv2", "GINDrop"])
    parser.add_argument("--gnn-intermediate-dim", type=int, default=256)
    parser.add_argument("--gat-attn-heads", type=int, default=0)
    parser.add_argument("--gat-dropout", type=float, default=0)

    # Transfer learning
    parser.add_argument("--transfer-learning-hq-or-lq", type=str, choices=["hq", "lq"], required=True)
    parser.add_argument("--transfer-learning-inductive-or-transductive", type=str, choices=["inductive", "transductive"])
    parser.add_argument("--transfer-learning-retrain-lq-to-hq", type=str, choices=["yes", "no"], default="no")

    # Learning hyperparameters
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--monitor-loss-name", type=str)
    parser.add_argument("--gradient-clip-val", type=float, default=0.5)
    parser.add_argument("--optimiser-weight-decay", type=float, default=1e-3)
    parser.add_argument("--regression-loss-fn", type=str, choices=["mae", "mse"])
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--train-regime", type=str, choices=["gpu-32", "gpu-bf16", "gpu-fp16", "cpu"])

    # Path/config arguments
    parser.add_argument("--ckpt-path", type=str)
    parser.add_argument("--out-path", type=str)
    parser.add_argument("--config-json-path", type=str)
    parser.add_argument("--wandb-project-name", type=str)

    args = parser.parse_args()

    if args.config_json_path:
        argsdict = load_gnn_arguments_from_json(args.config_json_path)
        validate_gnn_argparse_arguments(argsdict)
    else:
        argsdict = vars(args)
        validate_gnn_argparse_arguments(argsdict)
        del argsdict["config_json_path"]

    seed_everything(argsdict["seed"])

    # Dataset arguments
    download_dir = argsdict["dataset_download_dir"]
    target_name = argsdict["dataset_target_name"]
    train_regime = argsdict["train_regime"]

    # Transfer learning
    hq_or_lq = argsdict["transfer_learning_hq_or_lq"]
    ind_or_trans = argsdict['transfer_learning_inductive_or_transductive']
    retrain_lq_to_hq = argsdict['transfer_learning_retrain_lq_to_hq'] == "yes"

    is_transfer_learing_lq = False
    assert hq_or_lq in ["hq", "lq"]
    assert target_name in ["homo_dft", "lumo_dft", "homo_gw", "lumo_gw"]
    is_transfer_learing_lq = hq_or_lq == "lq"
    if is_transfer_learing_lq:
        assert "dft" in target_name, "LQ training must be done on one of the DFT targets!"
        assert not retrain_lq_to_hq, "Fine-tuning LQ to HQ requires a previously trained LQ model and works on HQ data!"
        assert monitor_loss_name == "train_loss", "When doing transfer learning on LQ data, only the train_loss can be used for monitoring!"
    is_transfer_learing_lq = hq_or_lq == "lq"

    if retrain_lq_to_hq:
        assert argsdict["ckpt_path"] is not None, "Must specify a trained LQ model checkpoint path!"
        assert "gw" in target_name, "Fine-tuning must be done on one of the GW targets!"

    # Learning hyperparameters
    batch_size = argsdict["batch_size"]
    early_stopping_patience = argsdict["early_stopping_patience"]
    gradient_clip_val = argsdict["gradient_clip_val"]
    monitor_loss_name = argsdict["monitor_loss_name"]

    # Path/config arguments
    ckpt_path = argsdict["ckpt_path"]
    out_path = argsdict["out_path"]
    wandb_project_name = argsdict["wandb_project_name"]

    if monitor_loss_name == 'MCC' or 'MCC' in monitor_loss_name:
        monitor_loss_name = 'Validation MCC'

    if is_transfer_learing_lq:
        assert monitor_loss_name == "train_loss", \
            "When training on the expressivity datasets or doing transfer learning on LQ data, only the train_loss can be used for monitoring"


    ############## Data loading ##############
    if hq_or_lq == "hq":
        train, val, test, num_classes, task_type, scaler = load_qm9_gw_hq_chemprop_3D_PyG(
            dataset_dir=download_dir,
            target_name=target_name,
        )
    elif hq_or_lq == "lq":
        train, _, _, num_classes, task_type, scaler = load_qm9_dft_lq_chemprop_3D_PyG(
            dataset_dir=download_dir,
            target_name=target_name,
            ind_or_trans=ind_or_trans,
        )        


    train_loader = GeometricDataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    if not is_transfer_learing_lq:
        val_loader = GeometricDataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = GeometricDataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    ############## Data loading ##############

    run_name = get_gnn_wandb_name(argsdict)

    output_save_dir = os.path.join(out_path, run_name)
    Path(output_save_dir).mkdir(exist_ok=True, parents=True)

    config_json_path = save_gnn_arguments_to_json(argsdict, output_save_dir)

    # Logging
    logger = WandbLogger(name=run_name, project=wandb_project_name)

    # Callbacks
    monitor_mode = "max" if "MCC" in monitor_loss_name else "min"
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_loss_name,
        dirpath=output_save_dir,
        filename="{epoch:03d}",
        mode=monitor_mode,
        save_top_k=1 if not is_transfer_learing_lq else -1,
    )

    early_stopping_callback = EarlyStopping(
        monitor=monitor_loss_name, patience=early_stopping_patience, mode="min"
    )

    if not is_transfer_learing_lq:
        callbacks = [checkpoint_callback, early_stopping_callback]
    else:
        callbacks = [checkpoint_callback]

    ############## Learning and model set-up ##############
    gnn_args = copy.deepcopy(argsdict)
    gnn_args = gnn_args | dict(
        task_type=task_type,
        num_features=None,
        linear_output_size=num_classes,
        scaler=scaler,
        edge_dim=None,
        out_path=output_save_dir,
        use_cpu=train_regime == "cpu",
    )

    if argsdict["conv_type"] == "PNA":
        gnn_args = gnn_args | dict(train_dataset_for_PNA=train)

    if not retrain_lq_to_hq:
        model = Estimator(**gnn_args)
    else:
        model = Estimator.load_from_checkpoint(argsdict["ckpt_path"], **gnn_args)

    if train_regime == "gpu-bf16":
        precision = "bf16-mixed"
    elif train_regime == "gpu-fp16":
        precision = "16-mixed"
    else:
        precision = "32"

    trainer_args = dict(
        callbacks=callbacks,
        logger=logger,
        min_epochs=1 if not is_transfer_learing_lq else 150,
        max_epochs=-1 if not is_transfer_learing_lq else 151,
        devices=1,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        precision=precision,
        gradient_clip_val=gradient_clip_val,
    )

    if "gpu" in train_regime:
        trainer_args = trainer_args | dict(accelerator="gpu")
    else:
        trainer_args = trainer_args | dict(accelerator="cpu")
    ############## Learning and model set-up ##############

    trainer = pl.Trainer(**trainer_args)

    if not is_transfer_learing_lq:
        if not retrain_lq_to_hq:
            # HQ only
            trainer.fit(
                model=model, train_dataloaders=train_loader, val_dataloaders=[val_loader, test_loader], ckpt_path=ckpt_path
            )
        else:
            # Fine-tune LQ to HQ
            trainer.fit(
                model=model, train_dataloaders=train_loader, val_dataloaders=[val_loader, test_loader],
            )
        trainer.test(model=model, dataloaders=test_loader, ckpt_path="best")
    else:
        # LQ only
        trainer.fit(
            model=model, train_dataloaders=train_loader, ckpt_path=ckpt_path
        )

    if not is_transfer_learing_lq:
        # Save test metrics
        preds_path = os.path.join(output_save_dir, "test_y_pred.npy")
        true_path = os.path.join(output_save_dir, "test_y_true.npy")
        metrics_path = os.path.join(output_save_dir, "test_metrics.npy")

        np.save(preds_path, model.test_output)
        np.save(true_path, model.test_true)
        np.save(metrics_path, model.test_metrics)

        wandb.save(preds_path)
        wandb.save(true_path)
        wandb.save(metrics_path)
        wandb.save(config_json_path)

        # ckpt_paths = [str(p) for p in Path(output_save_dir).rglob("*.ckpt")]
        # for cp in ckpt_paths:
        #     wandb.save(cp)

    wandb.finish()


if __name__ == "__main__":
    main()
