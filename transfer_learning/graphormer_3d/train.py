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

from transfer_learning.graphormer_3d.model import Estimator
from data_loading.data_loading_transfer_learning_QM9_3D import load_qm9_gw_hq_chemprop_3D_PyG, load_qm9_dft_lq_chemprop_3D_PyG

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

    # Graphormer arguments
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--embed-dim", type=int)
    parser.add_argument("--ffn-embed-dim", type=int)
    parser.add_argument("--attention-heads", type=int)
    parser.add_argument("--input-dropout", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--attention-dropout", type=float)
    parser.add_argument("--activation-dropout", type=float)
    parser.add_argument("--node-loss-weight", type=int)
    parser.add_argument("--min-node-loss-weight", type=int)
    parser.add_argument("--num-kernel", type=int)
    parser.add_argument("--proj-dim", type=int)

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
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--train-regime", type=str, choices=["gpu-32", "gpu-bf16", "gpu-fp16", "cpu"])

    # Path/config arguments
    parser.add_argument("--ckpt-path", type=str)
    parser.add_argument("--out-path", type=str)
    parser.add_argument("--config-json-path", type=str)
    parser.add_argument("--wandb-project-name", type=str)

    args = parser.parse_args()
    argsdict = vars(args)

    seed_everything(argsdict["seed"])

    # Dataset arguments
    download_dir = argsdict["dataset_download_dir"]
    target_name = argsdict["dataset_target_name"]
    train_regime = argsdict["train_regime"]

    # Learning hyperparameters
    batch_size = argsdict["batch_size"]
    early_stopping_patience = argsdict["early_stopping_patience"]
    gradient_clip_val = argsdict["gradient_clip_val"]
    monitor_loss_name = argsdict["monitor_loss_name"]
    regr_fn = argsdict["regression_loss_fn"]

    # Graphormer arguments
    blocks = argsdict["blocks"]
    layers = argsdict["layers"]
    embed_dim = argsdict["embed_dim"]
    ffn_embed_dim = argsdict["ffn_embed_dim"]
    attention_heads = argsdict["attention_heads"]
    input_dropout = argsdict["input_dropout"]
    dropout = argsdict["dropout"]
    attention_dropout = argsdict["attention_dropout"]
    activation_dropout = argsdict["activation_dropout"]
    node_loss_weight = argsdict["node_loss_weight"]
    min_node_loss_weight = argsdict["min_node_loss_weight"]
    num_kernel = argsdict["num_kernel"]
    proj_dim = argsdict["proj_dim"]

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

    assert regr_fn is not None, "A loss functions must be specified for regression tasks!"

    # Path/config arguments
    ckpt_path = argsdict["ckpt_path"]
    out_path = argsdict["out_path"]
    wandb_project_name = argsdict["wandb_project_name"]

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

    run_name = f"Graphormer3D+T={target_name}+B={blocks}+L={layers}+EMB={embed_dim}+FFN={ffn_embed_dim}+H={attention_heads}+IDrp={input_dropout}+Drp={dropout}+AttDrp={attention_dropout}+ActDrop={activation_dropout}+K={num_kernel}+PROJ={proj_dim}"

    run_name += f"+TF-HQorLQ={hq_or_lq}"
    run_name += f"+TF-IorT={ind_or_trans}"
    run_name += f"+TF-tune={retrain_lq_to_hq}"

    output_save_dir = os.path.join(out_path, run_name)
    Path(output_save_dir).mkdir(exist_ok=True, parents=True)

    # Logging
    logger = WandbLogger(name=run_name, project=wandb_project_name)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_loss_name,
        dirpath=output_save_dir,
        filename="{epoch:03d}",
        mode="min",
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
    graphormer_args = copy.deepcopy(argsdict)
    graphormer_args = graphormer_args | dict(
        task_type=task_type,
        linear_output_size=num_classes,
        scaler=scaler,
        out_path=output_save_dir,
        use_cpu=train_regime == "cpu",
    )

    if not retrain_lq_to_hq:
        model = Estimator(**graphormer_args)
    else:
        model = Estimator.load_from_checkpoint(argsdict["ckpt_path"], **graphormer_args)

    if train_regime == "gpu-bf16":
        precision = "bf16"
    elif train_regime == "gpu-fp16":
        precision = "16"
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

    # Save test metrics
    if not is_transfer_learing_lq:
        preds_path = os.path.join(output_save_dir, "test_y_pred.npy")
        true_path = os.path.join(output_save_dir, "test_y_true.npy")
        metrics_path = os.path.join(output_save_dir, "test_metrics.npy")

        np.save(preds_path, model.test_output)
        np.save(true_path, model.test_true)
        np.save(metrics_path, model.test_metrics)

        wandb.save(preds_path)
        wandb.save(true_path)
        wandb.save(metrics_path)

    # ckpt_paths = [str(p) for p in Path(output_save_dir).rglob("*.ckpt")]
    # for cp in ckpt_paths:
    #     wandb.save(cp)

    wandb.finish()


if __name__ == "__main__":
    main()
