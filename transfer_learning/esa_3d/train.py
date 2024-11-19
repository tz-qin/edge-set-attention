import sys
import os
import warnings
import argparse
import copy
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

from transfer_learning.esa_3d.models import Estimator

from data_loading.data_loading_transfer_learning_QM9_3D import load_qm9_gw_hq_chemprop_3D_PyG, load_qm9_dft_lq_chemprop_3D_PyG
from esa.config import (
    save_arguments_to_json,
    load_arguments_from_json,
    validate_argparse_arguments,
    get_wandb_name,
)

warnings.filterwarnings("ignore")

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

    # Learning hyperparameters
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--norm-type", type=str, choices=["BN", "LN"])
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--monitor-loss-name", type=str)
    parser.add_argument("--gradient-clip-val", type=float, default=0.5)
    parser.add_argument("--optimiser-weight-decay", type=float, default=1e-3)
    parser.add_argument("--regression-loss-fn", type=str, choices=["mae", "mse"])
    parser.add_argument("--use-bfloat16", default=True, action=argparse.BooleanOptionalAction)

    # Node/graph dimensions
    parser.add_argument("--graph-dim", type=int)

    # ESA arguments
    parser.add_argument("--xformers-or-torch-attn", type=str, choices=["xformers", "torch"])
    parser.add_argument("--apply-attention-on", type=str, choices=["node", "edge"], default="edge")
    parser.add_argument("--layer-types", type=str, nargs="+")
    parser.add_argument("--hidden-dims", type=int, nargs="+")
    parser.add_argument("--num-heads", type=int, nargs="+")
    parser.add_argument("--pre-or-post", type=str, choices=["pre", "post"], default="post")
    parser.add_argument("--sab-dropout", type=float, default=0.0)
    parser.add_argument("--mab-dropout", type=float, default=0.0)
    parser.add_argument("--pma-dropout", type=float, default=0.0)
    parser.add_argument("--attn-residual-dropout", type=float, default=0.0)
    parser.add_argument("--pma-residual-dropout", type=float, default=0.0)
    
    # MLP arguments
    parser.add_argument("--use-mlps", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--mlp-hidden-size", type=int, default=64)
    parser.add_argument("--mlp-layers", type=int, default=2)
    parser.add_argument("--mlp-type", type=str, choices=["standard", "gated_mlp"], default="gated_mlp")
    parser.add_argument("--mlp-dropout", type=float, default=0.0)
    parser.add_argument("--use-mlp-ln", type=str, choices=["yes", "no"], default="yes")

    # Transfer learning
    parser.add_argument("--transfer-learning-hq-or-lq", type=str, choices=["hq", "lq"], required=True)
    parser.add_argument("--transfer-learning-inductive-or-transductive", type=str, choices=["inductive", "transductive"])
    parser.add_argument("--transfer-learning-retrain-lq-to-hq", type=str, choices=["yes", "no"], default="no")

    # 3D OCP settings
    parser.add_argument("--ocp-num-kernels", type=int)
    parser.add_argument("--ocp-embed-dim", type=int)
    parser.add_argument("--ocp-cutoff-dist", type=float)
    parser.add_argument("--ocp-num-neigh", type=int)

    # Path/config arguments
    parser.add_argument("--ckpt-path", type=str)
    parser.add_argument("--out-path", type=str)
    parser.add_argument("--config-json-path", type=str)
    parser.add_argument("--wandb-project-name", type=str)
    
    args = parser.parse_args()

    if args.config_json_path:
        argsdict = load_arguments_from_json(args.config_json_path)
        validate_argparse_arguments(argsdict)
    else:
        argsdict = vars(args)
        validate_argparse_arguments(argsdict)
        del argsdict["config_json_path"]

    seed_everything(argsdict["seed"])

    # Dataset arguments
    target_name = argsdict["dataset_target_name"]
    download_dir = argsdict["dataset_download_dir"]

    # Learning hyperparameters
    batch_size = argsdict["batch_size"]
    early_stopping_patience = argsdict["early_stopping_patience"]
    gradient_clip_val = argsdict["gradient_clip_val"]
    use_bfloat16 = argsdict["use_bfloat16"]
    apply_attention_on = argsdict["apply_attention_on"]
    mlp_dropout = argsdict["mlp_dropout"]

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

    # # 3D OCP settings
    num_kernels = argsdict["ocp_num_kernels"]
    embed_dim = argsdict["ocp_embed_dim"]
    cutoff_dist = argsdict["ocp_cutoff_dist"]
    num_nn = argsdict["ocp_num_neigh"]

    # Path/config arguments
    ckpt_path = argsdict["ckpt_path"]
    out_path = argsdict["out_path"]
    wandb_project_name = argsdict["wandb_project_name"]
    monitor_loss_name = argsdict["monitor_loss_name"]
    num_mlp_layers = argsdict["mlp_layers"]
    pre_or_post = argsdict["pre_or_post"]
    pma_residual_dropout = argsdict["pma_residual_dropout"]
    use_mlp_ln = argsdict["use_mlp_ln"] == "yes"

    if monitor_loss_name == 'MCC' or 'MCC' in monitor_loss_name:
        monitor_loss_name = 'Validation MCC'
        
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

    run_name = get_wandb_name(argsdict)

    output_save_dir = os.path.join(out_path)
    Path(output_save_dir).mkdir(exist_ok=True, parents=True)

    config_json_path = save_arguments_to_json(argsdict, output_save_dir)

    # Logging
    logger = WandbLogger(name=run_name, project=wandb_project_name, save_dir=output_save_dir)

    monitor_mode = "max" if "MCC" in monitor_loss_name else "min"

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_loss_name,
        dirpath=output_save_dir,
        filename="{epoch:03d}",
        mode=monitor_mode,
        save_top_k=1 if not is_transfer_learing_lq else -1,
    )

    early_stopping_callback = EarlyStopping(
        monitor=monitor_loss_name, patience=early_stopping_patience, mode=monitor_mode
    )

    if not is_transfer_learing_lq:
        callbacks = [checkpoint_callback, early_stopping_callback]
    else:
        callbacks = [checkpoint_callback]

    ############## Learning and model set-up ##############
    model_args = copy.deepcopy(argsdict)
    set_max_items = None

    set_max_items = train[0].max_edge_global if apply_attention_on == "edge" else train[0].max_node_global

    model_args = model_args | dict(
        task_type=task_type, linear_output_size=num_classes, scaler=scaler,
        set_max_items=set_max_items, pma_residual_dropout=pma_residual_dropout,
        k=num_kernels, embed_dim=embed_dim, cutoff_dist=cutoff_dist, num_nn=num_nn,
        train_mask=None, val_mask=None, test_mask=None, num_features=None, edge_dim=None,
        num_mlp_layers=num_mlp_layers, pre_or_post=pre_or_post,
        use_mlp_ln=use_mlp_ln, mlp_dropout=mlp_dropout,
    )

    model_args |= dict(is_node_task=False)

    if not retrain_lq_to_hq:
        model = Estimator(**model_args)
    else:
        model = Estimator.load_from_checkpoint(argsdict["ckpt_path"], **model_args)
    model = model.cuda()

    trainer_args = dict(
        callbacks=callbacks,
        logger=logger,
        min_epochs=1 if not is_transfer_learing_lq else 150,
        max_epochs=-1 if not is_transfer_learing_lq else 151,
        devices=1,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        precision="bf16" if use_bfloat16 else "32",
        gradient_clip_val=gradient_clip_val,
    )

    trainer_args = trainer_args | dict(accelerator="gpu")

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
