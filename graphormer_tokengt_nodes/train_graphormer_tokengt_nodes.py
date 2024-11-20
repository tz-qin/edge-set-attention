import os
import argparse
import torch
import numpy as np
import bitsandbytes as bnb

from pathlib import Path
from torch_geometric.seed import seed_everything
from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from graphormer_tokengt_nodes.graphormer.modelling_graphormer import GraphormerForNodeClassification
from graphormer_tokengt.graphormer.configuration_graphormer import GraphormerConfig
from graphormer_tokengt.graphormer.collating_graphormer_safetensors import GraphormerDataCollator

from graphormer_tokengt_nodes.tokengt.modeling_tokengt import TokenGTForNodeClassification
from graphormer_tokengt.tokengt.configuration_tokengt import TokenGTConfig
from graphormer_tokengt.tokengt.collating_tokengt import TokenGTDataCollator

from data_loading.data_loading_graphormer import get_dataset_train_val_test_graphormer
from data_loading.data_loading_tokengt import get_dataset_train_val_test_tokengt

from utils.reporting import (
    get_regr_metrics_pt,
    get_cls_metrics_binary_pt,
    get_cls_metrics_multilabel_pt,
    get_cls_metrics_multiclass_pt,
)

os.environ["WANDB__SERVICE_WAIT"] = "500"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def check_is_node_level_dataset(dataset_name):
    if dataset_name in ["PPI", "Cora", "CiteSeer"]:
        return True
    elif "infected" in dataset_name:
        return True
    elif "hetero" in dataset_name:
        return True
    
    return False


class RegressionTrainer(Trainer):
    def __init__(self, *args, regression_loss=None, **kwargs):
        # Call the parent __init__ to initialize the standard Trainer
        super().__init__(*args, **kwargs)

        self.regression_loss = regression_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")

        if self.regression_loss == "mae":
            loss_fct = torch.nn.L1Loss()
        elif self.regression_loss == "mse":
            loss_fct = torch.nn.MSELoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dataset_download_dir", type=str)
    parser.add_argument("--dataset_one_hot", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dataset_target_name", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--wandb_project_name", type=str)
    parser.add_argument("--architecture", type=str, choices=["graphormer", "tokengt"])
    parser.add_argument("--early_stop_patience", type=int, default=30)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--regression_loss_fn", type=str, choices=["mae", "mse"])
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--optimiser_weight_decay", type=float, default=1e-3)
    parser.add_argument("--bfloat16", type=str, choices=["yes", "no"], default="no")
    parser.add_argument("--ckpt_path", type=str)
    args = parser.parse_args()

    seed_everything(args.seed)

    os.environ["HF_DATASETS_CACHE"] = args.dataset_download_dir
    os.environ["WANDB_PROJECT"] = args.wandb_project_name
    os.environ["WANDB_ENTITY"] = "<WANDB_USERNAME>"

    assert check_is_node_level_dataset(args.dataset_name), "Only node-level tasks are supported by this script!"
    assert args.dataset_one_hot == False, "huggingface requires integer features, not one-hot encodings!"

    use_bfloat16 = args.bfloat16 == "yes"
    ckpt_path = args.ckpt_path

    if args.architecture == "graphormer":
        get_dataset_splits = get_dataset_train_val_test_graphormer
    elif args.architecture == "tokengt":
        get_dataset_splits = get_dataset_train_val_test_tokengt

    train, val, test, num_classes, task_type, scaler, train_mask, val_mask, test_mask = get_dataset_splits(
        args.dataset_name,
        dataset_dir=args.dataset_download_dir,
        one_hot=args.dataset_one_hot,
        model=args.architecture,
        dataset_download_dir=args.dataset_download_dir,
        target_name=args.dataset_target_name,
    )

    if task_type == "regression":
        problem_type = "regression"

    if task_type == "multi_classification":
        problem_type = "multi_label_classification"
    
    if task_type == "binary_classification":
        problem_type = "single_label_classification"

    if args.architecture == "graphormer":
        graphormer_config = {
            "activation_dropout": 0.1,
            "activation_fn": "gelu",
            "apply_graphormer_init": True,
            "architectures": ["GraphormerForGraphClassification"],
            "attention_dropout": 0.1,
            "bias": True,
            "bos_token_id": 1,
            "dropout": 0.0,
            "edge_type": "multi_hop",
            "embed_scale": None,
            "embedding_dim": args.embedding_dim,
            "encoder_normalize_before": True,
            "eos_token_id": 2,
            "export": False,
            "ffn_embedding_dim": args.embedding_dim,
            "freeze_embeddings": False,
            "hidden_size": args.hidden_size,
            "init_fn": None,
            "kdim": None,
            "layerdrop": 0.0,
            "max_nodes": 20000,
            "model_type": "graphormer",
            "multi_hop_max_dist": 5,
            "no_token_positional_embeddings": False,
            "num_atoms": 20000,
            "num_attention_heads": args.num_attention_heads,
            "num_edge_dis": 128,
            "num_edges": 20000,
            "num_in_degree": 5000,
            "num_layers": args.num_layers,
            "num_hidden_layers": args.num_layers,
            "num_out_degree": 5000,
            "num_spatial": 512,
            "num_trans_layers_to_freeze": 0,
            "pad_token_id": 0,
            "pre_layernorm": False,
            "q_noise": 0.0,
            "qn_block_size": 8,
            "self_attention": True,
            "share_input_output_embed": False,
            "spatial_pos_max": 1024,
            "torch_dtype": "float32",
            "traceable": False,
            "vdim": None,
            "ignore_mismatched_sizes": True,
            "output_hidden_states": False,
            "output_attentions": False,
            "num_labels": num_classes,
            "num_classes": num_classes,
            "problem_type": problem_type,
            "regression_loss_fn": args.regression_loss_fn,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
        }

        config = GraphormerConfig(**graphormer_config)

        model = GraphormerForNodeClassification(config=config)
    else:
        tokentg_config = {
            "activation_dropout": 0.1,
            "activation_fn": "gelu",
            "apply_graphormer_init": True,
            "architectures": ["TokenGTForGraphClassification"],
            "attention_dropout": 0.1,
            "bias": True,
            "bos_token_id": 1,
            "dropout": 0.0,
            "edge_type": "multi_hop",
            "embed_scale": None,
            "embedding_dim": args.embedding_dim,
            "encoder_normalize_before": True,
            "prenorm": True,
            "eos_token_id": 2,
            "ffn_embedding_dim": args.embedding_dim,
            "freeze_embeddings": False,
            "init_fn": None,
            "kdim": None,
            "lap_node_id": True,
            "lap_node_id_eig_dropout": 0.0,
            "lap_node_id_k": 16,
            "lap_node_id_sign_flip": True,
            "layerdrop": 0.0,
            "layernorm_style": "prenorm",
            "max_nodes": 20000,
            "model_type": "tokengt",
            "multi_hop_max_dist": 5,
            "n_trans_layers_to_freeze": 0,
            "no_token_positional_embeddings": False,
            "num_atoms": 20000,
            "num_attention_heads": args.num_attention_heads,
            "num_edge_dis": 128,
            "num_edges": 20000,
            "num_in_degree": 512,
            "num_layers": args.num_layers,
            "num_out_degree": 512,
            "num_spatial": 512,
            "orf_node_id": False,
            "orf_node_id_dim": 64,
            "pad_token_id": 0,
            "performer": False,
            "performer_auto_check_redraw": True,
            "performer_feature_redraw_interval": 1000,
            "performer_finetune": False,
            "performer_generalized_attention": False,
            "performer_nb_features": None,
            "q_noise": 0.0,
            "qn_block_size": 8,
            "rand_node_id": False,
            "rand_node_id_dim": 64,
            "return_attention": False,
            "self_attention": True,
            "share_encoder_input_output_embed": False,
            "share_input_output_embed": False,
            "spatial_pos_max": 1024,
            "stochastic_depth": False,
            "tasks_weights": None,
            "torch_dtype": "float32",
            "traceable": False,
            "type_id": True,
            "uses_fixed_gaussian_features": False,
            "vdim": None,
            "output_hidden_states": False,
            "output_attentions": False,
            "num_labels": num_classes,
            "num_classes": num_classes,
            "problem_type": problem_type,
            "regression_loss_fn": args.regression_loss_fn,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
        }

        config = TokenGTConfig(**tokentg_config)

        model = TokenGTForNodeClassification(config=config)

    logs_dir = os.path.join(args.out_dir, "logs")
    Path(logs_dir).mkdir(exist_ok=True, parents=True)

    if "struct" in args.dataset_name:
        metric_for_best_model = "Average_MAE"
        greater_is_better = False
    elif "fn" in args.dataset_name:
        metric_for_best_model = "AP"
        greater_is_better = True
    else:
        metric_for_best_model = "loss"
        greater_is_better = False

    training_args = TrainingArguments(
        logging_dir=logs_dir,
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        auto_find_batch_size=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=False,
        num_train_epochs=10000,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        push_to_hub=False,
        use_cpu=False,
        bf16=use_bfloat16,
        tf32=use_bfloat16,
        do_train=True,
        do_eval=True,
        report_to=["wandb"],
        run_name=args.name,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        load_best_model_at_end=True,
        max_grad_norm=args.gradient_clip_val,
        gradient_checkpointing=False,
        eval_accumulation_steps=50,
        save_total_limit=1,
    )
    training_args._n_gpu = 1

    os.environ["WANDB_NOTEBOOK_NAME"] = args.name

    collator = GraphormerDataCollator() if args.architecture == "graphormer" else TokenGTDataCollator()

    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=args.optimiser_weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.early_stop_patience // 2, verbose=True, min_lr=0.00005
    )
    
    trainer_args = dict(
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience)],
        optimizers=(optimizer, lr_scheduler)
    )

    def compute_metrics_for_regression(eval_pred):
        try:
            logits, _ = eval_pred.predictions
        except Exception:
            logits = eval_pred.predictions
        labels = eval_pred.label_ids

        if scaler is not None:
            logits = scaler.inverse_transform(logits)
            labels = scaler.inverse_transform(labels)

        if "lrgb" in args.dataset_name:
            mae_avg = []
            for idx in range(11):
                metrics = get_regr_metrics_pt(torch.from_numpy(labels).squeeze()[:, idx], torch.from_numpy(logits).squeeze()[:, idx])
                mae_avg.append(metrics["MAE"])
            mae_avg = np.mean(np.array([x.item() for x in mae_avg]))
            return {"Average_MAE": mae_avg}

        else:
            metrics = get_regr_metrics_pt(torch.from_numpy(labels), torch.from_numpy(logits))
            return {k: v.item() for k, v in metrics.items()}

    def compute_metrics_for_binary_classification(eval_pred):
        try:
            logits, _ = eval_pred.predictions
        except Exception:
            logits = eval_pred.predictions
        labels = eval_pred.label_ids

        auroc, mcc, acc, f1 = get_cls_metrics_binary_pt(
            torch.from_numpy(labels),
            torch.from_numpy(logits),
        )

        return {"AUROC": auroc.item(), "MCC": mcc.item(), "Accuracy": acc.item(), "F1": f1.item()}

    def compute_metrics_for_multiclass_classification(eval_pred):
        try:
            logits, _ = eval_pred.predictions
        except Exception:
            logits = eval_pred.predictions
        labels = eval_pred.label_ids

        auroc, mcc, acc, f1, ap = get_cls_metrics_multiclass_pt(
            torch.from_numpy(labels).float(), torch.from_numpy(logits).float(), num_classes
        )

        return {"AUROC": auroc.item(), "MCC": mcc.item(), "Accuracy": acc.item(), "F1": f1.item(), "AP": ap.item()}

    def compute_metrics_for_binary_multilabel_classification(eval_pred):
        try:
            logits, _ = eval_pred.predictions
        except Exception:
            logits = eval_pred.predictions
        labels = eval_pred.label_ids

        auroc, mcc, acc, f1 = get_cls_metrics_multilabel_pt(
            torch.from_numpy(logits).float(), torch.from_numpy(labels).float(), num_classes
        )

        return {"AUROC": auroc.item(), "MCC": mcc.item(), "Accuracy": acc.item(), "F1": f1.item()}

    if args.regression_loss_fn:
        compute_metrics_fn = compute_metrics_for_regression
    else:
        if task_type == "binary_classification" and num_classes == 1:
            compute_metrics_fn = compute_metrics_for_binary_classification
        elif task_type == "binary_classification" and num_classes > 1:
            compute_metrics_fn = compute_metrics_for_binary_multilabel_classification
        elif task_type == "multi_classification" and num_classes > 1:
            compute_metrics_fn = compute_metrics_for_multiclass_classification

    trainer_args = trainer_args | dict(compute_metrics=compute_metrics_fn)
    if problem_type == "regression":
        trainer = RegressionTrainer(model=model, regression_loss=args.regression_loss_fn, **trainer_args)
    else:
        trainer = Trainer(model=model, **trainer_args)

    trainer.train(resume_from_checkpoint=ckpt_path)
    results = trainer.predict(test_dataset=test)

    print("Computing test metrics...")
    test_metrics = compute_metrics_fn(results)
    trainer.save_metrics(split="test", metrics=test_metrics)

    preds_path = os.path.join(args.out_dir, "test_preds.npy")
    true_path = os.path.join(args.out_dir, "test_true.npy")
    metrics_path = os.path.join(args.out_dir, "test_metrics.npy")

    np.save(preds_path, results.predictions[0])
    np.save(true_path, results.label_ids)
    np.save(metrics_path, test_metrics)


if __name__ == "__main__":
    main()
