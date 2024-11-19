import os
import json


def save_gnn_arguments_to_json(argsdict, out_path):
    json_out_path = os.path.join(out_path, "gnn_hyperparameters.json")
    with open(json_out_path, "w", encoding="UTF-8") as f:
        json.dump(argsdict, f)
    return json_out_path


def load_gnn_arguments_from_json(json_path):
    with open(json_path, "r", encoding="UTF-8") as f:
        argsdict = json.load(f)

    return argsdict


def validate_gnn_argparse_arguments(argsdict):
    assert "seed" in argsdict
    assert "dataset_download_dir" in argsdict
    assert "lr" in argsdict
    assert "batch_size" in argsdict
    assert "early_stopping_patience" in argsdict
    assert "output_node_dim" in argsdict
    assert "num_layers" in argsdict
    assert "conv_type" in argsdict
    assert "gnn_intermediate_dim" in argsdict
    assert "out_path" in argsdict

    if argsdict["conv_type"] in ["GAT", "GATv2"]:
        assert argsdict["gat_attn_heads"] >= 1


def get_gnn_wandb_name(argsdict):
    if "dataset" not in argsdict:
        dataset = None
    else:
        dataset = argsdict['dataset']
    name = f"GNN+{dataset}+T={argsdict['dataset_target_name']}+S={argsdict['seed']}+{argsdict['conv_type']}"
    name += f"+GC={argsdict['gradient_clip_val']}"
    name += f"+OPTD={argsdict['optimiser_weight_decay']}"

    if "dataset_one_hot" not in argsdict:
        dataset_one_hot = None
    else:
        dataset_one_hot = argsdict['dataset_one_hot']
    name += f"+OH={dataset_one_hot}"
    name += f"+NDIM={argsdict['output_node_dim']}"
    name += f"+NL={argsdict['num_layers']}+GIDIM={argsdict['gnn_intermediate_dim']}"
    name += f"+L={argsdict['regression_loss_fn']}"
    name += f"+GATH={argsdict['gat_attn_heads']}+GATD={argsdict['gat_dropout']}"

    if "transfer_learning_hq_or_lq" in argsdict:
        hq_or_lq = argsdict["transfer_learning_hq_or_lq"]
        ind_or_trans = argsdict['transfer_learning_inductive_or_transductive']
        retrain_lq_to_hq = argsdict['transfer_learning_retrain_lq_to_hq'] == "yes"

        name += f"+TF-HQorLQ={hq_or_lq}"
        name += f"+TF-IorT={ind_or_trans}"
        name += f"+TF-tune={retrain_lq_to_hq}"

    return name
