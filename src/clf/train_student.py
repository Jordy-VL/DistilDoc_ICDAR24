import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import argparse
import numpy as np
import torch
import json
import wandb
import string
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Distildoc ROOT
from default_config import SAVEROOT
from data.datasets import build_dataset
from models.models import build_model
from train_teacher import seed_everything

from distillation_modules import KnowledgeDistillation


def STUDENT_CONFIG():
    return {
        "batch_size": 32,
        "downsampling": 0,
        "epochs": 100,
        "lr": 1e-4,
        "optimizer": "AdamW",
        "seed": 42,
        "warmup_ratio": 0.1,
        "weight_decay": 0.0005,
        "gradient_accumulation_steps": 1,
        "temperature": 1,  # no smoothing of teacher logits
        "alpha": 0.5,  # student loss and kd equally weighted
        "gamma": 1,
        "beta": 0.5,
    }


def main(args):
    args.expt_name = expt_name(args)
    config = {**STUDENT_CONFIG(), **{k: v for k, v in args.__dict__.items() if v is not None}}
    seed_everything(config["seed"])
    if not config["skip_sync"] or config["precompute"]:
        wandb.init(project="DistilDoc", name=config["expt_name"], tags=["student"], config=config)

    # load teacher
    teacher, config = build_model(config, key="teacher_model")

    # load student
    key = "student_model" if config["student_weights"] == "" else "student_weights"
    student, config = build_model(config, key=key)

    # TODO: random initialized Resnet50 student

    # TODO: reload trained student for evaluation without overriding
    if config["eval_only"]:
        if "og_simkd" in config["student_weights"]:
            config["distill"] = "og_simkd"
        elif "simkd" in config["student_weights"]:
            config["distill"] = "simkd"
        elif "hint" in config["student_weights"]:
            config["distill"] = "hint"

        # reload projector

        # if any of those then need to pass all parameters and reload projector

    # load data (already processed)
    if config["eval_only"]:
        config["label2id"] = student.config.label2id
        train_dataset, eval_dataset, test_dataset = (
            None,
            None,
            build_dataset(config, "test", processor=student.processor, add_indices=config["precompute"]),
        )
    else:
        train_dataset = build_dataset(config, "train", processor=student.processor, add_indices=config["precompute"])
        eval_dataset = build_dataset(
            config, "validation", processor=student.processor, add_indices=config["precompute"]
        )
        test_dataset = build_dataset(config, "test", processor=student.processor, add_indices=config["precompute"])

    trainer = KnowledgeDistillation(
        config,
        teacher,
        student,
        train_dataset,
        eval_dataset,
        test_dataset=test_dataset,
        hyperparameters=None,  # already in config? - could use to filter hp_space
        hp_tuning=config["hp"],
    )

    print(json.dumps(config, indent=4))

    if config["precompute"]:
        trainer.precompute_teacher()
        sys.exit(1)

    if config["eval_only"]:
        student_trainer = trainer.setup()
        # This is a nasty monkey patch to reload projector weights which are skipped by HF; could be generalized based on (approximate) key matching
        if hasattr(student_trainer.model, "projector") and student_trainer.model.projector is not None:
            binary = glob.glob(
                os.path.join(
                    os.environ["HF_HOME"],
                    "models--" + (config["student_weights"].replace("/", "--")),
                    "snapshots",
                    "*",
                    "pytorch_model.bin",
                )
            )[-1]
            loaded = torch.load(binary)
            updates = {k.replace("projector.", "", 1): v for k, v in loaded.items() if "projector" in k}
            student_trainer.model.projector.load_state_dict(updates)
            print("Successfully reloaded projector weights")
        # for all shared projector keys, load weights into student_trainer.projector
        # reload weights from projector state dict...
        # student_trainer.load_state_dict(torch.load(os.path.join(SAVEROOT, config["expt_name"], "pytorch_model.bin")))

    else:
        try:
            student_trainer = trainer.train_student_model()
        except KeyboardInterrupt as e:
            print(e)

    if config["eval_only"]:
        student_trainer.evaluate(
            eval_dataset=test_dataset, metric_key_prefix="prefix"
        )  # bug yet wont fix as it is reported like this now
        return

    split = "test"
    output_path = os.path.join(SAVEROOT, config["expt_name"])  # /tmp?
    test_results = student_trainer.predict(test_dataset, metric_key_prefix=split)
    student_trainer.log(test_results.metrics)
    if not config["skip_logits"]:
        np.savez_compressed(os.path.join(output_path, f"{split}-references.npz"), test_results.label_ids)
        np.savez_compressed(os.path.join(output_path, f"{split}-logits.npz"), test_results.predictions)

        split = "validation"
        val_results = student_trainer.predict(eval_dataset, metric_key_prefix=split)
        student_trainer.log(test_results.metrics)
        np.savez_compressed(os.path.join(output_path, f"{split}-references.npz"), val_results.label_ids)
        np.savez_compressed(os.path.join(output_path, f"{split}-logits.npz"), val_results.predictions)

    # run teacher fully supervised evaluation
    ## TODO: do the same to RVL-CDIP-N
    if config["dataset"] == "rvl_cdip" and "100" in config["test_dataset"]:
        config["dataset"] = config["test_dataset"]
        test_dataset = build_dataset(config, "test", processor=student.processor)
        student_trainer.evaluate(
            eval_dataset=test_dataset, metric_key_prefix="test_100"
        )  # test on 25 examples per class

    student_trainer.push_to_hub("Saving best model to hub")


def expt_name(hp):
    joiner = [
        hp.teacher_model.split("/")[-1],
        hp.student_model.split("-")[-1],
        hp.dataset.split("/")[-1],
        hp.distill,
        hp.distill_loss.replace("+", ""),
    ]
    base = "{}-{}_{}_{}_{}".format(*tuple(joiner))

    if hp.distill_loss == "CE" or "simkd" in hp.distill or "hint" in hp.distill:
        base = "".join(
            [
                hp.teacher_model.split("/")[-1],
                "-",
                hp.student_model.split("-")[-1],
                "_",
                hp.dataset.split("/")[-1],
                "_",
                hp.distill,
            ]
        )

    if hp.distill_loss == "CE+KD":
        base += "_t{}_a{}".format(hp.temperature, hp.alpha)

    if hp.distill_loss == "NKD":
        base += "_t{}_g{}".format(hp.temperature, hp.gamma)

    if hp.beta is not None:
        base += "_b{}".format(hp.beta)

    if hp.expt_name != "student":
        base += "_{}".format(hp.expt_name)
    return base


def hint_combinations(argument):
    _, layer_strategy, cls_rep, projector = argument.split("-")
    PROJECTORS = ["CNN2D", "CNN1D", "CLS_MLP"]
    CLS_REP = ["", "add"]
    LAYER_STRATEGIES = ["all", "all_noembed", "4interval", "last"] + string.digits

    assert layer_strategy in LAYER_STRATEGIES
    assert cls_rep in CLS_REP
    assert projector in PROJECTORS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DistilDoc_logits_ViT")

    parser.add_argument("--expt_name", type=str, default="student")

    # data arguments
    parser.add_argument("--dataset", type=str, default="jordyvl/rvl_cdip_100_examples_per_class")
    parser.add_argument("--test_dataset", type=str, default="jordyvl/rvl_cdip_100_examples_per_class")

    # teacher arguments
    parser.add_argument("--teacher_model", type=str)

    parser.add_argument("--student_weights", type=str, default="")

    parser.add_argument(
        "--student_model",
        type=str,
        default="",
        choices=["dit-tiny", "vit-tiny", "swin-tiny", "dit-small", "vit-small", "swin-small", "cnn", ""],
    )

    # distillation
    parser.add_argument(
        "--distill",
        type=str,
        default="",
        choices=[
            "",
            "og_simkd",
            "simkd",
            "dualsimkd",
            "kd",
            "hint",
            # "attention",
            # "similarity",
            # "correlation",
            # "vid",
            # "crd",
            # "kdsvd",
            # "fsp",
            # "rkd",
            # "pkt",
            # "abound",
            # "factor",
            # "nst",
        ],
    )

    # Hyper-parameters
    parser.add_argument(
        "--distill_loss",
        default="",
        const="CE+KD",
        nargs="?",
        choices=("", "CE", "CE+KD", "MSE", "NKD"),  # just CE means no distillation
        help="Choice of Distillation Loss to optimize the student network (default: %(default)s)",
    )
    parser.add_argument("-t", "--temperature", type=float, help="temperature for KD")
    parser.add_argument("-a", "--alpha", type=float, help="weight balance for CE+KD")
    parser.add_argument("-g", "--gamma", type=float, help="weight for classification")
    parser.add_argument("-b", "--beta", type=float, help="weight balance for other losses")

    # optimization parameters
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument(
        "--precompute",
        action="store_true",
        default=False,
        help="precompute teacher logits (and hidden states) and dump to disk (or send to HF?)",
    )

    parser.add_argument("--skip_logits", action="store_true", default=False, help="Dump logits to disk/wandb")

    parser.add_argument("--skip_sync", action="store_true", default=False, help="disable wandb")

    parser.add_argument("--hp", action="store_true", default=False, help="activate hyperparameter tuning")
    parser.add_argument("--n_trials", type=int, default=5)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_only", action="store_true", default=False, help="only evaluate")

    hp = parser.parse_args()
    main(hp)
