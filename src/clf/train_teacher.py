import os
import sys
import argparse
import random
import numpy as np
import torch
import wandb

from transformers import Trainer, TrainingArguments, DefaultDataCollator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Distildoc ROOT
from default_config import MODELROOT
from data.datasets import build_dataset
from models.models import build_model
from metrics import compute_metrics


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def TEACHER_CONFIG():
    return {
        "batch_size": 4,
        "downsampling": 0,
        "epochs": 60,
        "lr": 2e-5,
        "optimizer": "AdamW",
        "seed": 42,
        "warmup_ratio": 0.1,
        "weight_decay": 0,
        "gradient_accumulation_steps": 16,
    }


def main(args):
    config = {**TEACHER_CONFIG(), **{k: v for k, v in args.__dict__.items() if v is not None}}
    seed_everything(config["seed"])
    wandb.init(project="DistilDoc", name=config["expt_name"], tags=["teacher"], config=config)

    # # load teacher
    # if config['sup_teacher_weights'] and config['eval_only']:
    #     from transformers import AutoModelForImageClassification, AutoProcessor
    #     model = AutoModelForImageClassification.from_pretrained(config['sup_teacher_weights'])
    #     model.processor = AutoProcessor.from_pretrained(config['sup_teacher_weights'])   

    if config["student_mode"]:  # not needed, just train_student.py with CE
        config["student_model"] = config["teacher_model"] + "-small"
        teacher, config = build_model(config, key="student_model")
        config["teacher_model"] = config["student_model"]
        config["expt_name"] = config["expt_name"] + "_small_student"
        config["lr"] = 1e-4
        config["batch_size"] = 16
        config["gradient_accumulation_steps"] = 1
    elif config["sup_teacher_weights"]:
        teacher, config = build_model(config, key="sup_teacher_weights")
    else:
        teacher, config = build_model(config, key="teacher_model", override=True)

    # load data (already processed)
    if config["eval_only"]:
        config['label2id'] = teacher.config.label2id
        train_dataset, eval_dataset, test_dataset = (
            None,
            None,
            build_dataset(config, "test", processor=teacher.processor),
        )
    else:
        train_dataset = build_dataset(config, "train", processor=teacher.processor)
        eval_dataset = build_dataset(config, "validation", processor=teacher.processor)
        test_dataset = build_dataset(config, "test", processor=teacher.processor)

    trainer_args = TrainingArguments(
        output_dir=os.path.join(MODELROOT, config["expt_name"]),
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=config["lr"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        save_total_limit=3 if not config["save_intermediate_teachers"] else 1000,
        push_to_hub=(not config["save_intermediate_teachers"]),
        hub_strategy="end",
        load_best_model_at_end=True,
        run_name=config["expt_name"],
        hub_model_id=config["expt_name"],  # this was the missing argument
    )
    if not config["eval_only"]:
        # put the model in training mode
        teacher.train()

    ## define trainer
    trainer = Trainer(
        teacher,
        trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=DefaultDataCollator(),
    )

    if not config["eval_only"]:
        try:
            train_results = trainer.train()
        except KeyboardInterrupt as e:
            print(e)

        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)

    prefix = 'test'
    if config['eval_only']:
        prefix += f'_{config["dataset"].split("/")[-1]})"]'
    trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="prefix")

    if config["eval_only"]:
        return
    # run teacher fully supervised evaluation
    if config["dataset"] == "rvl_cdip" and "100" in config["test_dataset"]:
        config["dataset"] = config["test_dataset"]
        test_dataset = build_dataset(config, "test", processor=teacher.processor)
        trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test_100")  # test on 25 examples per class

    trainer.push_to_hub("Saving best model to hub")


def teacher_argparse(return_parser=False):
    parser = argparse.ArgumentParser(description="DistilDoc_logits_ViT")

    parser.add_argument("--expt_name", type=str, default="teacher")

    # teacher arguments
    parser.add_argument("--teacher_model", type=str, default="dit", choices=["dit", "vit", "swin", "cnn"])

    parser.add_argument(
        "--sup_teacher_weights",
        type=str,
        default="",
    )

    # data arguments
    parser.add_argument("--dataset", type=str, default="rvl_cdip")
    parser.add_argument("--test_dataset", type=str, default="jordyvl/rvl_cdip_100_examples_per_class")

    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--hp", action="store_true", default=False, help="activate hyperparameter tuning")
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--student_mode", action="store_true", default=False, help="directly train student")
    parser.add_argument("--eval_only", action="store_true", default=False, help="only evaluate")
    parser.add_argument(
        "--save_intermediate_teachers",
        action="store_true",
        default=False,
        help="locally save teacher checkpoints at regular intervals",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    if return_parser:
        return parser
    hp = parser.parse_args()
    return hp


if __name__ == "__main__":
    main(teacher_argparse())
