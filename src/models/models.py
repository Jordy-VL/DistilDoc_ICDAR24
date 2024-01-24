import os
import json
from collections import OrderedDict
from transformers import (
    AutoConfig,
    AutoModelForImageClassification,
    AutoProcessor,
    BeitConfig,
    ViTConfig,
    ResNetConfig,
    Swinv2Config,
)

CONFIGS = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../configs"))


def retrieve_teacher_path(config, key="teacher_model"):
    if key == "teacher_model":
        if "dit" in config[key].lower():
            return "microsoft/dit-base-finetuned-rvlcdip"
        elif "vit" in config[key].lower():
            return "google/vit-base-patch16-224-in21k"
        elif "swin" in config[key].lower():
            return "microsoft/swinv2-base-patch4-window8-256"
        elif "resnet" in config[key].lower() or "cnn" in config[key].lower():
            return "microsoft/resnet-101"

    if key == "teacher_weights":
        key = "teacher_model"

        if config[key] not in [
            "microsoft/dit-base-finetuned-rvlcdip",
            "jordyvl/dit-base_tobacco",
            "jordyvl/vit-base_tobacco",
            "dit",
            "swin",
            "vit",
            "cnn",
            # "dit-base",
            "swin-base",
            "vit-base",
            "microsoft/resnet-101",
        ]:
            return config[key]

        if "dit" in config[key].lower():
            if "rvl" in config["dataset"].lower():  # load trained teacher
                return "microsoft/dit-base-finetuned-rvlcdip"
            elif config["dataset"] == "maveriq/tobacco3482":  # load pretrained teacher
                return "jordyvl/dit-base_tobacco"

        elif "vit" in config[key].lower():
            if "rvl" in config["dataset"].lower():  # load trained teacher
                return "jordyvl/vit-base_rvl-cdip"
            elif config["dataset"] == "maveriq/tobacco3482":  # load pretrained teacher
                return "jordyvl/vit-base_tobacco"

        elif "swin" in config[key].lower():
            return "microsoft/swinv2-base-patch4-window8-256"  # "microsoft/swinv2-base-patch4-window12-192-22k"  # imagenet pretrained, yet lower resolution
            # microsoft/swin-base-patch4-window7-224-in22k
            # microsoft/swinv2-base-patch4-window8-256

        elif "cnn" in config[key].lower() or "resnet" in config[key].lower():
            if "rvl" in config["dataset"].lower():  # load trained teacher
                return "bdpc/resnet101_rvl-cdip"
            elif config["dataset"] == "maveriq/tobacco3482":  # load pretrained teacher
                return "bdpc/resnet101-base_tobacco"


def retrieve_student_path(config, key="student_model"):
    if "resnet" in config[key].lower():
        return "microsoft/resnet-50"
    elif "small" in config[key].lower():
        return "WinKawaks/vit-small-patch16-224"
    elif "tiny" in config[key].lower():
        return "WinKawaks/vit-tiny-patch16-224"

    # for dit would need to port beit-tiny and small from timm


def model_to_backbone(model, return_config=False):
    if "dit" in model:
        if return_config:
            return ViTConfig()  # BeitConfig() #similar to ViT tbh...
        return "dit"
    elif "vit" in model:
        if return_config:
            return ViTConfig()
        return "vit"
    elif "resnet" in model or "cnn" in model:
        if return_config:
            return ResNetConfig()
        return "cnn"


def build_model(config, key="teacher_model", add_abstention_logit=False, override=False, random_init=False):
    # https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k -> alternate
    # https://github.com/microsoft/unilm/blob/master/dit/classification/README.md

    if key == "student_model":  # dit-base, dit-tiny
        if "dit" in config[key].lower():
            config_type = BeitConfig
        elif "vit" in config[key].lower():
            config_type = ViTConfig
        elif "swin" in config[key].lower():
            config_type = Swinv2Config

        if "resnet" in config[key].lower() or "cnn" in config[key].lower():
            hf_model_config = ResNetConfig()
        else:
            hf_model_config = config_type.from_json_file(os.path.join(CONFIGS, f"{config[key]}.json"))

    if random_init:
        hf_model_config = model_to_backbone(config[key], return_config=True)
    elif key == "student_weights":
        hf_model_config = AutoConfig.from_pretrained(config[key])  # existing model

    elif key == "sup_teacher_weights":
        hf_model_config = AutoConfig.from_pretrained(config[key])

    else:
        config[key] = retrieve_teacher_path(config, key="teacher_weights")
        hf_model_config = AutoConfig.from_pretrained(config[key])

    if config["dataset"] == "maveriq/tobacco3482":
        id2label = OrderedDict(
            {
                0: "ADVE",
                1: "Email",
                2: "Form",
                3: "Letter",
                4: "Memo",
                5: "News",
                6: "Note",
                7: "Report",
                8: "Resume",
                9: "Scientific",
            }
        )
    else:  # rvl
        id2label = OrderedDict(
            {
                0: "letter",
                1: "form",
                2: "email",
                3: "handwritten",
                4: "advertisement",
                5: "scientific_report",
                6: "scientific_publication",
                7: "specification",
                8: "file_folder",
                9: "news_article",
                10: "budget",
                11: "invoice",
                12: "presentation",
                13: "questionnaire",
                14: "resume",
                15: "memo",
            }
        )
    if add_abstention_logit:
        id2label[len(id2label)] = "abstain"

    """
    teacher for student finetuning with response-based distillation should have same number of labels as student
    we would expect it to be reinitialized for training a new teacher on a new dataset
    when we load a pretrained (finetuned) teacher, we should use it as is, and not change the number of labels
    """
    if not hasattr(hf_model_config, "num_labels") or override:
        hf_model_config.num_labels = len(id2label)
        hf_model_config.id2label = id2label
        hf_model_config.label2id = OrderedDict({v: k for k, v in id2label.items()})

    elif hf_model_config.num_labels != len(id2label):
        if "teacher" in key:
            if not any([x in config.get("distill", "") for x in ["simkd", "hint"]]):  # response-based distillation
                print(f"Teacher model has {hf_model_config.num_labels} labels, but dataset has {len(id2label)} labels")
            else:  # no harm for reinitializing teacher classifier for simkd and hint; could keep as is
                pass
        else:
            hf_model_config.num_labels = len(id2label)
            hf_model_config.id2label = id2label
            hf_model_config.label2id = OrderedDict({v: k for k, v in id2label.items()})

    if key == "student_model":
        # model.ignore_mismatched_sizes = True
        model = AutoModelForImageClassification.from_config(hf_model_config)
        model.processor = AutoProcessor.from_pretrained(
            retrieve_teacher_path(config, key="teacher_model")
        )  # use teacher's processor

    elif key == "student_weights":
        model = AutoModelForImageClassification.from_pretrained(
            config[key],
            config=hf_model_config,
            ignore_mismatched_sizes=True,
        )
        model.processor = AutoProcessor.from_pretrained(retrieve_student_path(config, key="student_weights"))

    else:
        model = AutoModelForImageClassification.from_pretrained(
            config[key],
            config=hf_model_config,
            ignore_mismatched_sizes=True,
        )
        model.processor = AutoProcessor.from_pretrained(retrieve_teacher_path(config, key="teacher_model"))

    if key == "teacher_weights":
        # freeze teacher weights for distillation
        print("Freezing teacher weights for distillation")
        for param in model.parameters():
            param.requires_grad = False

    model = model.to(config["device"])
    return model, config


def tiny_update():
    return {
        "image_size": 224,
        "hidden_size": 192,
        "intermediate_size": 768,
        "num_attention_heads": 3,
        "num_hidden_layers": 12,
    }


def small_update():
    return {
        "image_size": 224,
        "hidden_size": 384,
        "intermediate_size": 1536,
        "num_attention_heads": 8,
        "num_hidden_layers": 12,
    }


def test_models():
    config = {"device": "cuda"}
    for dataset in ["rvl_cdip", "maveriq/tobacco3482"]:
        config["dataset"] = dataset
        for teacher in ["dit", "vit", "swin"]:
            config["teacher_model"] = teacher
            model, config = build_model(config, key="teacher_model")

            model.config.to_json_file(f"configs/{teacher}-base.json")

            # build students from updated config
            from copy import deepcopy

            new_config = deepcopy(model.config)

            ## keepem
            # new_config.id2label = None
            # new_config.label2id = None
            # new_config.num_labels = None

            # if teacher == "swin":
            #     continue

            new_config.update(small_update())
            new_config.to_json_file(f"configs/{teacher}-small.json")

            new_config.update(tiny_update())
            new_config.to_json_file(f"configs/{teacher}-tiny.json")


def test_student_models():
    config = {"device": "cuda"}
    for dataset in ["rvl_cdip", "maveriq/tobacco3482"]:
        config["dataset"] = dataset
        if dataset == "maveriq/tobacco3482":
            continue
        for student in ["dit", "vit", "swin"]:
            if student == "swin":
                continue
            for size in ["tiny", "small", "base"]:
                config["student_model"] = f"{student}-{size}"
                model, config = build_model(config, key="student_model")
                print(
                    f"Model: {student}-{size} - {round(sum(p.numel() for p in model.parameters())/10e+6, 3)}M parameters"
                )


if __name__ == "__main__":
    # test_models()
    test_student_models()
