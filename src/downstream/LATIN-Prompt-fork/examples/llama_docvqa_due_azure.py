import sys

sys.path.append("./")
import os
from dataclasses import dataclass, field
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import (
    set_seed,
    TrainingArguments,
    HfArgumentParser,
)
from transformers.data.data_collator import DataCollatorMixin
import datasets
import wandb

from metric.anls import ANLS
from utils.model_path_config import model_path_config
from utils import space_layout


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_task": (
        "You are asked to answer questions asked on a document image.\n"
        "The answers to questions are short text spans taken verbatim from the document. "
        "This means that the answers comprise a set of contiguous text tokens present in the document.\n"
        "Document:\n{document}\n\n"
        "Question: {question}\n\n"
        "Directly extract the answer of the question from the document with as few words as possible .\n\n"
        "Answer:"
    ),
    "prompt_task_DLA": (
        "You are asked to answer questions asked on a document image.\n"
        "The answers to questions are short text spans taken verbatim from the document. "
        "This means that the answers comprise a set of contiguous text tokens present in the document.\n"
        "The logical layout regions are indicated with opening and closing tags such as <Table> and </Table> to help localizing answers. \n"
        "Document:\n{document}\n\n"
        "Question: {question}\n\n"
        "Directly extract the answer of the question from the document with as few words as possible .\n\n"
        "Answer:"
    ),
    "prompt_plain": (
        "Document:\n{document}\n\n"
        "Question: {question}\n\n"
        "Directly extract the answer of the question from the document.\n\n"
        "Answer:"
    ),
}


def load_llama(custom_args):
    if torch.cuda.is_available():
        # Load the entire model on the GPU 0
        device_map = {"": 0}

    else:
        device_map = None

    ################################################################################
    # bitsandbytes parameters
    ################################################################################

    # Activate 4-bit precision base model loading
    use_4bit = True

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        try:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16: accelerate training with bf16=True")
                print("=" * 80)
        except Exception as e:
            print(e)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        custom_args.model_name_or_path, quantization_config=bnb_config, device_map=device_map  # "auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    processor = AutoTokenizer.from_pretrained(custom_args.model_name_or_path, trust_remote_code=True)
    processor.pad_token = processor.eos_token
    processor.padding_side = "right"  # Fix weird overflow issue with fp16 training

    return processor, model


@dataclass
class CustomArguments:
    model_name_or_path: str = field(
        default="llama-7b",
        metadata={
            "help": "Path to pretrained model or model identifier\
                  from huggingface.co/models"
        },
    )
    dataset_name: str = field(
        default="docvqa", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    results_dir: str = field(default="results", metadata={"help": "The directory to save the results."})
    datas_dir: str = field(default="", metadata={"help": "The directory to save the datas."})
    wandb_project: str = field(default="Layout", metadata={"help": "The name of the wandb project."})
    prompt: str = field(default="plain", metadata={"help": "The prompt type. (plain, alpaca, layout)"})
    comment: str = field(default="", metadata={"help": "The comment for this run."})

    def __post_init__(self):
        self.model_name_or_path = model_path_config[self.model_name_or_path]
        self.datas_dir = os.path.expanduser(self.datas_dir)


class DataCollatorForDocVQA(DataCollatorMixin):
    def __init__(self, prompt_type, two_stage=False):
        super().__init__()
        self.prompt_type = prompt_type
        # DLA_FEATURES

    def space_layout(self, texts, boxes):
        return space_layout.space_layout(texts, boxes)

    def __call__(self, features):
        batch = {
            "text": [],
            "question": [],
            "answers": [],
            "questionId": [],
            "question_types": [],
        }
        for example in features:
            question = example["question"]
            batch["question"].append(question)
            batch["question_types"].append(example["question_types"])

            # TODO: extend with DLA features (done automatically in OCR, potentially extend prompt with DLA instruction)

            if self.prompt_type == "plain":
                doc = " ".join(example["texts"])
                text = PROMPT_DICT["prompt_plain"].format_map({"document": doc, "question": question})
            elif self.prompt_type == "task_instruction_space_DLA":
                space_line_texts = self.space_layout(
                    example["texts"],
                    example["text_boxes"],
                )
                doc = "\n".join(space_line_texts)
                text = PROMPT_DICT["prompt_task_DLA"].format_map({"document": doc, "question": question})
            elif self.prompt_type == "task_instruction_DLA":
                doc = " ".join(example["texts"])
                text = PROMPT_DICT["prompt_task_DLA"].format_map({"document": doc, "question": question})
            elif self.prompt_type == "task_instruction_space":
                space_line_texts = self.space_layout(
                    example["texts"],
                    example["text_boxes"],
                )
                doc = "\n".join(space_line_texts)
                text = PROMPT_DICT["prompt_task"].format_map({"document": doc, "question": question})
            elif self.prompt_type == "task_instruction":
                doc = " ".join(example["texts"])
                text = PROMPT_DICT["prompt_task"].format_map({"document": doc, "question": question})
            elif self.prompt_type == "space":
                space_line_texts = self.space_layout(
                    example["texts"],
                    example["text_boxes"],
                )
                doc = "\n".join(space_line_texts)
                text = PROMPT_DICT["prompt_plain"].format_map({"document": doc, "question": question})
            else:
                raise ValueError("Invalid prompt type.")

            batch["text"].append(text)
            batch["answers"].append(example["answers"])
            batch["questionId"].append(example["questionId"])

        return batch


def main():
    parser = HfArgumentParser((CustomArguments, TrainingArguments))
    custom_args, training_args = parser.parse_args_into_dataclasses()
    for k, v in custom_args.__dict__.items():
        print(k, v)

    to_log = ["dataset_name", "model_name_or_path", "prompt", "comment"]
    logged_config = {k: v for k, v in custom_args.__dict__.items() if k in to_log}
    # some debugging control variables
    SKIP_TEST = True
    DISABLE_WANDB = False
    DOWNSAMPLING = False

    if not DISABLE_WANDB:
        wandb.init(
            project=custom_args.wandb_project,
            name=training_args.run_name,
            config=logged_config,
        )

    set_seed(training_args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor, model = load_llama(custom_args)

    assert custom_args.dataset_name in ["docvqa_due_azure", "infographics_vqa_due_azure"]
    # safe reloading as HF likes to cache the DLA extended one...
    data = datasets.load_dataset(
        f"utils/{custom_args.dataset_name}.py",
        dla_model=custom_args.comment if custom_args.comment else "",
        download_mode="force_redownload",
        ignore_verifications=True,
    )

    anls_metric = ANLS(
        result_dir=custom_args.results_dir, exp_name=training_args.run_name, dataset_name=custom_args.dataset_name
    )

    collate_fn = DataCollatorForDocVQA(
        prompt_type=custom_args.prompt,
    )

    # evaluate on the validation dataset
    all_preds = []
    all_answers = []
    all_questions = []
    all_question_ids = []
    all_question_types = []
    count = 0
    print(f"Begin from the {count+1}-th example.")
    for i in tqdm(range(count, len(data["validation"])), desc="Processing"):
        example = data["validation"][i]
        print("=" * 30)
        batch = collate_fn([example])
        print(batch["text"][0])

        inputs = processor(
            text=batch["text"],
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        print(batch["answers"][0])
        try:
            generated_ids = model.generate(inputs.input_ids, max_new_tokens=100)
        except Exception as e:
            print(e)
            continue
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        generated_text = [t.strip()[len(batch["text"][i]) :].strip() for i, t in enumerate(generated_text)]

        print("Outputs:")
        print(generated_text)

        all_preds.extend(generated_text)
        all_answers.extend(batch["answers"])
        all_questions.extend(batch["question"])
        all_question_ids.extend(batch["questionId"])
        all_question_types.extend(batch["question_types"])
        if DOWNSAMPLING and i == 3:
            break
    val_anls_metrics = anls_metric.compute_and_save(
        qids=all_question_ids,
        questions=all_questions,
        predictions=all_preds,
        references=all_answers,
        question_types=all_question_types,
        split="val",
    )

    wandb.log(val_anls_metrics)
    print(val_anls_metrics)

    if SKIP_TEST:
        return
    # evaluate on the test dataset
    all_preds = []
    all_questions = []
    all_question_ids = []
    count = 0
    for i in tqdm(range(count, len(data["test"])), desc="Processing"):
        example = data["test"][i]
        print("=" * 30)
        batch = collate_fn([example])
        print(batch["text"][0])

        inputs = processor(
            text=batch["text"],
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        print(batch["answers"][0])
        try:
            generated_ids = model.generate(inputs.input_ids, max_new_tokens=100)
        except Exception as e:
            print(e)
            print(len(batch["text"][0]))
            continue
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # generated_text = [t.strip() for t in generated_text]
        generated_text = [t.strip()[len(batch["text"][i]) :].strip() for i, t in enumerate(generated_text)]

        print("Outputs:")
        print(generated_text)

        all_preds.extend(generated_text)
        all_questions.extend(batch["question"])
        all_question_ids.extend(batch["questionId"])

    test_anls_metrics = anls_metric.compute_and_save(
        qids=all_question_ids, questions=all_questions, predictions=all_preds, split="test"
    )
    wandb.log(test_anls_metrics)
    print(test_anls_metrics)


if __name__ == "__main__":
    main()
