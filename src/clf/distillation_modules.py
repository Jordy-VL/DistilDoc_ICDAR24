import os
import sys
from copy import deepcopy
import numpy as np
import torch

from transformers import Trainer, TrainingArguments, DefaultDataCollator
from transformers.modeling_outputs import ImageClassifierOutput

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Distildoc ROOT
from default_config import SAVEROOT
from metrics import compute_metrics

# distillation losses
from distiller_zoo import NKDLoss

# additional projectors
from models.projectors import SimKD, ConvReg, SimTransKD, DualSimTransKD


class DistillationTrainingArguments(TrainingArguments):
    """Extending original TrainingArguments to include distillation hyperparameters"""

    def __init__(self, *args, alpha=0.5, temperature=1, gamma=0, beta=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature  # softening/sharpening the temperature of teacher logits
        self.gamma = gamma  # only relevant for NKD loss -> best to set temperature to 1
        self.beta = beta  # only relevant when using different combinations of losses


class DistillationTrainer(Trainer):
    """Extending original Trainer class to apply KL Divergence or MSE wrt teacher model"""

    def __init__(self, *args, teacher_model=None, distillation_loss=None, distill=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.distillation_loss = distillation_loss
        self.distill = distill

    def compute_loss(self, model, inputs, return_outputs=False):
        loss_fct = torch.nn.MSELoss()

        projector, config = None, None
        if hasattr(model, "projector"):
            projector = model.projector
            config = model.config
        elif hasattr(model, "module"):
            if hasattr(model.module, "projector"):
                projector = model.module.projector
                config = model.module.config

        if "hint" in self.distill:
            # replace with the hint encoder layer?
            ## Teacher downprojected to student size, then inputted instead of other hidden layer
            ## https://github.com/huggingface/transformers/blob/v4.27.0/src/transformers/models/vit/modeling_vit.py#L392
            outputs_stu = model(**inputs, output_hidden_states=True)
            hidden_states_t = self.teacher_model(
                **{k: v for k, v in inputs.items() if k != "labels"}, output_hidden_states=True
            ).hidden_states
            feat_s = projector.hidden_states_to_feature_map(outputs_stu.hidden_states, distill="hint-middle")
            hidden_states_t = projector.hidden_states_to_feature_map(hidden_states_t, distill="hint-middle")
            proj_feat_s, proj_feat_t = projector(feat_s, hidden_states_t)
            loss = loss_fct(proj_feat_s, hidden_states_t)
            loss_ce = outputs_stu.loss.mean()  # harmless in case of 1, useful in case of dataparallel
            loss += loss_ce
            outputs_stu = ImageClassifierOutput(
                loss=None, logits=outputs_stu.logits, hidden_states=None, attentions=None
            )

            return (loss, outputs_stu) if return_outputs else loss

        elif self.distill == "og_simkd":
            feat_s = model(**inputs, output_hidden_states=True).hidden_states[-1]
            feat_t = self.teacher_model(
                **{k: v for k, v in inputs.items() if k != "labels"}, output_hidden_states=True
            ).hidden_states[-1]
            # heuristic: self-distillation in case of resnets:
            if feat_t.shape != feat_s.shape:
                feat_s = projector.hidden_states_to_feature_map(feat_s)
                feat_t = projector.hidden_states_to_feature_map(feat_t)
            proj_feat_s, proj_feat_t = projector(feat_s, feat_t)
            loss = loss_fct(proj_feat_s, feat_t)
            cls_pooling = projector.avg_pool(proj_feat_s).view(proj_feat_s.size(0), -1)

            # average pool from projector and view
            outputs_stu = self.teacher_model.classifier(cls_pooling)
            outputs_stu = ImageClassifierOutput(loss=None, logits=outputs_stu, hidden_states=None, attentions=None)

            # no gradient updates for classifier?
            return (loss, outputs_stu) if return_outputs else loss

        elif "simkd" in self.distill:
            if projector is None:  # just pass student features to teacher CLS
                outputs_stu = model(**inputs, output_hidden_states=True)
                feat_s = outputs_stu.hidden_states[-1]
                feat_t = self.teacher_model(
                    **{k: v for k, v in inputs.items() if k != "labels"}, output_hidden_states=True
                ).hidden_states[-1]
                loss = loss_fct(feat_s, feat_t)
                outputs_stu = self.teacher_model.classifier(self.teacher_model.resnet.pooler(feat_s))
                outputs_stu = ImageClassifierOutput(loss=None, logits=outputs_stu, hidden_states=None, attentions=None)
                return (loss, outputs_stu) if return_outputs else loss

            # https://github.com/MaitySubhajit/DistilDoc_Dev/blob/21a83cf7ad036987b41fe353a331acd61f9a6ae9/clf/train_student.py#L47
            backbone = self.teacher_model.config.model_type

            # could have also implemented it with hidden_states[-1]
            backbone_t = getattr(self.teacher_model, backbone)
            backbone_s = getattr(model, config.model_type)

            feat_s = backbone_s(**{k: v for k, v in inputs.items() if k != "labels"})

            feat_s = (
                feat_s.pooler_output
                if hasattr(feat_s, "pooler_output") and feat_s.pooler_output is not None
                else feat_s.last_hidden_state[:, 0, :]
            )  # final hidden state of the *[CLS]* tokenm, preceded by pooling/layernorm

            if self.distill == "simkd":
                proj_feat_s = model.projector(feat_s)

            feat_t = backbone_t(**{k: v for k, v in inputs.items() if k != "labels"})
            feat_t = (
                feat_t.pooler_output
                if hasattr(feat_t, "pooler_output") and feat_t.pooler_output is not None
                else feat_t.last_hidden_state[:, 0, :]
            )  # CLS token

            if "dual" in self.distill:
                proj_feat_s, proj_feat_t = projector(feat_s, feat_t)

            # gradient updates only happen
            loss = loss_fct(proj_feat_s, feat_t)
            if "dual" in self.distill:
                loss += loss_fct(proj_feat_t, feat_s)

            outputs_stu = self.teacher_model.classifier(proj_feat_s)
            outputs_stu = ImageClassifierOutput(loss=None, logits=outputs_stu, hidden_states=None, attentions=None)

            # no gradient updates for classifier?
            return (loss, outputs_stu) if return_outputs else loss

        outputs_stu = model(**inputs)  # trainable

        # Extract cross-entropy loss and logits from student
        loss_ce = outputs_stu.loss
        logits_stu = outputs_stu.logits

        # LOSS FUNCTIONS

        if self.distillation_loss == "CE":
            loss = loss_ce

        # Extract logits from teacher
        with torch.no_grad():  # fixed
            outputs_tea = self.teacher_model(**{k: v for k, v in inputs.items() if k != "labels"})
            logits_tea = outputs_tea.logits

        ## https://github.com/jhoon-oh/kd_data/blob/main/code/image_classfication/tools/losses.py
        if self.distillation_loss in "CE+KD":
            # Soften probabilities and compute distillation loss
            ## https://github.com/haitongli/knowledge-distillation-pytorch/issues/2
            loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
            loss_kd = self.args.temperature**2 * loss_fct(
                torch.nn.functional.log_softmax(logits_stu / self.args.temperature, dim=-1),
                torch.nn.functional.softmax(logits_tea / self.args.temperature, dim=-1),
            )
            # Return weighted student loss
            loss = self.args.alpha * loss_ce + (1.0 - self.args.alpha) * loss_kd

        # follow https://www.ijcai.org/proceedings/2021/0362.pdf
        ## direct logit matching
        ### Only sugggested when using a strong teacher
        elif self.distillation_loss == "MSE":
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(logits_stu, logits_tea)

        # TODO: IDEA first CE+KD, then MSE halfway through training [can get training?]

        # novel SOTA
        elif self.distillation_loss == "NKD":
            loss_fct = NKDLoss(self.args.temperature, self.args.gamma)
            loss = loss_fct(logits_stu, logits_tea, inputs["labels"])

        return (loss, outputs_stu) if return_outputs else loss


class KnowledgeDistillation:
    def __init__(
        self,
        config,
        teacher_model,
        student_model,
        train_dataset,
        eval_dataset,
        test_dataset=None,
        hyperparameters=None,
        hp_tuning=False,
    ):
        self.config = config
        self.experiment_name = config["expt_name"]
        self.distillation_loss = config["distill_loss"]
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.hyperparameters = hyperparameters
        self.hp_tuning = hp_tuning

    def load_student_model(self):
        if self.config["distill"] == "simkd":
            if not "hidden_size" in self.student_model.config.__dict__:
                projector_shape = [None, 2048, 7, 7]
                self.student_model.projector = None
            else:
                # https://arxiv.org/abs/2203.14001
                # https://github.com/DefangChen/SimKD/blob/main/train_student.py
                self.student_model.projector = SimTransKD(
                    s_n=self.student_model.config.hidden_size, t_n=self.teacher_model.config.hidden_size
                )

        elif self.config["distill"] == "og_simkd":
            if not "hidden_size" in self.student_model.config.__dict__:
                projector_shape = [None, 2048, 7, 7]
                self.student_model.projector = SimKD(
                    s_n=projector_shape[1],
                    t_n=projector_shape[1],
                    patch_dim=projector_shape[-1],
                )
            else:
                self.student_model.projector = SimKD(
                    s_n=self.student_model.config.hidden_size,
                    t_n=self.teacher_model.config.hidden_size,
                    patch_dim=self.student_model.config.image_size // self.student_model.config.patch_size,
                )

        elif self.config["distill"] == "dualsimkd":
            self.student_model.projector = DualSimTransKD(
                s_n=self.student_model.config.hidden_size, t_n=self.teacher_model.config.hidden_size
            )
        elif "hint" in self.config["distill"]:  # fitnet on all/some hidden layers
            # adjust config to return all hidden states

            if not "hidden_size" in self.student_model.config.__dict__:
                student_shape = [None, 1024, 14, 14]
                teacher_shape = student_shape  # seems similar
            else:
                student_shape = [
                    None,  # batch size
                    # self.student_model.config.num_hidden_layers + 1,  # x12 for transformer encoders + 1 embedding
                    self.student_model.config.hidden_size,
                    self.student_model.config.image_size // self.student_model.config.patch_size,
                    self.student_model.config.image_size // self.student_model.config.patch_size,
                ]
                teacher_shape = deepcopy(student_shape)
                teacher_shape[-3] = self.teacher_model.config.hidden_size
            self.student_model.projector = ConvReg(student_shape, teacher_shape)  # x12 for transformer encoders

        # set student ready to train -> also activated .grad for all new parameters [should classifier be gradable? if not -> keep in teacher with no updates]
        self.student_model.train()

        if self.distillation_loss == "CE":  # no need for teacher - just vanilla student finetuning
            self.teacher_model = None
        else:
            # set teacher without gradients
            self.teacher_model.eval()

        # ## set teacher on other device if using data parallel
        # if torch.cuda.device_count() > 1:
        #     devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        #     self.teacher_model = self.teacher_model.to(devices[-1])  # last device

        return self.student_model

    def load_training_args(self):
        student_training_args = DistillationTrainingArguments(
            output_dir=os.path.join(SAVEROOT, self.experiment_name),
            save_strategy="epoch",
            evaluation_strategy="epoch",
            learning_rate=self.config["lr"],
            per_device_train_batch_size=self.config["batch_size"],
            per_device_eval_batch_size=self.config["batch_size"],
            auto_find_batch_size=False,
            num_train_epochs=self.config["epochs"],
            weight_decay=self.config["weight_decay"],
            warmup_ratio=self.config["warmup_ratio"],
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            save_total_limit=3,
            push_to_hub=True,
            hub_strategy="end",
            load_best_model_at_end=True,
            run_name=self.experiment_name,
            hub_model_id=self.experiment_name,
            alpha=self.config["alpha"],
            temperature=self.config["temperature"],
            gamma=self.config["gamma"],
            beta=self.config["beta"],
            remove_unused_columns=not self.config["precompute"],
        )
        if self.hyperparameters:  # overrides the defaults?
            for k, v in self.hyperparameters.items():
                if hasattr(student_training_args, k):
                    setattr(student_training_args, k, v)
                else:
                    print(f"student_training_args does not have argument {k}")

        return student_training_args

    def setup(self):
        student_training_args = self.load_training_args()

        distill_trainer = DistillationTrainer(
            model_init=self.load_student_model,
            teacher_model=self.teacher_model,
            args=student_training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=compute_metrics,
            distill=self.config["distill"],
            distillation_loss=self.distillation_loss,
            data_collator=DefaultDataCollator(),
        )
        return distill_trainer

    def train_student_model(self):
        distill_trainer = self.setup()

        if self.hp_tuning:
            # default is the sum of all metrics, so maximizing accuracy in this instance
            best_run = distill_trainer.hyperparameter_search(
                n_trials=self.config["n_trials"], direction="maximize", hp_space=self.hp_space
            )

            # get the best hyperparams, then train the final model
            for n, v in best_run.hyperparameters.items():
                setattr(
                    distill_trainer.args, n, v
                )  # for running the experiment with the best hyperparameters from the hyperparameters search

        distill_trainer.train()

        return distill_trainer

    def hp_space(self, trial):  # pip3 install optuna
        return {
            # "num_train_epochs": trial.suggest_int("num_train_epochs", 5, 10),
            "alpha": trial.suggest_float("alpha", 0.5, 1, step=0.05),
            # "beta": trial.suggest_float("beta", 0.5, 1, step=0.05),
            "gamma": trial.suggest_float("gamma", 1.5, 3, step=0.25),
            "temperature": trial.suggest_float("temperature", 1, 20, step=2),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        }

    def precompute_teacher(self):
        distill_trainer = self.setup()
        # train

        # initialize logits
        logits = np.zeros((len(self.train_dataset), self.teacher_model.config.num_labels))
        hidden_states = np.zeros((len(self.train_dataset), self.teacher_model.config.hidden_size))

        for batch in distill_trainer.get_train_dataloader():
            from pdb import set_trace

            set_trace()
            batch = {k: v.to(distill_trainer.teacher_model.device) for k, v in batch.items()}
            og_indices = batch.pop("indices")
            outputs = distill_trainer.teacher_model(**batch)
            hidden_states = outputs.hidden_states
            # hidden_states = [x.detach().cpu().numpy() for x in hidden_states]
            logits = outputs.logits.detach().cpu().numpy()
