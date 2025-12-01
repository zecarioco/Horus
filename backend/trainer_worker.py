import json
import os
import torch
from datetime import datetime
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    AutoTokenizer,
    Trainer
)
import torch.nn as nn
from backend.models_registry import register_model

def compute_metrics(pred):
    logits = torch.sigmoid(torch.tensor(pred.predictions))
    preds = (logits > 0.5).int()
    labels = torch.tensor(pred.label_ids).int()

    true_pos = (preds & labels).sum(dim=0).float()
    pred_pos = preds.sum(dim=0).float()
    actual_pos = labels.sum(dim=0).float()

    precision = (true_pos / (pred_pos + 1e-8)).mean().item()
    recall = (true_pos / (actual_pos + 1e-8)).mean().item()
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {"precision": precision, "recall": recall, "f1": f1}

class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        probs = torch.sigmoid(logits)

        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_factor = torch.where(targets == 1, alpha, 1 - alpha)
            focal_weight = focal_weight * alpha_factor

        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class MultiLabelTrainer(Trainer):

    def __init__(self, alpha=None, gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.loss_fct = MultiLabelFocalLoss(alpha=alpha, gamma=gamma)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss = self.loss_fct(logits, labels.float())

        if return_outputs:
            return loss, outputs
        return loss

class TrainerWorker:
    def __init__(
        self,
        model_name="neuralmind/bert-base-portuguese-cased",
        base_output_dir="./results",
        learning_rate=2e-5,
        epochs=3,
        batch_size=8,
        fp16=True,
        run_name=None,
        display_name=None
    ):
        self.model_name = model_name
        self.display_name = display_name
        self.base_output_dir = base_output_dir
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.fp16 = fp16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if run_name is None:
            run_name = f"{model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = os.path.join(self.base_output_dir, run_name)
        os.makedirs(self.output_dir, exist_ok=True)



    @staticmethod
    def compute_class_weights(dataset, label_cols):
        labels = torch.stack([x["labels"] for x in dataset], dim=0)
        positives = labels.sum(dim=0)
        negatives = (labels.shape[0] - positives)
        pos_weight = negatives / (positives + 1e-8)
        return pos_weight



    def load_or_initialize_model(self, num_labels):
        checkpoint = os.path.join(self.output_dir, "checkpoint-final")
        if os.path.exists(checkpoint):
            try:
                return AutoModelForSequenceClassification.from_pretrained(checkpoint).to(self.device)
            except Exception:
                pass

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )
        model.config.problem_type = "multi_label_classification"
        return model.to(self.device)



    @staticmethod
    def compute_optimal_thresholds(model, dataset, device, label_cols):
        model.eval()
        all_probs = []
        all_labels = []

        loader = torch.utils.data.DataLoader(dataset, batch_size=16)

        with torch.no_grad():
            for batch in loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].cpu()

                logits = model(**inputs).logits
                probs = torch.sigmoid(logits).cpu()

                all_probs.append(probs)
                all_labels.append(labels)

        all_probs = torch.cat(all_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        thresholds = {}

        for i, label in enumerate(label_cols):
            best_f1 = 0
            best_t = 0.5

            for t in [x / 100 for x in range(1, 100)]:
                preds = (all_probs[:, i] > t).int()
                true = all_labels[:, i].int()

                tp = (preds & true).sum().item()
                fp = ((preds == 1) & (true == 0)).sum().item()
                fn = ((preds == 0) & (true == 1)).sum().item()

                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)

                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t

            thresholds[label] = best_t

        return thresholds

    def train(self, train_dataset, test_dataset, tokenizer, num_labels, label_cols):
        model = self.load_or_initialize_model(num_labels)

        model.config.id2label = {i: label for i, label in enumerate(label_cols)}
        model.config.label2id = {label: i for i, label in enumerate(label_cols)}
        model.config.num_labels = len(label_cols)
        model.config.problem_type = "multi_label_classification"

        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.epochs,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            fp16=self.fp16 and torch.cuda.is_available(),
            load_best_model_at_end=False
        )

        alpha = None

        trainer = MultiLabelTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            alpha=alpha,
            gamma=2.0
        )

        trainer.train()
        trainer.save_model(self.output_dir)

        metrics = trainer.evaluate()

        thresholds = TrainerWorker.compute_optimal_thresholds(
            model=model,
            dataset=test_dataset,
            device=self.device,
            label_cols=label_cols
        )

        with open(f"{self.output_dir}/thresholds.json", "w", encoding="utf-8") as f:
            json.dump(thresholds, f, indent=4, ensure_ascii=False)

        metadata = {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "fp16": self.fp16,
            "device": str(self.device)
        }

        model_info = {
            "display_name": self.display_name,
            "id": os.path.basename(self.output_dir),
            "model_dir": self.output_dir,
            "metadata": metadata,
            "metrics": metrics,
            "labels": label_cols,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        register_model(model_info)
        return metrics



    @staticmethod
    def run_detection(model_dir: str, text: str, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        try:
            with open(f"{model_dir}/thresholds.json", "r", encoding="utf-8") as f:
                thresholds = json.load(f)
        except:
            thresholds = {}

        model.to(device)
        model.eval()

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).cpu().squeeze().tolist()

        labels = [model.config.id2label[i] for i in range(len(probs))]

        all_probs = {label: float(prob) for label, prob in zip(labels, probs)}

        passed = []
        for label, prob in all_probs.items():
            threshold = thresholds.get(label, 0.5)
            if prob >= threshold:
                passed.append(label)

        return {
            "probabilities": all_probs,
            "passed_thresholds": passed,
            "thresholds_used": thresholds
        }