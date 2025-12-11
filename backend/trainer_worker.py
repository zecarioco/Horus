import json
import os
import torch
import torch.nn as nn
from datetime import datetime
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    AutoTokenizer,
    Trainer
)
from backend.models_registry import register_model
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

def compute_metrics(pred):
    logits = pred.predictions
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    labels = pred.label_ids
    
    preds = (probs > 0.5).astype(int) 

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='micro', zero_division=0
    )
    
    return {
        "precision": precision, 
        "recall": recall, 
        "f1": f1
    }

class WeightedBCE_Trainer(Trainer):
    def __init__(self, pos_weight=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.pos_weight is not None:
            weights = self.pos_weight.to(logits.device)
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=weights)
        else:
            loss_fct = nn.BCEWithLogitsLoss()

        loss = loss_fct(logits, labels.float())

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
        fp16=False, 
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
    def compute_class_weights(dataset):
        labels = torch.stack([x["labels"] for x in dataset], dim=0)
        
        positives = labels.sum(dim=0)
        negatives = labels.shape[0] - positives
        
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
        threshold_range = np.arange(0.01, 1.0, 0.01)

        for i, label in enumerate(label_cols):
            best_f1 = -1.0
            best_t = 0.5
            
            y_true = all_labels[:, i].int().numpy()
            probs_col = all_probs[:, i].numpy()

            for t in threshold_range:
                y_pred = (probs_col > t).astype(int)
                
                tp = np.sum((y_pred == 1) & (y_true == 1))
                fp = np.sum((y_pred == 1) & (y_true == 0))
                fn = np.sum((y_pred == 0) & (y_true == 1))
                
                f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)

                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t

            thresholds[label] = float(best_t)

        return thresholds

    def train(self, train_dataset, test_dataset, tokenizer, num_labels, label_cols):
        model = self.load_or_initialize_model(num_labels)

        model.config.id2label = {i: label for i, label in enumerate(label_cols)}
        model.config.label2id = {label: i for i, label in enumerate(label_cols)}
        
        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.epochs,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=self.fp16 and torch.cuda.is_available(),
        )

        pos_weight = self.compute_class_weights(train_dataset)

        trainer = WeightedBCE_Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            pos_weight=pos_weight
        )

        trainer.train()
        trainer.save_model(self.output_dir)

        metrics = trainer.evaluate()

        best_model = AutoModelForSequenceClassification.from_pretrained(self.output_dir).to(self.device)

        thresholds = self.compute_optimal_thresholds(
            model=best_model,
            dataset=test_dataset,
            device=self.device,
            label_cols=label_cols
        )

        with open(f"{self.output_dir}/thresholds.json", "w", encoding="utf-8") as f:
            json.dump(thresholds, f, indent=4, ensure_ascii=False)

        metadata = {
            "model_name": self.model_name,
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
        except Exception:
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

        if isinstance(probs, float):
            probs = [probs]
            
        labels = [model.config.id2label.get(i, f"class_{i}") for i in range(len(probs))]

        all_probs = {label: float(prob) for label, prob in zip(labels, probs)}

        passed = []
        for label, prob in all_probs.items():
            threshold = thresholds.get(label, 0.4)
            if prob >= threshold:
                passed.append(label)

        return {
            "probabilities": all_probs,
            "passed_thresholds": passed,
            "thresholds_used": thresholds
        }