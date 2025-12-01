from PySide6.QtCore import QObject, Signal
import shap
from lime.lime_text import LimeTextExplainer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import json
import os

class ExplainerRunner(QObject):

    log = Signal(str)
    finished = Signal(object)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            model_dir = self.config["model_dir"]
            text = self.config["text"]
            method = self.config["method"]
            num_samples = self.config.get("num_samples", 1000)

            self.log.emit("Carregando modelo...")
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model.eval()

            def predict(texts):
                if isinstance(texts, str):
                    texts = [texts]
                tokens = tokenizer(
                    texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=128
                )
                with torch.no_grad():
                    logits = model(**tokens).logits
                probs = torch.sigmoid(logits).cpu().numpy()
                return probs

            if method.lower() == "shap":
                self.log.emit("Executando SHAP...")
                masker = shap.maskers.Text(tokenizer)
                explainer = shap.Explainer(predict, masker)
                shap_values = explainer([text])
                tokens = shap_values.data[0]
                values = shap_values.values[0]
                word_importance = list(zip(tokens, values))
                result = {
                    "method": "shap",
                    "word_importance": word_importance,
                    "original_text": text
                }

            elif method.lower() == "lime":
                self.log.emit("Executando LIME...")
                labels_path = os.path.join(model_dir, "labels.json")
                if os.path.exists(labels_path):
                    labels = json.load(open(labels_path, "r", encoding="utf-8"))
                else:
                    labels = [f"class_{i}" for i in range(model.config.num_labels)]
                explainer = LimeTextExplainer(class_names=labels)
                exp = explainer.explain_instance(text, predict, num_samples=num_samples)
                lime_map = exp.as_map()
                explanation = {}
                for class_idx, token_vals in lime_map.items():
                    explanation[labels[class_idx]] = [
                        (exp.domain_mapper.index_to_word[int(tok_idx)], float(val))
                        for tok_idx, val in token_vals
                    ]
                result = {
                    "method": "lime",
                    "explanation": explanation,
                    "text": text
                }

            else:
                raise ValueError("Método inválido: use 'shap' ou 'lime'")

            self.log.emit("Explicação finalizada.")
            self.finished.emit(result)

        except Exception as e:
            self.log.emit(f"Erro: {str(e)}")
            self.finished.emit({"error": str(e)})