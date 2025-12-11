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
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.log.emit("Carregando modelo...")
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            
            model.to(device) 
            model.eval()

            def predict(texts):
                if not isinstance(texts, np.ndarray):
                    try:
                        texts = np.array(texts)
                    except:
                        if isinstance(texts, str):
                            texts = [texts]

                if isinstance(texts, np.ndarray):
                    texts = texts.flatten().tolist()
                
                texts = [str(t) for t in texts if t is not None and t]

                if not texts:
                    return np.zeros((0, model.config.num_labels), dtype=np.float32)

                tokens = tokenizer(
                    texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                ).to(device)

                with torch.no_grad():
                    logits = model(**tokens).logits
                    probs = torch.sigmoid(logits).cpu().numpy()
                    return probs

            if method.lower() == "shap":
                self.log.emit("Executando SHAP...")
                
                masker = shap.maskers.Text(tokenizer)
                explainer = shap.Explainer(predict, masker)
                
                if not isinstance(text, str):
                    text_input = str(text)
                else:
                    text_input = text

                shap_values = explainer([text_input])
                
                tokens = shap_values.data[0]
                values = shap_values.values[0]
                
                word_importance = list(zip(tokens, values.tolist()))
                
                result = {
                    "method": "shap",
                    "word_importance": word_importance,
                    "original_text": text
                }

            elif method.lower() == "lime":
                self.log.emit("Executando LIME...")
                
                labels_path = os.path.join(model_dir, "labels.json")
                labels = None

                if os.path.exists(labels_path):
                    with open(labels_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            labels = list(data.keys())
                        else:
                            labels = data
                else:
                    if hasattr(model.config, 'id2label'):
                        labels = list(model.config.id2label.values())
                    else:
                        labels = [f"class_{i}" for i in range(model.config.num_labels)]

                explainer = LimeTextExplainer(class_names=labels)
                
                exp = explainer.explain_instance(text, predict, num_samples=num_samples, top_labels=5)
                
                explanation = {}
                for class_idx in exp.available_labels():
                    label_name = labels[class_idx] if class_idx < len(labels) else str(class_idx)
                    explanation[label_name] = [
                        (exp.domain_mapper.index_to_word[int(tok_idx)], float(val))
                        for tok_idx, val in exp.as_list(class_idx)
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
            error_message = f"Erro: {str(e)}"
            self.log.emit(error_message)
            self.finished.emit({"error": error_message})