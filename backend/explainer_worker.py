import shap
from lime.lime_text import LimeTextExplainer
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import json

class ExplainerWorker:

    @staticmethod
    def run_explainer(model_dir, text, method="shap", num_samples=1000):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model.to(device)
            model.eval()
        except Exception as e:
            return {"error": f"Erro ao carregar modelo: {str(e)}"}

        def predict(texts):
            if not isinstance(texts, np.ndarray):
                try:
                    texts = np.array(texts)
                except:
                    if isinstance(texts, str):
                        texts = [texts]

            if isinstance(texts, np.ndarray):
                texts = texts.flatten().tolist()
            
            texts = [str(t) for t in texts if t is not None]

            texts = [t for t in texts if t]

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
                probs = torch.sigmoid(logits)
            
            return probs.cpu().numpy()

        if method == "shap":
            
            explainer = shap.Explainer(predict, masker=tokenizer)
            
            if not isinstance(text, str):
                text_input = str(text)
            else:
                text_input = text

            try:
                shap_values = explainer([text_input])
            except Exception as e:
                return {"error": f"Erro durante a execução do SHAP (Explainer): {str(e)}"}
            
            instance_values = shap_values[0]

            return {
                "method": "shap",
                "values": instance_values.values.tolist(),
                "base_values": instance_values.base_values.tolist() if hasattr(instance_values, 'base_values') else [],
                "tokens": instance_values.data,
                "feature_names": instance_values.feature_names if hasattr(instance_values, 'feature_names') else []
            }

        elif method == "lime":
            labels_path = os.path.join(model_dir, "labels.json")
            if os.path.exists(labels_path):
                with open(labels_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        class_names = list(data.keys())
                    else:
                        class_names = data
            else:
                class_names = [f"Class_{i}" for i in range(model.config.num_labels)]

            explainer = LimeTextExplainer(class_names=class_names)
            
            exp = explainer.explain_instance(
                text,
                predict,
                num_features=10,
                num_samples=num_samples,
                top_labels=5 
            )

            explanations = {}
            for label_idx in exp.available_labels():
                label_name = class_names[label_idx] if label_idx < len(class_names) else str(label_idx)
                explanations[label_name] = exp.as_list(label=label_idx)

            return {
                "method": "lime",
                "explanations": explanations,
                "text": text
            }

        else:
            raise ValueError("Método inválido (use 'shap' ou 'lime')")