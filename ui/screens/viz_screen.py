from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton, QTextEdit
from PySide6.QtCore import Signal
from backend.models_registry import load_registry
import json
import os

class VizScreen(QWidget):
    load_visualization_requested = Signal(str)

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(18)

        title = QLabel("Visualização do Modelo")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: white;")
        layout.addWidget(title)

        subtitle = QLabel("Selecione um modelo treinado e visualize métricas e saídas")
        subtitle.setStyleSheet("font-size: 14px; color: #cccccc;")
        layout.addWidget(subtitle)

        self.model_selector = QComboBox()
        self.model_selector.setStyleSheet(
            "background-color: #1e1e1e; color: white; padding: 6px;"
        )
        layout.addWidget(self.model_selector)

        self.load_button = QPushButton("Carregar Visualizações")
        self.load_button.setStyleSheet(
            "background-color: #0066cc; color: white; padding: 10px; border-radius: 6px;"
        )
        layout.addWidget(self.load_button)

        self.display_area = QTextEdit()
        self.display_area.setReadOnly(True)
        self.display_area.setStyleSheet(
            "background-color: #111111; color: #33ccff; padding: 12px; font-family: monospace;"
        )
        layout.addWidget(self.display_area)

        # conexões
        self.load_button.clicked.connect(self.emit_viz_request)

        # carregar modelos
        self.model_mapping = {}  # display_name -> (model_dir, id)
        self.load_models_from_registry()

    def emit_viz_request(self):
        model_name = self.model_selector.currentText()
        model_info = self.model_mapping.get(model_name)
        if model_info:
            model_dir, model_id = model_info
            self.load_visualization_requested.emit(model_dir)

            # carregar JSON correto na pasta models_registry
            registry_dir = "models_registry"  # pasta onde estão os JSONs
            json_path = os.path.join(registry_dir, f"{model_id}.json")
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    metrics = data.get("metrics", {})
                    if metrics:
                        metrics_to_show = {
                            "precision": metrics.get("eval_precision"),
                            "recall": metrics.get("eval_recall"),
                            "f1": metrics.get("eval_f1"),
                            "loss": metrics.get("eval_loss"),
                            "runtime": metrics.get("eval_runtime"),
                            "samples_per_second": metrics.get("eval_samples_per_second"),
                            "steps_per_second": metrics.get("eval_steps_per_second"),
                            "epoch": metrics.get("epoch")
                        }
                        self.show_model_metrics(metrics_to_show)
                    else:
                        self.show_visualization_text("Não há métricas disponíveis neste modelo.")
            else:
                self.show_visualization_text(f"JSON do modelo não encontrado em {json_path}.")

    def load_models_from_registry(self):
        """Carrega os modelos disponíveis do registro"""
        models = load_registry()
        display_names = []
        for m in models:
            name = m.get("display_name", m.get("id"))
            model_id = m.get("id")
            model_dir = m.get("model_dir")
            self.model_mapping[name] = (model_dir, model_id)
            display_names.append(name)
        self.populate_models(display_names)

    def populate_models(self, model_list):
        self.model_selector.clear()
        self.model_selector.addItems(model_list)

    def show_visualization_text(self, text):
        self.display_area.clear()
        self.display_area.append(text)

    def show_model_metrics(self, metrics: dict):
        """Exibe métricas principais do modelo"""
        text = "\n=== Métricas do Modelo ===\n"
        precision = metrics.get("precision")
        recall = metrics.get("recall")
        f1 = metrics.get("f1")
        loss = metrics.get("loss")
        runtime = metrics.get("runtime")
        samples_per_second = metrics.get("samples_per_second")
        steps_per_second = metrics.get("steps_per_second")
        epoch = metrics.get("epoch")

        if precision is not None:
            text += f"Precision: {precision:.3f}\n"
        if recall is not None:
            text += f"Recall:    {recall:.3f}\n"
        if f1 is not None:
            text += f"F1-score:  {f1:.3f}\n"
        if loss is not None:
            text += f"Loss:      {loss:.3f}\n"
        if runtime is not None:
            text += f"Runtime:   {runtime:.2f}s\n"
        if samples_per_second is not None:
            text += f"Amostras/s: {samples_per_second:.2f}\n"
        if steps_per_second is not None:
            text += f"Steps/s:    {steps_per_second:.2f}\n"
        if epoch is not None:
            text += f"Época:      {epoch}\n"

        self.display_area.clear()
        self.display_area.append(text)