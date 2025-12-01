from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSpinBox, QComboBox,
    QPushButton, QTextEdit, QLineEdit, QHBoxLayout
)
from PySide6.QtCore import Signal
from backend.models_registry import load_registry


class ExplainScreen(QWidget):

    explain_requested = Signal(dict)

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(12)

        title = QLabel("Explicação do Modelo (SHAP / LIME)")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: white;")
        layout.addWidget(title)

        subtitle = QLabel("Escolha o modelo e gere explicações para um texto")
        subtitle.setStyleSheet("font-size: 14px; color: #cccccc;")
        layout.addWidget(subtitle)

        layout.addWidget(QLabel("Selecione o modelo:"))
        self.model_selector = QComboBox()
        self.model_selector.setStyleSheet("background-color: #2b2b2b; color: white; padding: 6px;")
        layout.addWidget(self.model_selector)

        self.model_mapping = {}
        self.load_models_from_registry()

        layout.addWidget(QLabel("Texto a ser explicado:"))
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Digite aqui o texto…")
        self.text_input.setStyleSheet("background-color: #1e1e1e; color: white; padding: 6px;")
        self.text_input.setFixedHeight(100)
        layout.addWidget(self.text_input)

        layout.addWidget(QLabel("Método de explicação:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["SHAP", "LIME"])
        self.method_combo.setStyleSheet("background-color: #1e1e1e; color: white; padding: 6px;")
        layout.addWidget(self.method_combo)

        params_label = QLabel("Parâmetros do método:")
        params_label.setStyleSheet("font-size: 16px; color: #cccccc; margin-top: 10px;")
        layout.addWidget(params_label)

        self.lime_params_container = QWidget()
        lime_container_layout = QVBoxLayout(self.lime_params_container)
        lime_container_layout.setContentsMargins(0, 0, 0, 0)

        lime_samples_layout = QHBoxLayout()
        self.num_samples_input = QSpinBox()
        self.num_samples_input.setRange(50, 5000)
        self.num_samples_input.setValue(1000)
        self.num_samples_input.setStyleSheet("background-color: #1e1e1e; color: white; padding: 6px;")

        lime_samples_layout.addWidget(QLabel("LIME - num_samples:"))
        lime_samples_layout.addWidget(self.num_samples_input)
        lime_container_layout.addLayout(lime_samples_layout)

        lime_feat_layout = QHBoxLayout()
        self.num_features_input = QSpinBox()
        self.num_features_input.setRange(1, 50)
        self.num_features_input.setValue(10)
        self.num_features_input.setStyleSheet("background-color: #1e1e1e; color: white; padding: 6px;")

        lime_feat_layout.addWidget(QLabel("LIME - num_features:"))
        lime_feat_layout.addWidget(self.num_features_input)
        lime_container_layout.addLayout(lime_feat_layout)

        layout.addWidget(self.lime_params_container)

        self.start_button = QPushButton("Gerar Explicação")
        self.start_button.setStyleSheet(
            "background-color: #0066cc; color: white; padding: 10px; border-radius: 6px;"
        )
        layout.addWidget(self.start_button)

        self.output_console = QTextEdit()
        self.output_console.setReadOnly(True)
        self.output_console.setStyleSheet(
            "background-color: #111111; color: #00ff99; padding: 10px; font-family: monospace;"
        )
        layout.addWidget(self.output_console)

        self.start_button.clicked.connect(self.emit_explain_request)
        self.method_combo.currentTextChanged.connect(self.update_visible_params)

        self.update_visible_params(self.method_combo.currentText())

    def update_visible_params(self, method):
        is_lime = (method == "LIME")
        self.lime_params_container.setVisible(is_lime)

    def load_models_from_registry(self):
        models = load_registry()
        display_names = []
        for m in models:
            name = m.get("display_name", m["id"])
            self.model_mapping[name] = m["model_dir"]
            display_names.append(name)
        self.populate_models(display_names)

    def populate_models(self, model_list):
        self.model_selector.clear()
        self.model_selector.addItems(model_list)

    def emit_explain_request(self):
        display_name = self.model_selector.currentText()
        model_dir = self.model_mapping.get(display_name)

        config = {
            "model_dir": model_dir,
            "text": self.text_input.toPlainText().strip(),
            "method": self.method_combo.currentText().lower(),
            "num_samples": self.num_samples_input.value(),
            "num_features": self.num_features_input.value()
        }

        self.explain_requested.emit(config)

    def append_console(self, text):
        self.output_console.append(text)