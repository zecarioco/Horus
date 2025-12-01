from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QTextEdit, QLineEdit
from PySide6.QtCore import Signal
from backend.models_registry import load_registry

class BiasScreen(QWidget):
    detection_requested = Signal(str, str)

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        title = QLabel("Detector de Discurso de Ódio")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: white;")
        layout.addWidget(title)

        self.model_selector = QComboBox()
        self.model_selector.setStyleSheet("background-color: #2b2b2b; color: white; padding: 6px;")
        layout.addWidget(QLabel("Selecione o modelo:"))
        layout.addWidget(self.model_selector)

        self.input_text = QLineEdit()
        self.input_text.setPlaceholderText("Digite a frase para análise")
        self.input_text.setStyleSheet("background-color: #1e1e1e; color: white; padding: 6px;")
        layout.addWidget(self.input_text)

        self.run_button = QPushButton("Rodar Detecção")
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0a84ff;
            }
        """)
        layout.addWidget(self.run_button)

        self.results_box = QTextEdit()
        self.results_box.setReadOnly(True)
        self.results_box.setStyleSheet("background-color: #1e1e1e; color: white; padding: 10px;")
        layout.addWidget(self.results_box)

        self.run_button.clicked.connect(self._emit_detection_request)

        self.model_mapping = {}
        self.load_models_from_registry()

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

    def _emit_detection_request(self):
        display_name = self.model_selector.currentText()
        text = self.input_text.text().strip()
        if text and display_name in self.model_mapping:
            model_dir = self.model_mapping[display_name]
            self.detection_requested.emit(model_dir, text)

    def display_results(self, text):
        self.results_box.setText(text)