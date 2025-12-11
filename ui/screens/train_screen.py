from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSpinBox,
    QDoubleSpinBox, QPushButton, QTextEdit, QLineEdit
)
from PySide6.QtCore import Signal

class TrainScreen(QWidget):

    start_training_requested = Signal(dict)

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(12)

        title = QLabel("Treinamento do Modelo")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: white;")
        layout.addWidget(title)

        subtitle = QLabel("Configure os parâmetros antes de iniciar o treinamento")
        subtitle.setStyleSheet("font-size: 14px; color: #cccccc;")
        layout.addWidget(subtitle)

        self.display_name_input = QLineEdit()
        self.display_name_input.setPlaceholderText("Digite o nome do modelo")
        self.display_name_input.setStyleSheet(
            "background-color: #1e1e1e; color: white; padding: 6px;"
        )
        layout.addWidget(QLabel("Nome do Modelo:"))
        layout.addWidget(self.display_name_input)

        self.max_length_input = QSpinBox()
        self.max_length_input.setRange(16, 512)
        self.max_length_input.setValue(128)
        self.max_length_input.setSingleStep(16)
        self.max_length_input.setStyleSheet("background-color: #1e1e1e; color: white; padding: 6px;")
        layout.addWidget(QLabel("Comprimento máximo:"))
        layout.addWidget(self.max_length_input)

        self.epoch_input = QSpinBox()
        self.epoch_input.setRange(1, 300000)
        self.epoch_input.setValue(3)
        self.epoch_input.setStyleSheet("background-color: #1e1e1e; color: white; padding: 6px;")
        layout.addWidget(QLabel("Épocas:"))
        layout.addWidget(self.epoch_input)

        self.batch_input = QSpinBox()
        self.batch_input.setRange(1, 128)
        self.batch_input.setValue(8)
        self.batch_input.setStyleSheet("background-color: #1e1e1e; color: white; padding: 6px;")
        layout.addWidget(QLabel("Batch size:"))
        layout.addWidget(self.batch_input)

        self.lr_input = QDoubleSpinBox()
        self.lr_input.setDecimals(7)
        self.lr_input.setRange(0.0000001, 1.0)
        self.lr_input.setSingleStep(0.000001)
        self.lr_input.setValue(0.00002)
        self.lr_input.setStyleSheet("background-color: #1e1e1e; color: white; padding: 6px;")
        layout.addWidget(QLabel("Learning rate:"))
        layout.addWidget(self.lr_input)

        self.start_button = QPushButton("Iniciar Treinamento")
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

        self.start_button.clicked.connect(self.emit_training_request)

    def emit_training_request(self):
        config = {
            "display_name": self.display_name_input.text().strip(),
            "max_length": self.max_length_input.value(),
            "epochs": self.epoch_input.value(),
            "batch_size": self.batch_input.value(),
            "learning_rate": self.lr_input.value()
        }
        self.start_training_requested.emit(config)

    def append_console(self, text):
        self.output_console.append(text)