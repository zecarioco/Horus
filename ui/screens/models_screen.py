from PySide6.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QLabel
from PySide6.QtCore import Signal, Qt
import os

from backend.models_registry import load_registry


class ModelsScreen(QWidget):
    model_selected = Signal(dict)

    def __init__(self):
        super().__init__()
        self.models = []

        layout = QVBoxLayout(self)

        self.title = QLabel("Modelos Treinados Disponíveis")
        self.title.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
        layout.addWidget(self.title)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([
            "Nome", "F1", "Precision", "Recall"
        ])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.table.cellClicked.connect(self.on_model_clicked)

        layout.addWidget(self.table)

        self.load_models()

    def load_models(self):
        self.models = load_registry()
        self.table.setRowCount(len(self.models))

        for row, entry in enumerate(self.models):
            metrics = entry.get("metrics", {})
            metadata = entry.get("metadata", {})

            f1 = metrics.get("eval_f1", "-")
            precision = metrics.get("eval_precision", "-")
            recall = metrics.get("eval_recall", "-")

            timestamp = entry.get("timestamp", metadata.get("timestamp", ""))

            model_dir = entry.get("model_dir", "")
            display_name = os.path.basename(os.path.dirname(model_dir)) if not model_dir else entry.get("display_name", "N/A")

            self.table.setItem(row, 0, QTableWidgetItem(display_name))
            self.table.setItem(row, 1, QTableWidgetItem(str(f1)))
            self.table.setItem(row, 2, QTableWidgetItem(str(precision)))
            self.table.setItem(row, 3, QTableWidgetItem(str(recall)))

        self.table.resizeColumnsToContents()

    def on_model_clicked(self, row, col):
        entry = self.models[row]
        self.model_selected.emit(entry)