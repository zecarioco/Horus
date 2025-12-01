from PySide6.QtCore import QObject, Signal, QThread
from backend.train_script import run_training


class TrainingRunner(QObject):

    log = Signal(str)
    finished = Signal(dict)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        self.log.emit("Treinamento em andamento...\n")

        metrics = run_training(
            display_name=self.config["display_name"],
            max_length=self.config["max_length"],
            epochs=self.config["epochs"],
            lr=self.config["learning_rate"],
            batch_size=self.config["batch_size"]
        )

        self.log.emit("Treinamento completo.\n")
        self.finished.emit(metrics)