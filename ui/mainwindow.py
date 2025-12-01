import json
import os
from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QStackedWidget
from PySide6.QtGui import QIcon
from PySide6.QtCore import QThread
from backend.training_runner import TrainingRunner
from backend.trainer_worker import TrainerWorker
from backend.explainer_runner import ExplainerRunner

from ui.sidebar import SideBar
from ui.screens.models_screen import ModelsScreen
from ui.screens.train_screen import TrainScreen
from ui.screens.explain_screen import ExplainScreen
from ui.screens.viz_screen import VizScreen
from ui.screens.bias_screen import BiasScreen


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hórus")
        self.setMinimumSize(1000, 600)
        self.setWindowIcon(QIcon("assets/horusicon.png"))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.sidebar = SideBar()
        self.stack = QStackedWidget()

        layout.addWidget(self.sidebar)
        layout.addWidget(self.stack)

        self.screens = {
            "models": ModelsScreen(),
            "train": TrainScreen(),
            "explain": ExplainScreen(),
            "viz": VizScreen(),
            "bias": BiasScreen()
        }

        for screen in self.screens.values():
            self.stack.addWidget(screen)

        self.sidebar.navigation_requested.connect(self.navigate)

        self.screens["train"].start_training_requested.connect(self.start_training_thread)
        self.screens["bias"].detection_requested.connect(self.handle_detection)
        self.screens["viz"].load_visualization_requested.connect(self.load_viz_metrics)
        self.screens["explain"].explain_requested.connect(self.start_explainer_thread)

    def navigate(self, screen_name):
        if screen_name in self.screens:
            self.stack.setCurrentWidget(self.screens[screen_name])


    def start_training_thread(self, config):
        self.thread = QThread()
        self.runner = TrainingRunner(config)
        self.runner.moveToThread(self.thread)

        self.runner.log.connect(self.screens["train"].append_console)
        self.runner.finished.connect(self.training_done)

        self.thread.started.connect(self.runner.run)
        self.thread.start()

    def training_done(self, metrics):
        self.screens["train"].append_console("Final metrics:\n")
        self.screens["train"].append_console(str(metrics))
        self.thread.quit()

    def handle_detection(self, model_dir, text):
        try:
            results = TrainerWorker.run_detection(model_dir, text)

            probs = results["probabilities"]
            passed = results["passed_thresholds"]
            thresholds = results["thresholds_used"]

            formatted = "Probabilidades:\n"
            for label, prob in probs.items():
                formatted += f"  {label}: {prob:.2f}  (thr={thresholds.get(label, 0.5):.2f})\n"

            formatted += "\nDetectados:\n"
            if passed:
                for label in passed:
                    formatted += f"  - {label}\n"
            else:
                formatted += "  Nenhum\n"

            self.screens["bias"].display_results(formatted)

        except Exception as e:
            self.screens["bias"].display_results(f"Erro ao rodar detecção:\n{str(e)}")

    def start_explainer_thread(self, config):
        self.thread = QThread()
        self.runner = ExplainerRunner(config)
        self.runner.moveToThread(self.thread)
        self.runner.log.connect(self.screens["explain"].append_console)
        self.runner.finished.connect(self.explainer_done)
        self.thread.started.connect(self.runner.run)
        self.thread.start()


    def explainer_done(self, result):
        screen = self.screens["explain"]

        screen.append_console("\n=== Resultado ===\n")
        screen.append_console(json.dumps(result, indent=2, ensure_ascii=False))

        self.thread.quit()
    
    def load_viz_metrics(self, model_dir: str):
        """
        Função chamada quando o usuário clica em 'Carregar Visualizações'.
        Carrega métricas do JSON do modelo e mostra na tela VizScreen.
        """
        metrics_path = os.path.join(model_dir, "metrics.json")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                self.screens["viz"].show_model_metrics(metrics)
            except Exception as e:
                self.screens["viz"].show_visualization_text(f"Erro ao carregar métricas:\n{str(e)}")
        else:
            self.screens["viz"].show_visualization_text("Nenhum arquivo de métricas encontrado.")