from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PySide6.QtCore import Qt, Signal


class SideBar(QWidget):

    navigation_requested = Signal(str)

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        self.setFixedWidth(200)
        self.setStyleSheet("background-color: #181818;")

        buttons = [
            ("Início", "home"),
            ("Modelos", "models"),
            ("Treinamento", "train"),
            ("Explicabilidade", "explain"),
            ("Viés", "bias"),
            ("Visualização", "viz"),
        ]

        for text, key in buttons:
            btn = QPushButton(text)
            btn.setStyleSheet("""
                QPushButton {
                    color: white;
                    background-color: #202020;
                    border-radius: 6px;
                    padding: 10px;
                    text-align: left;
                }
                QPushButton:hover {
                    background-color: #2e2e2e;
                }
            """)
            btn.clicked.connect(lambda _, k=key: self.navigation_requested.emit(k))
            layout.addWidget(btn)

        layout.addStretch(1)