import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from ui.mainwindow import MainWindow
import os

def main():
    app = QApplication(sys.argv)

    icon_path = os.path.join(os.path.dirname(__file__), "assets", "horusicon.ico")
    app.setWindowIcon(QIcon(icon_path))


    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
