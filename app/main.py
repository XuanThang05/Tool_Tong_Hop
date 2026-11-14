import sys
import os
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    
    # Optionally, apply global stylesheet here
    # with open("app/ui/styles.qss", "r") as f:
    #     app.setStyleSheet(f.read())
        
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    # Ensure current working directory is the project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
