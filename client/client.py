import sys
from PySide6 import QtGui
from PySide6.QtWidgets import (
    QApplication, 
    QMainWindow, 
    QLabel, 
    QPushButton
    )

from constants.constants import WINDOW_TITLE, SWITCH_CHARACTER
from constants.style import WINDOW_WIDTH, WINDOW_HEIGHT


class ApplicationWindow(QMainWindow):
    SAM_PATH = "./assets/sam.jpg"
    FIREFLY_PATH = './assets/firefly.jpg'
    image_path = FIREFLY_PATH
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self._center()

        # 创建QLabel部件
        self.label = QLabel(self)
        self.setCentralWidget(self.label)
        
        # 加载图片
        self.pixmap = QtGui.QPixmap(ApplicationWindow.image_path).scaled(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.label.setPixmap(self.pixmap)
        # 自适应图片大小
        self.label.setScaledContents(True)
        
        self.btn0 = QPushButton(SWITCH_CHARACTER, self)
        self.btn0.clicked.connect(self._switch_character)
        
    # move the application window to the center of the screen
    def _center(self):
        centerPoint = QtGui.QScreen.availableGeometry(QApplication.primaryScreen()).center()
        fg = self.frameGeometry()
        fg.moveCenter(centerPoint)
        self.move(fg.topLeft())
        
    def _switch_character(self):
        print("switch image")
        if ApplicationWindow.image_path == ApplicationWindow.FIREFLY_PATH:
            ApplicationWindow.image_path = ApplicationWindow.SAM_PATH
        else:
            ApplicationWindow.image_path = ApplicationWindow.FIREFLY_PATH
        self.pixmap = QtGui.QPixmap(ApplicationWindow.image_path).scaled(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.label.setPixmap(self.pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ApplicationWindow()
    window.show()
    sys.exit(app.exec())
