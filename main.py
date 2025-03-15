#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PySide6.QtCore import Qt, Slot

from ui.main_window import MainWindow

def main():
    """Main function to start the application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Video Label Tool")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 