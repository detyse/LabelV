#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
from PySide6.QtWidgets import QApplication, QSplashScreen
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QPainter, QFont, QColor

from ui.main_window import MainWindow

def create_splash_screen():
    """创建启动画面"""
    # 创建简单的启动画面
    pixmap = QPixmap(400, 300)
    pixmap.fill(QColor(45, 52, 54))
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # 绘制标题
    font = QFont()
    font.setPointSize(24)
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QColor(255, 255, 255))
    painter.drawText(pixmap.rect(), Qt.AlignCenter, "🎬\n视频标注工具\nVideo Label Tool")
    
    # 绘制版本信息
    font.setPointSize(12)
    font.setBold(False)
    painter.setFont(font)
    painter.setPen(QColor(200, 200, 200))
    painter.drawText(pixmap.rect().adjusted(0, 80, 0, 0), Qt.AlignCenter, "正在启动...")
    
    painter.end()
    return pixmap

def main():
    """Main function to start the application."""
    # 设置应用程序属性
    app = QApplication(sys.argv)
    app.setApplicationName("Video Label Tool")
    app.setApplicationDisplayName("视频标注工具")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("LabelV")
    app.setOrganizationDomain("labelv.com")
    
    # 设置应用程序样式
    app.setStyle("Fusion")  # 使用 Fusion 样式以获得更好的跨平台外观
    
    # 设置默认字体以支持中文
    from PySide6.QtGui import QFont
    default_font = QFont()
    default_font.setFamily("Microsoft YaHei, SimHei, Arial Unicode MS, sans-serif")
    default_font.setPointSize(9)
    app.setFont(default_font)
    
    # 创建并显示启动画面
    splash_pixmap = create_splash_screen()
    splash = QSplashScreen(splash_pixmap)
    splash.show()
    
    # 处理启动画面消息
    splash.showMessage("正在初始化界面...", Qt.AlignBottom | Qt.AlignCenter, QColor(255, 255, 255))
    app.processEvents()
    
    # 模拟加载时间
    QTimer.singleShot(1000, lambda: splash.showMessage("正在加载组件...", Qt.AlignBottom | Qt.AlignCenter, QColor(255, 255, 255)))
    QTimer.singleShot(1500, lambda: splash.showMessage("启动完成！", Qt.AlignBottom | Qt.AlignCenter, QColor(255, 255, 255)))
    
    # 创建主窗口变量
    window = None
    
    # 创建主窗口
    def show_main_window():
        nonlocal window
        splash.close()
        window = MainWindow()
        window.show()
    
    # 延迟显示主窗口
    QTimer.singleShot(2000, show_main_window)
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()