#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import uuid
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                              QToolBar, QFileDialog, QMessageBox, QDockWidget, QFrame,
                              QProgressBar, QSplitter, QStatusBar, QLabel, QApplication
                              )
from PySide6.QtCore import Qt, Slot, QSettings, QEvent, QTimer, QThread, Signal
from PySide6.QtGui import QAction, QKeySequence, QColor, QIcon, QPixmap, QPainter

from ui.video_player import VideoPlayer
from ui.timeline import TimelineWidget
from ui.label_panel import LabelPanel

class AutoSaveThread(QThread):
    """后台自动保存线程"""
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.running = True
        
    def run(self):
        while self.running:
            self.sleep(30)  # 每30秒检查一次
            if self.running and self.main_window.current_video_path:
                self.main_window.auto_save_labels()
    
    def stop(self):
        self.running = False
        self.quit()
        self.wait()

class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Video Label Tool - 视频标注工具")
        self.resize(1400, 900)  # 增大默认窗口尺寸
        
        self.settings = QSettings("LabelV", "VideoLabelTool")
        
        # 撤销/重做系统
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo_stack = 50
        
        # 自动保存定时器
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self.auto_save_labels)
        self.auto_save_timer.start(60000)  # 每分钟自动保存
        
        # Create components
        self.video_player = VideoPlayer()
        self.timeline = TimelineWidget()
        self.label_panel = LabelPanel()
        
        # Set up central widget with splitter for better layout
        self.setup_main_layout()
        
        # Connect signals
        self.connect_signals()
        
        # Create toolbar and actions
        self.create_actions()
        self.create_toolbar()
        self.create_status_bar()
        
        # Current project data
        self.current_video_path = None
        self.current_project_path = None
        self.labels = []
        self.has_unsaved_changes = False
        
        # Override the timeline's keyPressEvent with our custom handler
        self.timeline.keyPressEvent = lambda event: self.handle_timeline_key_press(event)
        
        # Initialize UI mode based on the default timeline mode
        self.update_mode(self.timeline.current_mode)
        
        # Install event filter for global shortcuts
        self.installEventFilter(self)
        
        # Force low quality on all interactions
        self.video_player.scrubbing_mode = "low"
        self.video_player.set_scrubbing_quality("low")
        self.video_player.quality_lock = True
        
        # Load window geometry
        self.restore_settings()
        
        # 显示欢迎信息
        self.show_welcome_message()
    
    def setup_main_layout(self):
        """设置主布局使用分割器"""
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(main_splitter)
        
        # Create left panel for video and timeline
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # Add video player to left layout
        left_layout.addWidget(self.video_player, 3)
        
        # Add a separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setLineWidth(1)
        left_layout.addWidget(separator)
        
        # Add timeline to left layout
        left_layout.addWidget(self.timeline, 1)
        
        # Add to splitter
        main_splitter.addWidget(left_widget)
        
        # Create dock widget for label panel
        label_dock = QDockWidget("标签面板", self)
        label_dock.setObjectName("LabelPanelDock")  # Set object name to avoid warnings
        label_dock.setWidget(self.label_panel)
        label_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        label_dock.setMinimumWidth(300)
        self.addDockWidget(Qt.RightDockWidgetArea, label_dock)
        
        # Set splitter proportions
        main_splitter.setSizes([1000, 400])
    
    def connect_signals(self):
        """连接所有信号"""
        self.video_player.position_changed.connect(self.timeline.update_position)
        self.timeline.position_changed.connect(self.video_player.set_position)
        
        self.label_panel.label_added.connect(self.timeline.add_label)
        self.label_panel.label_deleted.connect(self.timeline.remove_label)
        self.label_panel.label_selected.connect(self.timeline.select_label)
        
        # Connect timeline label selection to panel
        self.timeline.label_selected.connect(self.on_timeline_label_selected)
        
        # Connect label playback request to player
        self.timeline.label_playback_requested.connect(self.play_label_segment)
        
        # Connect signals for timeline-label panel synchronization
        self.timeline.label_created.connect(self.on_label_created)
        
        # Connect label name change signal to timeline
        self.label_panel.label_name_changed.connect(self.timeline.update_label_name)
        
        # Connect label color change signal to timeline
        self.label_panel.label_color_changed.connect(self.timeline.update_label_color)
        
        # Add this connection
        self.label_panel.label_template_list.itemClicked.connect(
            self.update_template_selection
        )
        
        # Connect label panel selection to timeline
        self.label_panel.label_selected.connect(self.timeline.select_label)
        
        # 监听数据变化以标记未保存状态
        self.timeline.label_created.connect(self.mark_unsaved_changes)
        self.label_panel.label_updated.connect(self.mark_unsaved_changes)
        self.label_panel.label_deleted.connect(self.mark_unsaved_changes)
    
    def create_actions(self):
        """Create application actions."""
        # File operations
        self.open_video_action = QAction("打开视频 (&O)", self)
        self.open_video_action.setShortcut(QKeySequence.Open)
        self.open_video_action.setStatusTip("打开视频文件进行标注")
        self.open_video_action.triggered.connect(self.open_video)
        
        self.save_labels_action = QAction("保存标签 (&S)", self)
        self.save_labels_action.setShortcut(QKeySequence.Save)
        self.save_labels_action.setStatusTip("保存当前标签到文件")
        self.save_labels_action.setEnabled(False)
        self.save_labels_action.triggered.connect(self.save_labels)
        
        self.save_as_action = QAction("另存为...", self)
        self.save_as_action.setShortcut(QKeySequence.SaveAs)
        self.save_as_action.setStatusTip("将标签保存到指定文件")
        self.save_as_action.setEnabled(False)
        self.save_as_action.triggered.connect(self.save_labels_as)
        
        self.load_labels_action = QAction("加载标签 (&L)", self)
        self.load_labels_action.setShortcut(QKeySequence("Ctrl+L"))
        self.load_labels_action.setStatusTip("从文件加载标签")
        self.load_labels_action.setEnabled(False)
        self.load_labels_action.triggered.connect(self.load_labels)
        
        self.export_labels_action = QAction("导出标签 (&E)", self)
        self.export_labels_action.setShortcut(QKeySequence("Ctrl+E"))
        self.export_labels_action.setStatusTip("将标签导出为JSON或CSV格式")
        self.export_labels_action.setEnabled(False)
        self.export_labels_action.triggered.connect(self.export_labels)
        
        # Edit operations
        self.undo_action = QAction("撤销 (&Z)", self)
        self.undo_action.setShortcut(QKeySequence.Undo)
        self.undo_action.setStatusTip("撤销上一个操作")
        self.undo_action.setEnabled(False)
        self.undo_action.triggered.connect(self.undo)
        
        self.redo_action = QAction("重做 (&Y)", self)
        self.redo_action.setShortcut(QKeySequence.Redo)
        self.redo_action.setStatusTip("重做上一个撤销的操作")
        self.redo_action.setEnabled(False)
        self.redo_action.triggered.connect(self.redo)
        
        # Mode actions
        self.mode_action_choose = QAction("查看模式 (&C)", self)
        self.mode_action_choose.setShortcut(QKeySequence("C"))
        self.mode_action_choose.setStatusTip("切换到查看模式：浏览和播放标签段")
        self.mode_action_choose.setCheckable(True)
        self.mode_action_choose.triggered.connect(lambda: self.timeline.set_mode(self.timeline.CHOOSE_MODE))
        
        self.mode_action_edit = QAction("编辑模式 (&X)", self)
        self.mode_action_edit.setShortcut(QKeySequence("X"))
        self.mode_action_edit.setStatusTip("切换到编辑模式：创建和修改标签")
        self.mode_action_edit.setCheckable(True)
        self.mode_action_edit.triggered.connect(lambda: self.timeline.set_mode(self.timeline.EDIT_MODE))
        
        # Playback controls
        self.play_pause_action = QAction("播放/暂停", self)
        self.play_pause_action.setShortcut(QKeySequence("Space"))
        self.play_pause_action.setStatusTip("播放或暂停视频 (空格键)")
        self.play_pause_action.triggered.connect(self.toggle_playback)
        
        self.jump_backward_action = QAction("后退5秒", self)
        self.jump_backward_action.setShortcut(QKeySequence("Left"))
        self.jump_backward_action.setStatusTip("后退5秒 (左箭头)")
        self.jump_backward_action.triggered.connect(self.jump_backward)
        
        self.jump_forward_action = QAction("前进5秒", self)
        self.jump_forward_action.setShortcut(QKeySequence("Right"))
        self.jump_forward_action.setStatusTip("前进5秒 (右箭头)")
        self.jump_forward_action.triggered.connect(self.jump_forward)
        
        # Help action
        self.help_action = QAction("快捷键帮助 (&H)", self)
        self.help_action.setShortcut(QKeySequence.HelpContents)
        self.help_action.setStatusTip("显示快捷键帮助")
        self.help_action.triggered.connect(self.show_help)
    
    def create_toolbar(self):
        """Create application toolbar."""
        toolbar = QToolBar("主工具栏")
        toolbar.setObjectName("MainToolBar")  # Set object name to avoid warnings
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # File operations
        toolbar.addAction(self.open_video_action)
        toolbar.addSeparator()
        toolbar.addAction(self.save_labels_action)
        toolbar.addAction(self.load_labels_action)
        toolbar.addAction(self.export_labels_action)
        
        toolbar.addSeparator()
        
        # Edit operations
        toolbar.addAction(self.undo_action)
        toolbar.addAction(self.redo_action)
        
        toolbar.addSeparator()
        
        # Mode actions
        toolbar.addAction(self.mode_action_choose)
        toolbar.addAction(self.mode_action_edit)
        
        toolbar.addSeparator()
        
        # Playback controls
        toolbar.addAction(self.play_pause_action)
        toolbar.addAction(self.jump_backward_action)
        toolbar.addAction(self.jump_forward_action)
        
        toolbar.addSeparator()
        
        # Help
        toolbar.addAction(self.help_action)
    
    def create_status_bar(self):
        """创建状态栏"""
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        
        # 主要状态信息
        self.status_label = QLabel("就绪")
        status_bar.addWidget(self.status_label)
        
        # 模式指示器
        self.mode_label = QLabel("查看模式")
        self.mode_label.setStyleSheet("QLabel { color: blue; font-weight: bold; }")
        status_bar.addPermanentWidget(self.mode_label)
        
        # 进度条（用于显示加载进度）
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        status_bar.addPermanentWidget(self.progress_bar)
        
        # 保存状态指示器
        self.save_status_label = QLabel("已保存")
        self.save_status_label.setStyleSheet("QLabel { color: green; }")
        status_bar.addPermanentWidget(self.save_status_label)
    
    def show_welcome_message(self):
        """显示欢迎信息"""
        welcome_msg = (
            "欢迎使用视频标注工具！\n\n"
            "快速开始：\n"
            "1. 按 Ctrl+O 打开视频文件\n"
            "2. 按 X 进入编辑模式\n"
            "3. 右键拖拽创建标签\n"
            "4. 按 C 进入查看模式\n"
            "5. 按 F1 查看完整快捷键列表\n\n"
            "提示：所有操作都有对应的快捷键，让您的标注工作更高效！"
        )
        self.status_label.setText(welcome_msg.split('\n')[0])
    
    def show_help(self):
        """显示快捷键帮助"""
        help_text = """
<h2>视频标注工具 - 快捷键帮助</h2>

<h3>文件操作</h3>
<b>Ctrl+O</b> - 打开视频文件<br>
<b>Ctrl+S</b> - 保存标签<br>
<b>Ctrl+Shift+S</b> - 另存为<br>
<b>Ctrl+L</b> - 加载标签<br>
<b>Ctrl+E</b> - 导出标签<br>

<h3>编辑操作</h3>
<b>Ctrl+Z</b> - 撤销<br>
<b>Ctrl+Y</b> - 重做<br>
<b>Delete</b> - 删除选中的标签<br>

<h3>模式切换</h3>
<b>C</b> - 查看模式（浏览和播放）<br>
<b>X</b> - 编辑模式（创建和修改标签）<br>

<h3>视频控制</h3>
<b>空格</b> - 播放/暂停<br>
<b>←</b> - 后退5秒<br>
<b>→</b> - 前进5秒<br>
<b>↑/↓</b> - 调整播放速度<br>

<h3>标签操作</h3>
<b>右键拖拽</b> - 创建新标签（编辑模式）<br>
<b>左键点击</b> - 选择标签<br>
<b>左键拖拽</b> - 移动标签或调整大小<br>
<b>双击</b> - 播放标签段<br>

<h3>时间轴操作</h3>
<b>滚轮</b> - 缩放时间轴<br>
<b>鼠标中键拖拽</b> - 平移时间轴<br>

<p><i>提示：工具栏和菜单中的每个功能都有对应的快捷键，让您的标注工作更加高效！</i></p>
        """
        
        help_dialog = QMessageBox()
        help_dialog.setWindowTitle("快捷键帮助")
        help_dialog.setTextFormat(Qt.RichText)
        help_dialog.setText(help_text)
        help_dialog.setStandardButtons(QMessageBox.Ok)
        help_dialog.exec()
    
    def mark_unsaved_changes(self):
        """标记有未保存的更改"""
        if not self.has_unsaved_changes:
            self.has_unsaved_changes = True
            self.save_status_label.setText("未保存")
            self.save_status_label.setStyleSheet("QLabel { color: red; }")
            
            # 更新窗口标题
            title = self.windowTitle()
            if not title.endswith("*"):
                self.setWindowTitle(title + "*")
    
    def mark_saved(self):
        """标记为已保存状态"""
        self.has_unsaved_changes = False
        self.save_status_label.setText("已保存")
        self.save_status_label.setStyleSheet("QLabel { color: green; }")
        
        # 更新窗口标题
        title = self.windowTitle()
        if title.endswith("*"):
            self.setWindowTitle(title[:-1])
    
    def auto_save_labels(self):
        """自动保存标签"""
        if self.current_video_path and self.has_unsaved_changes:
            try:
                self.save_labels(silent=True)
                self.status_label.setText("自动保存完成")
                QTimer.singleShot(3000, lambda: self.status_label.setText("就绪"))
            except Exception as e:
                print(f"自动保存失败: {e}")
    
    def save_labels_as(self):
        """另存为标签文件"""
        if not self.current_video_path:
            QMessageBox.warning(self, "警告", "请先加载视频文件")
            return
        
        last_dir = self.settings.value("last_export_dir", "")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "另存为标签文件", last_dir,
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            self.settings.setValue("last_export_dir", os.path.dirname(file_path))
            if not file_path.endswith(".json"):
                file_path += ".json"
            
            try:
                self.save_labels_to_file(file_path)
                self.mark_saved()
                QMessageBox.information(self, "成功", f"标签已保存到 {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
    
    def undo(self):
        """撤销操作"""
        if self.undo_stack:
            current_state = self.get_current_state()
            self.redo_stack.append(current_state)
            
            previous_state = self.undo_stack.pop()
            self.restore_state(previous_state)
            
            self.update_undo_redo_actions()
            self.status_label.setText("已撤销操作")
    
    def redo(self):
        """重做操作"""
        if self.redo_stack:
            current_state = self.get_current_state()
            self.undo_stack.append(current_state)
            
            next_state = self.redo_stack.pop()
            self.restore_state(next_state)
            
            self.update_undo_redo_actions()
            self.status_label.setText("已重做操作")
    
    def get_current_state(self):
        """获取当前状态用于撤销/重做"""
        return {
            'labels': [label.to_dict() for label in self.timeline.labels],
            'current_frame': self.timeline.current_frame
        }
    
    def restore_state(self, state):
        """恢复到指定状态"""
        # 清除当前标签
        self.timeline.clear()
        self.label_panel.clear()
        
        # 恢复标签
        for label_data in state['labels']:
            self.timeline.add_label(label_data)
            self.label_panel.add_label_to_list(label_data)
        
        # 恢复当前帧
        if 'current_frame' in state:
            self.timeline.current_frame = state['current_frame']
            self.video_player.set_position(state['current_frame'])
        
        self.timeline.update()
    
    def save_state_for_undo(self):
        """保存当前状态到撤销栈"""
        current_state = self.get_current_state()
        self.undo_stack.append(current_state)
        
        # 限制撤销栈大小
        if len(self.undo_stack) > self.max_undo_stack:
            self.undo_stack.pop(0)
        
        # 清空重做栈
        self.redo_stack.clear()
        
        self.update_undo_redo_actions()
    
    def update_undo_redo_actions(self):
        """更新撤销/重做按钮状态"""
        self.undo_action.setEnabled(bool(self.undo_stack))
        self.redo_action.setEnabled(bool(self.redo_stack))
    
    def toggle_playback(self):
        """切换播放/暂停状态"""
        if hasattr(self.video_player, 'toggle_play'):
            self.video_player.toggle_play()
    
    def jump_backward(self):
        """后退5秒"""
        if hasattr(self.video_player, 'fast_backward'):
            self.video_player.fast_backward()
    
    def jump_forward(self):
        """前进5秒"""
        if hasattr(self.video_player, 'fast_forward'):
            self.video_player.fast_forward()
    
    def restore_settings(self):
        """恢复窗口设置"""
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)
    
    def closeEvent(self, event):
        """关闭事件处理"""
        if self.has_unsaved_changes:
            reply = QMessageBox.question(
                self, "未保存的更改",
                "有未保存的更改，是否要保存？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Save:
                self.save_labels()
                if self.has_unsaved_changes:  # 保存失败
                    event.ignore()
                    return
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
        
        # 保存窗口设置
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        
        # 停止自动保存定时器
        self.auto_save_timer.stop()
        
        event.accept()

    def update_mode(self, mode):
        """Update UI based on timeline mode."""
        if mode == self.timeline.CHOOSE_MODE:
            # View mode - disable label editing
            self.mode_action_choose.setChecked(True)
            self.mode_action_edit.setChecked(False)
            
            # Update mode indicator
            self.mode_label.setText("查看模式")
            self.mode_label.setStyleSheet("QLabel { color: blue; font-weight: bold; }")
            
            # Set status message
            self.status_label.setText("查看模式：点击标签播放片段，使用 X 键切换到编辑模式")
        else:  # EDIT_MODE
            # Edit mode - enable label editing
            self.mode_action_choose.setChecked(False)
            self.mode_action_edit.setChecked(True)
            
            # Update mode indicator
            self.mode_label.setText("编辑模式")
            self.mode_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")
            
            # Set status message
            self.status_label.setText("编辑模式：右键拖拽创建标签，使用 C 键切换到查看模式")

    def handle_timeline_key_press(self, event):
        """Handle timeline widget key presses."""
        # Store the original keyPressEvent 
        original_key_press = type(self.timeline).keyPressEvent
        
        # Call the original implementation
        original_key_press(self.timeline, event)
        
        # Then handle mode-specific updates
        if event.key() == Qt.Key_C or event.key() == Qt.Key_X:
            self.update_mode(self.timeline.current_mode)
    
    @Slot()
    def open_video(self):
        """Open a video file."""
        # 检查是否有未保存的更改
        if self.has_unsaved_changes:
            reply = QMessageBox.question(
                self, "未保存的更改",
                "当前有未保存的更改，是否要保存？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Save:
                self.save_labels()
                if self.has_unsaved_changes:  # 保存失败
                    return
            elif reply == QMessageBox.Cancel:
                return
        
        last_dir = self.settings.value("last_video_dir", "")
        video_path, _ = QFileDialog.getOpenFileName(
            self, "打开视频文件", last_dir,
            "视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm);;所有文件 (*)"
        )
        
        if video_path:
            # 显示加载进度
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
            self.status_label.setText("正在加载视频...")
            
            # 处理UI事件以显示进度条
            QApplication.processEvents()
            
            self.settings.setValue("last_video_dir", os.path.dirname(video_path))
            self.current_video_path = video_path
            self.setWindowTitle(f"Video Label Tool - 视频标注工具 - {os.path.basename(video_path)}")
            
            # Load video
            success = self.video_player.load_video(video_path)
            
            # 隐藏进度条
            self.progress_bar.setVisible(False)
            
            if success:
                # 保存当前状态用于撤销
                self.save_state_for_undo()
                
                self.timeline.clear()
                self.label_panel.clear()
                
                # Explicitly set the frame count
                self.timeline.set_frame_count(self.video_player.frame_count)
                
                # Ensure timeline gets updated
                self.timeline.update()
                
                # Enable label operations
                self.save_labels_action.setEnabled(True)
                self.save_as_action.setEnabled(True)
                self.load_labels_action.setEnabled(True)
                self.export_labels_action.setEnabled(True)
                
                # 重置保存状态
                self.mark_saved()
                
                # Try to auto-load matching labels file
                json_path = os.path.splitext(video_path)[0] + ".json"
                if os.path.exists(json_path):
                    response = QMessageBox.question(self, "加载标签",
                        f"发现该视频的标签文件，是否加载？",
                        QMessageBox.Yes | QMessageBox.No)
                    if response == QMessageBox.Yes:
                        self.load_labels()
                
                self.status_label.setText(f"视频加载成功：{os.path.basename(video_path)}")
                QTimer.singleShot(3000, lambda: self.status_label.setText("就绪"))
            else:
                QMessageBox.critical(self, "错误", "无法打开视频文件，请检查文件格式和完整性。")
    
    @Slot()
    def export_labels(self):
        """Export labels to a JSON file."""
        if not self.current_video_path:
            QMessageBox.warning(self, "警告", "请先加载视频文件")
            return
            
        if not self.timeline.labels:
            QMessageBox.information(self, "提示", "没有可导出的标签")
            return
            
        last_dir = self.settings.value("last_export_dir", "")
        export_path, _ = QFileDialog.getSaveFileName(
            self, "导出标签", last_dir,
            "JSON 文件 (*.json);;CSV 文件 (*.csv);;所有文件 (*)"
        )
        
        if not export_path:
            return
        
        # 显示进度
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("正在导出标签...")
        QApplication.processEvents()
        
        try:
            # Determine export format based on extension
            if export_path.endswith(".csv"):
                self.export_labels_csv(export_path)
            else:
                if not export_path.endswith(".json"):
                    export_path += ".json"
                self.export_labels_json(export_path)
                
            self.settings.setValue("last_export_dir", os.path.dirname(export_path))
            self.status_label.setText(f"标签已导出到：{export_path}")
            QTimer.singleShot(3000, lambda: self.status_label.setText("就绪"))
            
        finally:
            self.progress_bar.setVisible(False)

    def export_labels_json(self, export_path):
        """Export labels to a JSON file."""
        # Get current video FPS
        fps = self.video_player.fps if hasattr(self.video_player, 'fps') else 30.0
        
        # Collect label data with timestamps
        labels = self.timeline.get_labels_for_export(fps)
        
        # Add metadata
        export_data = {
            "video_file": os.path.basename(self.current_video_path),
            "video_path": self.current_video_path,
            "fps": fps,
            "total_frames": self.video_player.frame_count,
            "duration": self.video_player.format_time(self.video_player.duration_sec),
            "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "labels": labels
        }
        
        # Export to file
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "成功", "标签导出成功！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")

    def export_labels_csv(self, export_path):
        """Export labels to a CSV file."""
        # Get current video FPS
        fps = self.video_player.fps if hasattr(self.video_player, 'fps') else 30.0
        
        # Collect label data with timestamps
        labels = self.timeline.get_labels_for_export(fps)
        
        try:
            with open(export_path, 'w', newline='', encoding='utf-8-sig') as f:  # Use UTF-8 with BOM for better Excel compatibility
                import csv
                fieldnames = ["id", "category", "name", "start_frame", "end_frame", 
                             "start_time", "end_time", "duration", "description"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for label in labels:
                    # Ensure all string fields are properly encoded
                    encoded_label = {}
                    for key, value in label.items():
                        if isinstance(value, str):
                            encoded_label[key] = value
                        else:
                            encoded_label[key] = value
                    writer.writerow(encoded_label)
            QMessageBox.information(self, "成功", "标签成功导出为CSV格式！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"CSV导出失败: {str(e)}")

    def play_label_segment(self, start_frame, end_frame):
        """Play a specific video segment with improved state management."""
        # Stop any current playback first
        if self.video_player.is_playing():
            self.video_player.pause()
        
        # Ensure video player is properly set to low quality mode before playback
        if hasattr(self.video_player, 'set_scrubbing_quality'):
            self.video_player.set_scrubbing_quality("low")
        
        # Force the video player's scrubbing mode directly to ensure it takes effect
        self.video_player.scrubbing_mode = "low"
        
        # Reset any active scrubbing state that might interfere with playback
        self.video_player.scrubbing_active = False
        
        # Additional logging for debugging
        self.video_player.logger.info(f"Playing label segment from {start_frame} to {end_frame}")
        
        # First seek to the start frame before setting playback range
        self.video_player.set_position(start_frame)
        
        # Set playback state
        self.video_player.playback_state['segment_mode'] = True
        self.video_player.playback_state['last_playback_mode'] = 'segment'
        
        # Then set playback range
        self.video_player.set_playback_range(start_frame, end_frame)
        
        # Make sure playback speed is properly applied
        current_speed = self.video_player.playback_speed
        if current_speed < 0.5:  # If speed is too slow, reset it
            self.video_player.set_playback_speed("1.0x")
        
        # Allow a small delay for UI to update before starting playback
        QTimer.singleShot(50, self.video_player.play)
        
        # Update status bar to confirm low quality mode
        self.statusBar().showMessage(f"Playing segment in fast mode (frames {start_frame} to {end_frame})", 2000)

    def on_label_created(self, label_data):
        """Handle when a label is created in the timeline."""
        # Add the label to the label panel
        self.label_panel.add_label_to_list(label_data)

    def on_timeline_label_selected(self, label_id):
        """Handle label selection in timeline by updating label panel."""
        # Find the label data
        for label in self.timeline.labels:
            if label.id == label_id:
                # Update the label panel with this data
                self.label_panel.update_label_data(label.to_dict())
                
                # Also select the corresponding item in the label list
                for i in range(self.label_panel.label_list.count()):
                    item = self.label_panel.label_list.item(i)
                    if item.data(Qt.UserRole) == label_id:
                        self.label_panel.label_list.setCurrentItem(item)
                        break
                break

    def eventFilter(self, obj, event):
        """Global event filter to handle keyboard shortcuts."""
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Space:
                # Toggle play/pause regardless of focus
                if self.video_player.playing:
                    self.video_player.toggle_play()
                    return True  # Event handled
                else:
                    self.video_player.toggle_play()
                    return True  # Event handled
        
        # Pass event to standard event processing
        return super().eventFilter(obj, event) 

    def update_template_selection(self, item):
        # Simply update the timeline without calling viewport()
        self.timeline.update() 

    def on_timeline_position_changed(self, frame):
        """Handle timeline position changes with state preservation."""
        # Get current playing state before changing position
        was_playing = self.video_player.playback_state['playing']
        
        # Update position
        self.video_player.set_position(frame)
        
        # Optionally resume playback if it was playing before
        if was_playing and self.video_player.current_mode == self.video_player.CHOOSE_MODE:
            self.video_player.playback_state['continuous_mode'] = True
            QTimer.singleShot(50, self.video_player.play) 

    def save_labels(self, silent=False):
        """Save labels to JSON file with same name as video."""
        if not self.current_video_path:
            if not silent:
                QMessageBox.warning(self, "警告", "请先加载视频文件")
            return
        
        # Generate JSON path with same name as video
        video_path = self.current_video_path
        json_path = os.path.splitext(video_path)[0] + ".json"
        
        try:
            self.save_labels_to_file(json_path)
            if not silent:
                self.status_label.setText(f"标签已保存到 {json_path}")
                QTimer.singleShot(3000, lambda: self.status_label.setText("就绪"))
        except Exception as e:
            if not silent:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
            raise e
    
    def save_labels_to_file(self, file_path):
        """Save labels to specified file path."""
        # Get current video FPS
        fps = self.video_player.fps if hasattr(self.video_player, 'fps') else 30.0
        
        # Collect label data with timestamps
        labels = self.timeline.get_labels_for_export(fps)
        
        # Add metadata for verification
        export_data = {
            "video_file": os.path.basename(self.current_video_path),
            "video_path": self.current_video_path,
            "fps": fps,
            "total_frames": self.video_player.frame_count,
            "duration": self.video_player.format_time(self.video_player.duration_sec),
            "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "labels": labels
        }
        
        # Export to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.mark_saved()

    def load_labels(self):
        """Load labels from JSON file with same name as video."""
        if not self.current_video_path:
            QMessageBox.warning(self, "警告", "请先加载视频文件")
            return
        
        # Generate JSON path with same name as video
        video_path = self.current_video_path
        json_path = os.path.splitext(video_path)[0] + ".json"
        
        # Check if file exists
        if not os.path.exists(json_path):
            QMessageBox.information(self, "信息", "未找到该视频的标签文件")
            return
        
        try:
            # Load JSON data
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Verify metadata matches
            video_basename = os.path.basename(self.current_video_path)
            if data.get("video_file") != video_basename:
                QMessageBox.warning(self, "警告", 
                                 f"标签文件与视频不匹配：期望 {video_basename}，实际 {data.get('video_file')}")
                return
            
            # Check frame count to ensure video hasn't changed
            if data.get("total_frames") != self.video_player.frame_count:
                response = QMessageBox.question(self, "警告", 
                    "帧数不匹配，视频文件可能已更改。是否仍要加载标签？",
                    QMessageBox.Yes | QMessageBox.No)
                if response == QMessageBox.No:
                    return
            
            # Clear existing labels properly
            self.timeline.clear()
            self.timeline.set_frame_count(self.video_player.frame_count)  # Restore frame count
            self.label_panel.clear_editor()
            
            # Process and add each label
            labels_loaded = 0
            for label_data in data.get("labels", []):
                # Get order and category
                order = label_data.get("order", 0)
                category = label_data.get("category", "default")
                
                # Format name as "order. category"
                formatted_name = f"{order}. {category}" if order > 0 else category
                
                # Create internal label with proper fields and default color
                internal_label = {
                    "id": label_data.get("id", str(uuid.uuid4())),
                    "text": formatted_name,
                    "name": formatted_name,
                    "start_frame": label_data.get("start_frame", 0),
                    "end_frame": label_data.get("end_frame", 0),
                    "category": category,
                    "description": label_data.get("description", ""),
                    # No color specified - will use default or category-based
                }
                
                # Add to timeline
                success = self.timeline.add_label(internal_label)
                if success:
                    # Also add to label panel list
                    self.label_panel.add_label_to_list(internal_label)
                    labels_loaded += 1
            
            # Explicitly update the UI after loading
            self.timeline.update()
            
            self.status_label.setText(f"已从 {json_path} 加载 {labels_loaded} 个标签")
            QTimer.singleShot(3000, lambda: self.status_label.setText("就绪"))
            self.mark_saved() # Mark as saved after successful load
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            QMessageBox.critical(self, "错误", f"加载标签失败: {str(e)}\n\n详细信息: {error_details}") 