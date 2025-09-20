#!/usr/bin/env python
# -*- coding: utf-8 -*-

import uuid
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                              QLabel, QLineEdit, QListWidget, QListWidgetItem,
                              QColorDialog, QComboBox, QTextEdit, QFormLayout,
                              QGroupBox, QSplitter, QFrame, QMenu, QPlainTextEdit,
                              QAbstractItemView, QToolButton, QScrollArea, QSpacerItem,
                              QSizePolicy)
from PySide6.QtCore import Qt, Signal, Slot, QSize, QTimer
from PySide6.QtGui import QColor, QIcon, QPixmap, QBrush, QPainter, QFont

class ColorButton(QPushButton):
    """Button that displays and selects a color."""
    
    colorChanged = Signal(QColor)
    
    def __init__(self, color=None, parent=None):
        super().__init__(parent)
        self._color = color or QColor(255, 165, 0)  # Default orange
        self.setFixedSize(30, 30)  # Slightly larger for better visibility
        self.setToolTip("点击选择颜色")
        self.update_color()
        self.clicked.connect(self.choose_color)
    
    def color(self):
        """Get the current color."""
        return self._color
    
    def setColor(self, color):
        """Set the current color."""
        if self._color != color:
            self._color = color
            self.update_color()
            self.colorChanged.emit(color)
    
    def update_color(self):
        """Update the button's appearance based on the selected color."""
        pixmap = QPixmap(self.size())
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setBrush(QBrush(self._color))
        painter.setPen(Qt.black)
        painter.drawRoundedRect(2, 2, self.width() - 4, self.height() - 4, 5, 5)
        
        # Add a small highlight for better visibility
        highlight = QColor(255, 255, 255, 80)
        painter.setBrush(QBrush(highlight))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(4, 4, self.width() - 20, self.height() - 20, 3, 3)
        
        painter.end()
        
        self.setIcon(QIcon(pixmap))
    
    def choose_color(self):
        """Open color dialog to select a new color."""
        color = QColorDialog.getColor(self._color, self, "选择颜色", 
                                     QColorDialog.ShowAlphaChannel)
        if color.isValid():
            self.setColor(color)


class LabelPanel(QWidget):
    """Panel for managing video labels."""
    
    # Signals
    label_added = Signal(dict)  # New label data
    label_deleted = Signal(str)  # Label ID
    label_updated = Signal(dict)  # Updated label data
    label_selected = Signal(str)  # Label ID
    label_name_changed = Signal(str, str)  # old_id, new_name
    label_color_changed = Signal(str, object)  # label_id, QColor
    label_category_changed = Signal(str, str)  # label_id, category key
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Label categories with Chinese names
        self.categories = {
            "default": "默认",
            "action": "动作",
            "object": "物体", 
            "person": "人物",
            "scene": "场景",
            "text": "文字"
        }
        
        # Color palette for different categories
        self.category_colors = {
            "default": QColor(255, 165, 0, 180),   # Orange
            "action": QColor(255, 99, 71, 180),    # Tomato
            "object": QColor(60, 179, 113, 180),   # Medium Sea Green
            "person": QColor(106, 90, 205, 180),   # Slate Blue
            "scene": QColor(255, 20, 147, 180),    # Deep Pink
            "text": QColor(32, 178, 170, 180)      # Light Sea Green
        }
        
        self.selected_category_key = "default"
        self._category_change_in_progress = False

        # Initialize label properties
        self.current_label_id = None
        self.label_index = "1"  # Default label index
        
        # Settings for persistent templates (must be before setup_ui)
        from PySide6.QtCore import QSettings
        self.settings = QSettings("LabelV", "VideoLabelTool")
        
        # Set up UI
        self.setup_ui()
        
        if hasattr(self, "category_combo"):
            self.selected_category_key = self.category_combo.currentData() or "default"
        # Set up auto-save timer for real-time updates
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.on_label_property_changed)
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Create header with title and info
        header_layout = QHBoxLayout()
        title_label = QLabel("标签管理")
        title_label.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; color: #2c3e50; }")
        header_layout.addWidget(title_label)
        
        info_label = QLabel("📋")
        info_label.setToolTip("管理视频标签：创建、编辑和删除标签")
        header_layout.addWidget(info_label)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Create scrollable content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(8)
        
        # Create list view for labels with improved styling
        self.setup_label_list(content_layout)
        
        # Create label editor section
        self.setup_label_editor(content_layout)
        
        # Create label templates section
        self.setup_templates_section(content_layout)
        
        # Create action buttons
        self.setup_action_buttons(content_layout)
        
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)
    
    def setup_label_list(self, layout):
        """设置标签列表"""
        list_group = QGroupBox("标签列表")
        list_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        list_layout = QVBoxLayout(list_group)
        
        # Add label count info
        self.label_count_label = QLabel("共 0 个标签")
        self.label_count_label.setStyleSheet("QLabel { color: #7f8c8d; font-size: 12px; }")
        list_layout.addWidget(self.label_count_label)
        
        # Create list widget
        self.label_list = QListWidget()
        self.label_list.setSelectionMode(QListWidget.SingleSelection)
        self.label_list.setAlternatingRowColors(True)
        
        # Set font for Chinese support
        font = QFont()
        font.setFamily("Microsoft YaHei, SimHei, Arial Unicode MS, sans-serif")
        font.setPointSize(9)
        self.label_list.setFont(font)
        
        self.label_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: #ffffff;
                selection-background-color: #3498db;
                selection-color: white;
                font-family: "Microsoft YaHei", "SimHei", "Arial Unicode MS", sans-serif;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #ecf0f1;
            }
            QListWidget::item:hover {
                background-color: #f8f9fa;
            }
            QListWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
        """)
        self.label_list.currentItemChanged.connect(self.on_label_selected)
        self.label_list.setMinimumHeight(150)
        self.label_list.setToolTip("双击标签播放对应片段")
        list_layout.addWidget(self.label_list)
        
        layout.addWidget(list_group)
    
    def setup_label_editor(self, layout):
        """设置标签编辑器"""
        editor_group = QGroupBox("标签编辑器")
        editor_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        editor_layout = QFormLayout()
        editor_layout.setSpacing(10)
        
        # Label name edit
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("输入标签名称...")
        self.name_edit.setToolTip("标签的具体名称或描述")
        
        # Set font for Chinese support
        name_font = QFont()
        name_font.setFamily("Microsoft YaHei, SimHei, Arial Unicode MS, sans-serif")
        name_font.setPointSize(10)
        self.name_edit.setFont(name_font)
        
        self.name_edit.textChanged.connect(self.delayed_update)
        editor_layout.addRow("名称:", self.name_edit)
        
        # Color selector row
        color_layout = QHBoxLayout()
        self.color_button = ColorButton()
        self.color_button.colorChanged.connect(self.delayed_update)
        color_layout.addWidget(self.color_button)
        
        # Color presets
        preset_colors = [
            QColor(255, 99, 71, 180),    # Tomato
            QColor(60, 179, 113, 180),   # Medium Sea Green
            QColor(106, 90, 205, 180),   # Slate Blue
            QColor(255, 20, 147, 180),   # Deep Pink
            QColor(255, 165, 0, 180),    # Orange
            QColor(32, 178, 170, 180)    # Light Sea Green
        ]
        
        for color in preset_colors:
            preset_btn = QPushButton()
            preset_btn.setFixedSize(20, 20)
            preset_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: rgba({color.red()}, {color.green()}, {color.blue()}, {color.alpha()});
                    border: 1px solid #000;
                    border-radius: 10px;
                }}
                QPushButton:hover {{
                    border: 2px solid #3498db;
                }}
            """)
            preset_btn.setToolTip(f"使用预设颜色")
            preset_btn.clicked.connect(lambda checked, c=color: self.color_button.setColor(c))
            color_layout.addWidget(preset_btn)
        
        color_layout.addStretch()
        editor_layout.addRow("颜色:", color_layout)
        
        # Description text edit
        self.description_edit = QPlainTextEdit()
        self.description_edit.setPlaceholderText("输入标签描述...")
        self.description_edit.setMaximumHeight(80)
        self.description_edit.setToolTip("标签的详细描述信息")
        
        # Set font for Chinese support
        desc_font = QFont()
        desc_font.setFamily("Microsoft YaHei, SimHei, Arial Unicode MS, sans-serif")
        desc_font.setPointSize(9)
        self.description_edit.setFont(desc_font)
        
        self.description_edit.textChanged.connect(self.delayed_update)
        editor_layout.addRow("描述:", self.description_edit)
        
        # Frame range display with enhanced styling
        frame_info_layout = QHBoxLayout()
        
        self.start_frame_label = QLabel("0 (00:00:00)")
        self.start_frame_label.setStyleSheet("""
            QLabel {
                background-color: #ecf0f1;
                padding: 4px 8px;
                border-radius: 4px;
                font-family: monospace;
            }
        """)
        
        separator_label = QLabel("→")
        separator_label.setAlignment(Qt.AlignCenter)
        separator_label.setStyleSheet("QLabel { font-weight: bold; color: #3498db; }")
        
        self.end_frame_label = QLabel("0 (00:00:00)")
        self.end_frame_label.setStyleSheet("""
            QLabel {
                background-color: #ecf0f1;
                padding: 4px 8px;
                border-radius: 4px;
                font-family: monospace;
            }
        """)
        
        frame_info_layout.addWidget(self.start_frame_label)
        frame_info_layout.addWidget(separator_label)
        frame_info_layout.addWidget(self.end_frame_label)
        frame_info_layout.addStretch()
        
        editor_layout.addRow("帧范围:", frame_info_layout)
        
        # Duration display
        self.duration_label = QLabel("时长: 0.0秒")
        self.duration_label.setStyleSheet("QLabel { color: #7f8c8d; font-style: italic; }")
        editor_layout.addRow("", self.duration_label)
        
        # Set the layout for the editor group
        editor_group.setLayout(editor_layout)
        layout.addWidget(editor_group)
    
    def setup_templates_section(self, layout):
        """设置模板选择区域"""
        templates_group = QGroupBox("标签模板")
        templates_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        templates_layout = QVBoxLayout(templates_group)
        
        # Template info
        info_label = QLabel("💡 选择模板快速创建标签")
        info_label.setStyleSheet("QLabel { color: #7f8c8d; font-size: 12px; }")
        templates_layout.addWidget(info_label)
        
        # Label list widget
        self.label_template_list = QListWidget()
        self.label_template_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.label_template_list.setMaximumHeight(100)
        
        # Set font for Chinese support
        template_font = QFont()
        template_font.setFamily("Microsoft YaHei, SimHei, Arial Unicode MS, sans-serif")
        template_font.setPointSize(9)
        self.label_template_list.setFont(template_font)
        
        self.label_template_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: #f8f9fa;
                font-family: "Microsoft YaHei", "SimHei", "Arial Unicode MS", sans-serif;
            }
            QListWidget::item {
                padding: 4px 8px;
                border-bottom: 1px solid #ecf0f1;
            }
            QListWidget::item:hover {
                background-color: #e9ecef;
            }
            QListWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
        """)
        self.label_template_list.itemClicked.connect(self.on_template_selected)
        self.label_template_list.itemSelectionChanged.connect(self.on_template_selection_changed)
        self.label_template_list.setToolTip("点击选择模板，双击应用到当前标签")
        templates_layout.addWidget(self.label_template_list)
        
        # Template management buttons
        template_buttons_layout = QHBoxLayout()
        
        self.add_template_button = QPushButton("➕ 添加")
        self.add_template_button.setToolTip("将当前标签添加为模板")
        self.add_template_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        self.add_template_button.clicked.connect(self.add_current_to_templates)
        
        self.delete_template_button = QPushButton("🗑️ 删除")
        self.delete_template_button.setToolTip("删除选中的模板")
        self.delete_template_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        self.delete_template_button.clicked.connect(self.delete_selected_template)
        self.delete_template_button.setEnabled(False)  # Initially disabled
        
        self.clear_template_button = QPushButton("📝 编辑")
        self.clear_template_button.setToolTip("编辑模板名称")
        self.clear_template_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        self.clear_template_button.clicked.connect(self.edit_selected_template)
        self.clear_template_button.setEnabled(False)  # Initially disabled
        
        template_buttons_layout.addWidget(self.add_template_button)
        template_buttons_layout.addWidget(self.delete_template_button)
        template_buttons_layout.addWidget(self.clear_template_button)
        templates_layout.addLayout(template_buttons_layout)
        
        # Add the templates group to the main layout
        layout.addWidget(templates_group)
        
        # Initialize with default templates
        self.initialize_templates()
    
    def setup_action_buttons(self, layout):
        """设置操作按钮"""
        button_layout = QHBoxLayout()
        
        # Remove button with improved styling
        self.remove_button = QPushButton("🗑️ 删除标签")
        self.remove_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        self.remove_button.setToolTip("删除当前选中的标签 (Del键)")
        self.remove_button.clicked.connect(self.on_remove_label)
        
        # Duplicate button
        self.duplicate_button = QPushButton("📋 复制标签")
        self.duplicate_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        self.duplicate_button.setToolTip("复制当前标签设置")
        self.duplicate_button.clicked.connect(self.duplicate_current_label)
        
        button_layout.addWidget(self.duplicate_button)
        button_layout.addWidget(self.remove_button)
        
        layout.addLayout(button_layout)
        
        # Initially disable editor
        self.set_editor_enabled(False)
    
    def initialize_templates(self):
        """初始化鼠标行为模板"""
        # Load templates from settings or use defaults
        saved_templates = self.load_templates_from_settings()
        
        if saved_templates:
            default_templates = saved_templates
        else:
            # Default mouse behavior templates
            default_templates = [
                "grooming", "standing", "walking", "running", "rearing",
                "sniffing", "drinking", "eating", "sleeping", "exploring"
            ]
        
        for template in default_templates:
            self.label_template_list.addItem(template)
        
        # Set default selection
        self.selected_template = default_templates[0] if default_templates else "grooming"
        if self.label_template_list.count() > 0:
            self.label_template_list.setCurrentRow(0)
    
    def load_templates_from_settings(self):
        """从设置中加载模板"""
        templates = self.settings.value("label_templates", [])
        if isinstance(templates, str):
            templates = [templates]  # Handle single template case
        return templates if templates else []
    
    def save_templates_to_settings(self):
        """保存模板到设置"""
        templates = []
        for i in range(self.label_template_list.count()):
            templates.append(self.label_template_list.item(i).text())
        self.settings.setValue("label_templates", templates)
    
    def auto_add_label_to_templates(self, label_name):
        """自动将新标签添加到模板"""
        # Remove number prefix if exists (e.g., "1. grooming" -> "grooming")
        clean_name = label_name
        if ". " in label_name:
            parts = label_name.split(". ", 1)
            if len(parts) > 1:
                clean_name = parts[1]
        
        # Check if template already exists
        existing_templates = []
        for i in range(self.label_template_list.count()):
            existing_templates.append(self.label_template_list.item(i).text())
        
        if clean_name and clean_name not in existing_templates:
            self.label_template_list.addItem(clean_name)
            self.save_templates_to_settings()
    
    def delayed_update(self):
        """延迟更新以避免频繁的信号发射"""
        self.update_timer.start(300)  # 300ms delay
    

    
    def duplicate_current_label(self):
        """复制当前标签"""
        if not self.current_label_id:
            return
        
        # Save current label data as template
        template_name = f"{self.name_edit.text()}_副本"
        if template_name not in [self.label_template_list.item(i).text() 
                                for i in range(self.label_template_list.count())]:
            self.label_template_list.addItem(template_name)
        
        # Select the new template
        for i in range(self.label_template_list.count()):
            if self.label_template_list.item(i).text() == template_name:
                self.label_template_list.setCurrentRow(i)
                break
    
    def update_label_count(self):
        """更新标签数量显示"""
        count = self.label_list.count()
        self.label_count_label.setText(f"共 {count} 个标签")
    
    def set_editor_enabled(self, enabled):
        """Enable or disable the label editor."""
        self.name_edit.setEnabled(enabled)
        self.color_button.setEnabled(enabled)
        self.category_combo.setEnabled(enabled)
        self.description_edit.setEnabled(enabled)
        self.start_frame_label.setEnabled(enabled)
        self.end_frame_label.setEnabled(enabled)
        self.add_template_button.setEnabled(enabled)
        self.clear_template_button.setEnabled(enabled)
        self.remove_button.setEnabled(enabled)
        self.duplicate_button.setEnabled(enabled)
    
    def on_category_changed(self, index):
        """Handle category changes from the combo box."""
        category = self.category_combo.itemData(index)
        if category is None:
            return
        self.selected_category_key = category
        default_color = self.category_colors.get(category)
        self.update_timer.stop()
        self._category_change_in_progress = True
        try:
            if default_color:
                self.color_button.blockSignals(True)
                self.color_button.setColor(default_color)
                self.color_button.blockSignals(False)
            if self.current_label_id:
                self.label_category_changed.emit(self.current_label_id, category)
                self.on_label_property_changed()
        finally:
            self._category_change_in_progress = False
    def clear(self):
        """Clear all labels."""
        self.label_list.clear()
        self.current_label_id = None
        self.set_editor_enabled(False)
        self.remove_button.setEnabled(False)
        self.update_label_count()
    
    @Slot()
    def on_add_label(self):
        """Add a new label."""
        # Create a new label
        label_id = str(uuid.uuid4())
        
        # Get the next number based on existing labels
        count = self.label_list.count() + 1
        
        default_color = self.category_colors.get(self.selected_category_key, QColor(255, 165, 0, 180))
        label_data = {
            "id": label_id,
            "name": f"Label {count}",
            "start_frame": 0,
            "end_frame": 0,
            "color": [default_color.red(), default_color.green(), default_color.blue(), default_color.alpha()],
            "category": self.selected_category_key,
            "color_is_custom": False,
            "description": ""
        }
        
        # Emit signal to add label
        self.label_added.emit(label_data)
        
        # Add to list
        self.add_label_to_list(label_data)
        
        # Select the new item
        for i in range(self.label_list.count()):
            item = self.label_list.item(i)
            if item.data(Qt.UserRole) == label_id:
                self.label_list.setCurrentItem(item)
                break
        self.update_label_count()
    
    @Slot()
    def on_remove_label(self):
        """Remove the selected label."""
        current_item = self.label_list.currentItem()
        if current_item:
            label_id = current_item.data(Qt.UserRole)
            
            # Remove from list
            row = self.label_list.row(current_item)
            self.label_list.takeItem(row)
            
            # Emit signal to remove from timeline
            self.label_deleted.emit(label_id)
            
            # Clear the editor
            self.clear_editor()
            
            # If there are still items in the list, select the previous one
            # or the first one if we removed the first item
            if self.label_list.count() > 0:
                new_row = min(row, self.label_list.count() - 1)
                self.label_list.setCurrentRow(new_row)
            self.update_label_count()
    
    @Slot()
    def on_label_property_changed(self):
        """Handle changes to label properties."""
        if not self.current_label_id:
            return
        
        action_name = self.name_edit.text().strip()
        full_name = f"{self.label_index}. {action_name}" if action_name else f"{self.label_index}."
        category = self.category_combo.currentData() or "default"
        color = self.color_button.color()
        default_color = self.category_colors.get(category)
        if self._category_change_in_progress:
            color_is_custom = False
        else:
            color_is_custom = not (default_color and color == default_color)
        
        label_data = {
            "id": self.current_label_id,
            "name": full_name,
            "color": [color.red(), color.green(), color.blue(), color.alpha()],
            "category": category,
            "color_is_custom": color_is_custom,
            "description": self.description_edit.toPlainText()
        }
        
        for i in range(self.label_list.count()):
            item = self.label_list.item(i)
            if item.data(Qt.UserRole) == self.current_label_id:
                item.setText(full_name)
                pixmap = QPixmap(16, 16)
                pixmap.fill(Qt.transparent)
                painter = QPainter(pixmap)
                painter.setBrush(QBrush(color))
                painter.setPen(Qt.black)
                painter.drawRect(0, 0, 15, 15)
                painter.end()
                item.setIcon(QIcon(pixmap))
                item.setData(Qt.UserRole + 1, category)
                break
        
        self.label_updated.emit(label_data)
        if not self._category_change_in_progress:
            self.label_color_changed.emit(self.current_label_id, color)
        
        self.auto_add_label_to_templates(full_name)
        self.label_name_changed.emit(self.current_label_id, full_name)

    @Slot(QListWidgetItem, QListWidgetItem)
    def on_label_selected(self, current, previous):
        """Handle selection of a label from the list."""
        if not current:
            self.current_label_id = None
            self.set_editor_enabled(False)
            self.remove_button.setEnabled(False)
            self.duplicate_button.setEnabled(False)
            return
            
        # Get label ID from item
        label_id = current.data(Qt.UserRole)
        self.current_label_id = label_id
        
        # Enable editor and remove button
        self.set_editor_enabled(True)
        self.remove_button.setEnabled(True)
        self.duplicate_button.setEnabled(True)
        
        # Emit signal for selected label
        self.label_selected.emit(label_id)
        
        # FETCH THE LABEL DATA FROM THE TIMELINE AND UPDATE THE EDITOR
        # Get the parent window to access timeline
        parent = self.parent()
        while parent:
            if hasattr(parent, 'timeline'):
                # Find and update the label data
                for label in parent.timeline.labels:
                    if label.id == label_id:
                        self.update_label_data(label.to_dict())
                        break
                break
            parent = parent.parent()
    
    def add_label_to_list(self, label_data):
        """Add a label to the list widget."""
        # Update the name format when adding new labels
        if label_data["name"].startswith("Label ") or label_data["name"] == "New Label":
            # Apply template naming
            label_data["name"] = self.selected_template
        
        # Check if label already exists in the list
        label_id = label_data.get("id", "")
        for i in range(self.label_list.count()):
            if self.label_list.item(i).data(Qt.UserRole) == label_id:
                # Label already exists, just update it
                self.update_list_item(i, label_data)
                return
        
        # Create new list item
        name = label_data.get("name", "")
        color = label_data.get("color", [255, 165, 0, 180])
        
        item = QListWidgetItem(name)
        item.setData(Qt.UserRole, label_id)
        item.setData(Qt.UserRole + 1, label_data.get("category", "default"))
        
        # Set icon with label color
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setBrush(QBrush(QColor(*color)))
        painter.setPen(Qt.black)
        painter.drawRect(0, 0, 15, 15)
        painter.end()
        
        item.setIcon(QIcon(pixmap))
        
        # Add to list
        self.label_list.addItem(item)
        
        # Select the new item if no item is currently selected
        if not self.label_list.currentItem():
            self.label_list.setCurrentItem(item)
        self.update_label_count()
    
    def update_list_item(self, row, label_data):
        """Update an existing list item with new data."""
        item = self.label_list.item(row)
        if not item:
            return
        
        name = label_data.get("name", "")
        color = label_data.get("color", [255, 165, 0, 180])
        
        item.setText(name)
        item.setData(Qt.UserRole + 1, label_data.get("category", "default"))
        
        # Update icon with new color
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setBrush(QBrush(QColor(*color)))
        painter.setPen(Qt.black)
        painter.drawRect(0, 0, 15, 15)
        painter.end()
        
        item.setIcon(QIcon(pixmap))
    
    def update_label_data(self, label_data):
        """Update the editor with label data."""
        self.current_label_id = label_data["id"]
        category_key = label_data.get("category", "default")
        self.selected_category_key = category_key
        if hasattr(self, "category_combo"):
            self.category_combo.blockSignals(True)
            combo_index = self.category_combo.findData(category_key)
            if combo_index >= 0:
                self.category_combo.setCurrentIndex(combo_index)
            else:
                self.category_combo.setCurrentIndex(0)
                self.selected_category_key = self.category_combo.currentData() or "default"
            self.category_combo.blockSignals(False)
        
        # Parse label name - maintain the index format "1. Action"
        name = label_data.get("name", "")
        # Split at the first period to separate the index from the action
        parts = name.split(".", 1)
        
        if len(parts) > 1 and parts[0].strip().isdigit():
            # Already has format "1. Action"
            self.name_edit.setText(parts[1].strip())
            self.label_index = parts[0].strip()
        else:
            # No proper format yet, use the full name
            self.name_edit.setText(name)
            # Try to extract index from the front if it's a digit
            if name and name[0].isdigit():
                index_end = 0
                while index_end < len(name) and name[index_end].isdigit():
                    index_end += 1
                self.label_index = name[:index_end]
            else:
                # No index found, use the list position
                for i in range(self.label_list.count()):
                    if self.label_list.item(i).data(Qt.UserRole) == self.current_label_id:
                        self.label_index = str(i + 1)
                        break
                else:
                    # Fallback if no match found
                    self.label_index = "1"
        
        color_rgba = label_data.get("color", [255, 165, 0, 180])
        self.color_button.blockSignals(True)
        self.color_button.setColor(QColor(*color_rgba))
        self.color_button.blockSignals(False)
        
        self.description_edit.setPlainText(label_data.get("description", ""))
        
        # Update frame range display with frames and timestamps
        start_frame = label_data.get("start_frame", 0)
        end_frame = label_data.get("end_frame", 0)
        
        # Get parent window to access FPS
        fps = 30.0  # Default
        parent = self.parent()
        while parent:
            if hasattr(parent, 'video_player') and hasattr(parent.video_player, 'fps'):
                fps = parent.video_player.fps
                break
            parent = parent.parent()
        
        # Calculate timestamps
        start_time_sec = start_frame / fps if fps > 0 else 0
        end_time_sec = end_frame / fps if fps > 0 else 0
        duration_sec = end_time_sec - start_time_sec
        
        # Format time as HH:MM:SS
        start_time = self.format_time(start_time_sec)
        end_time = self.format_time(end_time_sec)
        
        self.start_frame_label.setText(f"{start_frame} ({start_time})")
        self.end_frame_label.setText(f"{end_frame} ({end_time})")
        self.duration_label.setText(f"时长: {duration_sec:.1f}秒")
        
        # Enable editor
        self.set_editor_enabled(True)
        self.remove_button.setEnabled(True)
        self.duplicate_button.setEnabled(True)
    
    def current_category(self):
        """Return the currently selected category key."""
        return self.selected_category_key

    def update_frame_range(self, label_id, start_frame, end_frame):
        """Update the displayed frame range for a label."""
        if label_id == self.current_label_id:
            self.start_frame_label.setText(str(start_frame))
            self.end_frame_label.setText(str(end_frame))

    def format_time(self, seconds):
        """Format seconds to HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def clear_editor(self):
        """Clear the editor fields."""
        self.current_label_id = None
        self.name_edit.clear()
        self.description_edit.clear()
        if hasattr(self, "category_combo"):
            self.category_combo.blockSignals(True)
            default_index = self.category_combo.findData("default")
            if default_index >= 0:
                self.category_combo.setCurrentIndex(default_index)
            self.category_combo.blockSignals(False)
        self.selected_category_key = "default"
        self.color_button.blockSignals(True)
        self.color_button.setColor(QColor(255, 165, 0, 180))  # Reset to default color
        self.color_button.blockSignals(False)
        self.start_frame_label.setText("0 (00:00:00)")
        self.end_frame_label.setText("0 (00:00:00)")
        self.set_editor_enabled(False) 

    def on_template_selected(self, item):
        """Handle template selection."""
        self.selected_template = item.text()
        
        # Automatically update the name field when a template is selected
        if self.current_label_id is None:  # Only when creating new labels
            self.name_edit.setText(self.selected_template)
    
    def on_template_selection_changed(self):
        """Handle template selection change to enable/disable buttons."""
        has_selection = len(self.label_template_list.selectedItems()) > 0
        self.delete_template_button.setEnabled(has_selection)
        self.clear_template_button.setEnabled(has_selection)
    
    def delete_selected_template(self):
        """Delete the selected template."""
        current_item = self.label_template_list.currentItem()
        if current_item:
            template_name = current_item.text()
            
            # Confirm deletion
            from PySide6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self, "删除模板",
                f"确定要删除模板 '{template_name}' 吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Remove from list
                row = self.label_template_list.row(current_item)
                self.label_template_list.takeItem(row)
                
                # Save to settings
                self.save_templates_to_settings()
                
                # Update button states
                self.on_template_selection_changed()
    
    def edit_selected_template(self):
        """Edit the selected template name."""
        current_item = self.label_template_list.currentItem()
        if current_item:
            # Get new name from user
            from PySide6.QtWidgets import QInputDialog
            old_name = current_item.text()
            new_name, ok = QInputDialog.getText(
                self, "编辑模板", 
                "请输入新的模板名称:", 
                text=old_name
            )
            
            if ok and new_name.strip():
                new_name = new_name.strip()
                
                # Check if name already exists
                existing_names = []
                for i in range(self.label_template_list.count()):
                    if i != self.label_template_list.row(current_item):
                        existing_names.append(self.label_template_list.item(i).text())
                
                if new_name in existing_names:
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.warning(self, "重复名称", "该模板名称已存在！")
                    return
                
                # Update the item
                current_item.setText(new_name)
                
                # Save to settings
                self.save_templates_to_settings()
                
                # Update selected template if this was the selected one
                if self.selected_template == old_name:
                    self.selected_template = new_name

    def add_current_to_templates(self):
        """Add the current label name to templates."""
        current_name = self.name_edit.text().strip()
        if not current_name:
            return
        
        # Check if already exists
        items = [self.label_template_list.item(i).text() 
                 for i in range(self.label_template_list.count())]
        
        if current_name not in items:
            self.label_template_list.addItem(current_name)
            self.save_templates_to_settings()
        

        
    def create_new_label(self):
        """Create a new label."""
        # Modify your existing method to use the selected template
        # if no name is provided
        
        # Assuming you have a name_edit field for the label name
        name = self.name_edit.text().strip()
        if not name:
            name = self.selected_template
            self.name_edit.setText(name)
        
        # Rest of your existing label creation code
        # ... 

    def apply_changes(self):
        """Apply changes to the current label."""
        if self.current_label_id is None:
            return
        
        # Get current values
        name = self.name_edit.text()
        description = self.description_edit.toPlainText()
        color = self.color_button.color()
        
        # Extract category from name (e.g., "3. return" -> "return")
        category = "default"
        if '.' in name:
            parts = name.split('.')
            if len(parts) > 1:
                category = parts[1].strip()
        
        # Update the label in the timeline
        self.label_updated.emit(self.current_label_id, {
            'name': name,
            'description': description,
            'color': color,
            'category': category
        }) 
