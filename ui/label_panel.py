#!/usr/bin/env python
# -*- coding: utf-8 -*-

import uuid
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                              QLabel, QLineEdit, QListWidget, QListWidgetItem,
                              QColorDialog, QComboBox, QTextEdit, QFormLayout,
                              QGroupBox, QSplitter, QFrame, QMenu)
from PySide6.QtCore import Qt, Signal, Slot, QSize
from PySide6.QtGui import QColor, QIcon, QPixmap, QBrush, QPainter

class ColorButton(QPushButton):
    """Button that displays and selects a color."""
    
    colorChanged = Signal(QColor)
    
    def __init__(self, color=None, parent=None):
        super().__init__(parent)
        self._color = color or QColor(255, 165, 0)  # Default orange
        self.setFixedSize(24, 24)
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
        painter.drawRoundedRect(2, 2, self.width() - 4, self.height() - 4, 3, 3)
        painter.end()
        
        self.setIcon(QIcon(pixmap))
    
    def choose_color(self):
        """Open color dialog to select a new color."""
        color = QColorDialog.getColor(self._color, self, "Select Color", 
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
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Label categories
        self.categories = ["default", "action", "object", "person"]
        
        # Currently editing label
        self.current_label_id = None
        
        # Set up UI
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Create label list section
        list_group = QGroupBox("Labels")
        list_layout = QVBoxLayout(list_group)
        
        # Label list
        self.label_list = QListWidget()
        self.label_list.setSelectionMode(QListWidget.SingleSelection)
        self.label_list.currentItemChanged.connect(self.on_label_selected)
        list_layout.addWidget(self.label_list)
        
        # Add/Remove buttons
        buttons_layout = QHBoxLayout()
        
        self.add_label_button = QPushButton("New Label")
        self.add_label_button.clicked.connect(self.on_add_label)
        buttons_layout.addWidget(self.add_label_button)
        
        self.remove_label_button = QPushButton("Remove")
        self.remove_label_button.setEnabled(False)
        self.remove_label_button.clicked.connect(self.on_remove_label)
        buttons_layout.addWidget(self.remove_label_button)
        
        list_layout.addLayout(buttons_layout)
        
        # Create label editor section
        editor_group = QGroupBox("Label Properties")
        editor_layout = QVBoxLayout(editor_group)
        
        # Form layout for label properties
        form_layout = QFormLayout()
        
        # Label name
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Label name")
        self.name_edit.textChanged.connect(self.on_label_property_changed)
        form_layout.addRow("Name:", self.name_edit)
        
        # Category
        self.category_combo = QComboBox()
        self.category_combo.addItems(self.categories)
        self.category_combo.setEditable(True)
        self.category_combo.currentTextChanged.connect(self.on_label_property_changed)
        form_layout.addRow("Category:", self.category_combo)
        
        # Color
        self.color_button = ColorButton()
        self.color_button.colorChanged.connect(self.on_label_property_changed)
        form_layout.addRow("Color:", self.color_button)
        
        editor_layout.addLayout(form_layout)
        
        # Frame range display
        range_layout = QHBoxLayout()
        
        self.start_frame_label = QLabel("0")
        range_layout.addWidget(QLabel("Start Frame:"))
        range_layout.addWidget(self.start_frame_label)
        range_layout.addStretch()
        
        self.end_frame_label = QLabel("0")
        range_layout.addWidget(QLabel("End Frame:"))
        range_layout.addWidget(self.end_frame_label)
        
        editor_layout.addLayout(range_layout)
        
        # Description
        editor_layout.addWidget(QLabel("Description:"))
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        self.description_edit.textChanged.connect(self.on_label_property_changed)
        editor_layout.addWidget(self.description_edit)
        
        # Add to layout with splitter
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(list_group)
        splitter.addWidget(editor_group)
        layout.addWidget(splitter)
        
        # Disable editor initially
        self.set_editor_enabled(False)
    
    def set_editor_enabled(self, enabled):
        """Enable or disable the label editor."""
        self.name_edit.setEnabled(enabled)
        self.category_combo.setEnabled(enabled)
        self.color_button.setEnabled(enabled)
        self.description_edit.setEnabled(enabled)
    
    def clear(self):
        """Clear all labels."""
        self.label_list.clear()
        self.current_label_id = None
        self.set_editor_enabled(False)
        self.remove_label_button.setEnabled(False)
    
    @Slot()
    def on_add_label(self):
        """Add a new label."""
        # Create a new label
        label_id = str(uuid.uuid4())
        label_data = {
            "id": label_id,
            "name": "New Label",
            "start_frame": 0,
            "end_frame": 0,
            "color": [255, 165, 0, 180],  # Orange with transparency
            "category": "default",
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
    
    @Slot()
    def on_remove_label(self):
        """Remove the selected label."""
        if not self.current_label_id:
            return
            
        # Emit signal to remove label
        self.label_deleted.emit(self.current_label_id)
        
        # Remove from list
        for i in range(self.label_list.count()):
            item = self.label_list.item(i)
            if item.data(Qt.UserRole) == self.current_label_id:
                self.label_list.takeItem(i)
                break
        
        # Clear editor
        self.current_label_id = None
        self.set_editor_enabled(False)
        self.remove_label_button.setEnabled(False)
    
    @Slot()
    def on_label_property_changed(self):
        """Handle changes to label properties."""
        if not self.current_label_id:
            return
            
        # Create updated label data
        label_data = {
            "id": self.current_label_id,
            "name": self.name_edit.text(),
            "category": self.category_combo.currentText(),
            "color": [
                self.color_button.color().red(),
                self.color_button.color().green(),
                self.color_button.color().blue(),
                self.color_button.color().alpha()
            ],
            "description": self.description_edit.toPlainText()
        }
        
        # Update list item text
        for i in range(self.label_list.count()):
            item = self.label_list.item(i)
            if item.data(Qt.UserRole) == self.current_label_id:
                item.setText(f"{label_data['name']} ({label_data['category']})")
                
                # Set color of item
                pixmap = QPixmap(16, 16)
                pixmap.fill(Qt.transparent)
                
                painter = QPainter(pixmap)
                painter.setBrush(QBrush(self.color_button.color()))
                painter.setPen(Qt.black)
                painter.drawRect(0, 0, 15, 15)
                painter.end()
                
                item.setIcon(QIcon(pixmap))
                break
        
        # Emit signal to update label
        self.label_updated.emit(label_data)
    
    @Slot(QListWidgetItem, QListWidgetItem)
    def on_label_selected(self, current, previous):
        """Handle selection of a label from the list."""
        if not current:
            self.current_label_id = None
            self.set_editor_enabled(False)
            self.remove_label_button.setEnabled(False)
            return
            
        # Get label ID from item
        label_id = current.data(Qt.UserRole)
        self.current_label_id = label_id
        
        # Enable editor and remove button
        self.set_editor_enabled(True)
        self.remove_label_button.setEnabled(True)
        
        # Emit signal for selected label
        self.label_selected.emit(label_id)
    
    def add_label_to_list(self, label_data):
        """Add a label to the list widget."""
        label_id = label_data["id"]
        label_name = label_data.get("name", "Unnamed")
        category = label_data.get("category", "default")
        color_rgba = label_data.get("color", [255, 165, 0, 180])
        
        # Create list item
        item = QListWidgetItem(f"{label_name} ({category})")
        item.setData(Qt.UserRole, label_id)
        
        # Create color icon
        color = QColor(*color_rgba)
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.black)
        painter.drawRect(0, 0, 15, 15)
        painter.end()
        
        item.setIcon(QIcon(pixmap))
        
        # Add to list
        self.label_list.addItem(item)
    
    def update_label_data(self, label_data):
        """Update the editor with label data."""
        self.current_label_id = label_data["id"]
        
        # Update form fields
        self.name_edit.setText(label_data.get("name", ""))
        
        category = label_data.get("category", "default")
        index = self.category_combo.findText(category)
        if index >= 0:
            self.category_combo.setCurrentIndex(index)
        else:
            self.category_combo.setCurrentText(category)
            
        color_rgba = label_data.get("color", [255, 165, 0, 180])
        self.color_button.setColor(QColor(*color_rgba))
        
        self.description_edit.setPlainText(label_data.get("description", ""))
        
        # Update frame range display
        self.start_frame_label.setText(str(label_data.get("start_frame", 0)))
        self.end_frame_label.setText(str(label_data.get("end_frame", 0)))
        
        # Enable editor
        self.set_editor_enabled(True)
        self.remove_label_button.setEnabled(True)
    
    def update_frame_range(self, label_id, start_frame, end_frame):
        """Update the displayed frame range for a label."""
        if label_id == self.current_label_id:
            self.start_frame_label.setText(str(start_frame))
            self.end_frame_label.setText(str(end_frame)) 