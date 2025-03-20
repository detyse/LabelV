#!/usr/bin/env python
# -*- coding: utf-8 -*-

import uuid
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                              QLabel, QLineEdit, QListWidget, QListWidgetItem,
                              QColorDialog, QComboBox, QTextEdit, QFormLayout,
                              QGroupBox, QSplitter, QFrame, QMenu, QPlainTextEdit)
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
        
        # Initialize label properties
        self.current_label_id = None
        self.label_index = "1"  # Default label index
        
        # Set up UI
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Create list view for labels
        self.label_list = QListWidget()
        self.label_list.setSelectionMode(QListWidget.SingleSelection)
        self.label_list.currentItemChanged.connect(self.on_label_selected)
        layout.addWidget(self.label_list)
        
        # Create label editor section
        editor_group = QGroupBox("Label Editor")
        editor_layout = QFormLayout()
        
        # Label name edit
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Label name")
        self.name_edit.editingFinished.connect(self.on_label_property_changed)
        editor_layout.addRow("Name:", self.name_edit)
        
        # Color selector
        self.color_button = ColorButton()
        self.color_button.colorChanged.connect(self.on_label_property_changed)
        editor_layout.addRow("Color:", self.color_button)
        
        # Description text edit
        self.description_edit = QPlainTextEdit()
        self.description_edit.setPlaceholderText("Enter description...")
        self.description_edit.textChanged.connect(self.on_label_property_changed)
        self.description_edit.setMaximumHeight(80)
        editor_layout.addRow("Description:", self.description_edit)
        
        # Frame range display
        self.start_frame_label = QLabel("0")
        self.end_frame_label = QLabel("0")
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(self.start_frame_label)
        frame_layout.addWidget(QLabel("-"))
        frame_layout.addWidget(self.end_frame_label)
        editor_layout.addRow("Frames:", frame_layout)
        
        # Set the layout for the editor group
        editor_group.setLayout(editor_layout)
        layout.addWidget(editor_group)
        
        # Create button section
        button_layout = QHBoxLayout()
        
        # Remove button
        self.remove_button = QPushButton("Remove Label")
        self.remove_button.clicked.connect(self.on_remove_label)
        button_layout.addWidget(self.remove_button)
        
        layout.addLayout(button_layout)
        
        # Disable editor initially
        self.set_editor_enabled(False)
    
    def set_editor_enabled(self, enabled):
        """Enable or disable the label editor."""
        self.name_edit.setEnabled(enabled)
        self.color_button.setEnabled(enabled)
        self.description_edit.setEnabled(enabled)
    
    def clear(self):
        """Clear all labels."""
        self.label_list.clear()
        self.current_label_id = None
        self.set_editor_enabled(False)
        self.remove_button.setEnabled(False)
    
    @Slot()
    def on_add_label(self):
        """Add a new label."""
        # Create a new label
        label_id = str(uuid.uuid4())
        
        # Get the next number based on existing labels
        count = self.label_list.count() + 1
        
        label_data = {
            "id": label_id,
            "name": f"Label {count}",
            "start_frame": 0,
            "end_frame": 0,
            "color": [255, 165, 0, 180],  # Orange with transparency
            "category": "default",  # Keep for compatibility
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
    
    @Slot()
    def on_label_property_changed(self):
        """Handle changes to label properties."""
        if not self.current_label_id:
            return
            
        # Get the action name from the text field
        action_name = self.name_edit.text().strip()
        
        # Format the full name with index prefix
        full_name = f"{self.label_index}. {action_name}" if action_name else f"{self.label_index}."
        
        # Create updated label data
        label_data = {
            "id": self.current_label_id,
            "name": full_name,
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
                item.setText(full_name)
                
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
            self.remove_button.setEnabled(False)
            return
            
        # Get label ID from item
        label_id = current.data(Qt.UserRole)
        self.current_label_id = label_id
        
        # Enable editor and remove button
        self.set_editor_enabled(True)
        self.remove_button.setEnabled(True)
        
        # Emit signal for selected label
        self.label_selected.emit(label_id)
    
    def add_label_to_list(self, label_data):
        """Add a label to the list widget."""
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
    
    def update_list_item(self, row, label_data):
        """Update an existing list item with new data."""
        item = self.label_list.item(row)
        if not item:
            return
        
        name = label_data.get("name", "")
        color = label_data.get("color", [255, 165, 0, 180])
        
        item.setText(name)
        
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
        self.color_button.setColor(QColor(*color_rgba))
        
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
        
        # Format time as HH:MM:SS
        start_time = self.format_time(start_time_sec)
        end_time = self.format_time(end_time_sec)
        
        self.start_frame_label.setText(f"{start_frame} ({start_time})")
        self.end_frame_label.setText(f"{end_frame} ({end_time})")
        
        # Enable editor
        self.set_editor_enabled(True)
        self.remove_button.setEnabled(True)
    
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
        self.color_button.setColor(QColor(255, 165, 0, 180))  # Reset to default color
        self.start_frame_label.setText("0")
        self.end_frame_label.setText("0")
        self.set_editor_enabled(False) 