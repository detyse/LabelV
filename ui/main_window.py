#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                              QToolBar, QFileDialog, QMessageBox, QDockWidget, QFrame)
from PySide6.QtCore import Qt, Slot, QSettings
from PySide6.QtGui import QAction, QKeySequence

from ui.video_player import VideoPlayer
from ui.timeline import TimelineWidget
from ui.label_panel import LabelPanel

class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Video Label Tool")
        self.resize(1200, 800)
        
        self.settings = QSettings("LabelV", "VideoLabelTool")
        
        # Create components
        self.video_player = VideoPlayer()
        self.timeline = TimelineWidget()
        self.label_panel = LabelPanel()
        
        # Set up central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add video player to main layout
        main_layout.addWidget(self.video_player, 4)
        
        # Add a separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setLineWidth(1)
        main_layout.addWidget(separator)
        
        # Add timeline to main layout with more space
        main_layout.addWidget(self.timeline, 2)
        
        # Create dock widget for label panel
        label_dock = QDockWidget("Labels", self)
        label_dock.setWidget(self.label_panel)
        label_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea, label_dock)
        
        # Connect signals
        self.video_player.position_changed.connect(self.timeline.update_position)
        self.timeline.position_changed.connect(self.video_player.set_position)
        
        self.label_panel.label_added.connect(self.timeline.add_label)
        self.label_panel.label_deleted.connect(self.timeline.remove_label)
        self.label_panel.label_selected.connect(self.timeline.select_label)
        
        # Create toolbar and actions
        self.create_actions()
        self.create_toolbar()
        
        # Current project data
        self.current_video_path = None
        self.current_project_path = None
        self.labels = []
    
    def create_actions(self):
        """Create application actions."""
        # Open video action
        self.open_video_action = QAction("Open Video", self)
        self.open_video_action.setShortcut(QKeySequence.Open)
        self.open_video_action.triggered.connect(self.open_video)
        
        # Save project action
        self.save_project_action = QAction("Save Project", self)
        self.save_project_action.setShortcut(QKeySequence.Save)
        self.save_project_action.setEnabled(False)
        self.save_project_action.triggered.connect(self.save_project)
        
        # Export labels action
        self.export_labels_action = QAction("Export Labels", self)
        self.export_labels_action.setEnabled(False)
        self.export_labels_action.triggered.connect(self.export_labels)
    
    def create_toolbar(self):
        """Create application toolbar."""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        toolbar.addAction(self.open_video_action)
        toolbar.addAction(self.save_project_action)
        toolbar.addAction(self.export_labels_action)
    
    @Slot()
    def open_video(self):
        """Open a video file."""
        last_dir = self.settings.value("last_video_dir", "")
        video_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", last_dir,
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)"
        )
        
        if video_path:
            self.settings.setValue("last_video_dir", os.path.dirname(video_path))
            self.current_video_path = video_path
            self.setWindowTitle(f"Video Label Tool - {os.path.basename(video_path)}")
            
            # Load video
            success = self.video_player.load_video(video_path)
            if success:
                self.timeline.clear()
                self.label_panel.clear()
                
                # Explicitly set the frame count
                self.timeline.set_frame_count(self.video_player.frame_count)
                
                # Ensure timeline gets updated
                self.timeline.update()
                
                self.save_project_action.setEnabled(True)
                self.export_labels_action.setEnabled(True)
            else:
                QMessageBox.critical(self, "Error", "Failed to open video file.")
    
    @Slot()
    def save_project(self):
        """Save the current project."""
        if not self.current_video_path:
            return
            
        if not self.current_project_path:
            last_dir = self.settings.value("last_project_dir", "")
            project_path, _ = QFileDialog.getSaveFileName(
                self, "Save Project", last_dir,
                "Label Project (*.lvp);;All Files (*)"
            )
            if not project_path:
                return
                
            if not project_path.endswith(".lvp"):
                project_path += ".lvp"
                
            self.current_project_path = project_path
            self.settings.setValue("last_project_dir", os.path.dirname(project_path))
        
        # Collect data
        project_data = {
            "video_path": self.current_video_path,
            "labels": self.timeline.get_labels()
        }
        
        # Save project file
        try:
            with open(self.current_project_path, 'w') as f:
                json.dump(project_data, f, indent=2)
            QMessageBox.information(self, "Success", "Project saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save project: {str(e)}")
    
    @Slot()
    def export_labels(self):
        """Export labels to a JSON file."""
        if not self.current_video_path:
            return
            
        last_dir = self.settings.value("last_export_dir", "")
        export_path, _ = QFileDialog.getSaveFileName(
            self, "Export Labels", last_dir,
            "JSON Files (*.json);;CSV Files (*.csv);;All Files (*)"
        )
        
        if not export_path:
            return
            
        # Determine export format based on extension
        if export_path.endswith(".csv"):
            self.export_labels_csv(export_path)
        else:
            if not export_path.endswith(".json"):
                export_path += ".json"
            self.export_labels_json(export_path)
            
        self.settings.setValue("last_export_dir", os.path.dirname(export_path))

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
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            QMessageBox.information(self, "Success", "Labels exported successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export labels: {str(e)}")

    def export_labels_csv(self, export_path):
        """Export labels to a CSV file."""
        # Get current video FPS
        fps = self.video_player.fps if hasattr(self.video_player, 'fps') else 30.0
        
        # Collect label data with timestamps
        labels = self.timeline.get_labels_for_export(fps)
        
        try:
            with open(export_path, 'w', newline='') as f:
                import csv
                fieldnames = ["id", "category", "name", "start_frame", "end_frame", 
                             "start_time", "end_time", "duration", "description"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for label in labels:
                    writer.writerow(label)
            QMessageBox.information(self, "Success", "Labels exported successfully to CSV.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export labels to CSV: {str(e)}") 