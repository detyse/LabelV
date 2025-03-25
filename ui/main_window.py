#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import uuid
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                              QToolBar, QFileDialog, QMessageBox, QDockWidget, QFrame)
from PySide6.QtCore import Qt, Slot, QSettings, QEvent, QTimer
from PySide6.QtGui import QAction, QKeySequence, QColor

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
        
        # Connect timeline label selection to panel
        self.timeline.label_selected.connect(self.on_timeline_label_selected)
        
        # Connect label playback request to player
        self.timeline.label_playback_requested.connect(self.play_label_segment)
        
        # Connect signals for timeline-label panel synchronization
        self.timeline.label_created.connect(self.on_label_created)
        
        # Connect label name change signal to timeline
        self.label_panel.label_name_changed.connect(self.timeline.update_label_name)
        
        # Add this connection
        self.label_panel.label_template_list.itemClicked.connect(
            self.update_template_selection
        )
        
        # Connect label panel selection to timeline
        self.label_panel.label_selected.connect(self.timeline.select_label)
        
        # Create toolbar and actions
        self.create_actions()
        self.create_toolbar()
        
        # Current project data
        self.current_video_path = None
        self.current_project_path = None
        self.labels = []
        
        # Setup the status bar
        self.statusBar().showMessage("Ready")
        
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
    
    def create_actions(self):
        """Create application actions."""
        # Open video action
        self.open_video_action = QAction("Open Video", self)
        self.open_video_action.setShortcut(QKeySequence.Open)
        self.open_video_action.triggered.connect(self.open_video)
        
        # Export labels action
        self.export_labels_action = QAction("Export Labels", self)
        self.export_labels_action.setEnabled(False)
        self.export_labels_action.triggered.connect(self.export_labels)
        
        # Save labels action (new)
        self.save_labels_action = QAction("Save Labels", self)
        self.save_labels_action.setShortcut(QKeySequence("Ctrl+S"))
        self.save_labels_action.setEnabled(False)
        self.save_labels_action.triggered.connect(self.save_labels)
        
        # Load labels action (new)
        self.load_labels_action = QAction("Load Labels", self)
        self.load_labels_action.setShortcut(QKeySequence("Ctrl+L"))
        self.load_labels_action.setEnabled(False)
        self.load_labels_action.triggered.connect(self.load_labels)
        
        # Mode actions
        self.mode_action_choose = QAction("View Mode", self)
        self.mode_action_choose.setShortcut(QKeySequence("C"))
        self.mode_action_choose.setCheckable(True)
        self.mode_action_choose.triggered.connect(lambda: self.timeline.set_mode(self.timeline.CHOOSE_MODE))
        
        self.mode_action_edit = QAction("Edit Mode", self)
        self.mode_action_edit.setShortcut(QKeySequence("X"))
        self.mode_action_edit.setCheckable(True)
        self.mode_action_edit.triggered.connect(lambda: self.timeline.set_mode(self.timeline.EDIT_MODE))
    
    def create_toolbar(self):
        """Create application toolbar."""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        toolbar.addAction(self.open_video_action)
        toolbar.addAction(self.save_labels_action)
        toolbar.addAction(self.load_labels_action)
        toolbar.addAction(self.export_labels_action)
        
        # Add separator
        toolbar.addSeparator()
        
        # Add mode actions
        toolbar.addAction(self.mode_action_choose)
        toolbar.addAction(self.mode_action_edit)
    
    def update_mode(self, mode):
        """Update UI based on timeline mode."""
        if mode == self.timeline.CHOOSE_MODE:
            # View mode - disable label editing
            self.mode_action_choose.setChecked(True)
            self.mode_action_edit.setChecked(False)
            
            # No need to disable a button that doesn't exist anymore
            # self.label_panel.add_label_button.setEnabled(False)
            
            # Set status message
            self.statusBar().showMessage("View Mode: Select labels to play segments", 2000)
        else:  # EDIT_MODE
            # Edit mode - enable label editing
            self.mode_action_choose.setChecked(False)
            self.mode_action_edit.setChecked(True)
            
            # No need to enable a button that doesn't exist anymore
            # self.label_panel.add_label_button.setEnabled(True)
            
            # Set status message
            self.statusBar().showMessage("Edit Mode: Right-click and drag to create labels", 2000)

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
                
                # Enable label operations
                self.save_labels_action.setEnabled(True)
                self.load_labels_action.setEnabled(True)
                self.export_labels_action.setEnabled(True)
                
                # Try to auto-load matching labels file
                json_path = os.path.splitext(video_path)[0] + ".json"
                if os.path.exists(json_path):
                    response = QMessageBox.question(self, "Load Labels",
                        f"Found label file for this video. Load it?",
                        QMessageBox.Yes | QMessageBox.No)
                    if response == QMessageBox.Yes:
                        self.load_labels()
            else:
                QMessageBox.critical(self, "Error", "Failed to open video file.")
    
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

    def save_labels(self):
        """Save labels to JSON file with same name as video."""
        if not self.current_video_path:
            QMessageBox.warning(self, "Warning", "No video loaded")
            return
        
        # Generate JSON path with same name as video
        video_path = self.current_video_path
        json_path = os.path.splitext(video_path)[0] + ".json"
        
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
        try:
            with open(json_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            self.statusBar().showMessage(f"Labels saved to {json_path}", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save labels: {str(e)}")

    def load_labels(self):
        """Load labels from JSON file with same name as video."""
        if not self.current_video_path:
            QMessageBox.warning(self, "Warning", "No video loaded")
            return
        
        # Generate JSON path with same name as video
        video_path = self.current_video_path
        json_path = os.path.splitext(video_path)[0] + ".json"
        
        # Check if file exists
        if not os.path.exists(json_path):
            QMessageBox.information(self, "Information", "No label file found for this video")
            return
        
        try:
            # Load JSON data
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Verify metadata matches
            video_basename = os.path.basename(self.current_video_path)
            if data.get("video_file") != video_basename:
                QMessageBox.warning(self, "Warning", 
                                 f"Label file doesn't match video: expected {video_basename}, found {data.get('video_file')}")
                return
            
            # Check frame count to ensure video hasn't changed
            if data.get("total_frames") != self.video_player.frame_count:
                response = QMessageBox.question(self, "Warning", 
                    "Frame count mismatch. Video may have changed. Load labels anyway?",
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
            
            self.statusBar().showMessage(f"Loaded {labels_loaded} labels from {json_path}", 3000)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            QMessageBox.critical(self, "Error", f"Failed to load labels: {str(e)}\n\nDetails: {error_details}") 