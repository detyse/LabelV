#!/usr/bin/env python
# -*- coding: utf-8 -*-

import uuid
import math
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PySide6.QtCore import Qt, Signal, Slot, QRectF, QPointF
from PySide6.QtGui import (QPainter, QBrush, QPen, QColor, QPainterPath, 
                          QLinearGradient, QFont, QFontMetrics, QRadialGradient)

class Label:
    """Represents a video label with start and end frames."""
    
    def __init__(self, label_id=None, name="Unnamed", start_frame=0, end_frame=0, 
                 color=None, category=None, description=None):
        self.id = label_id if label_id else str(uuid.uuid4())
        self.name = name
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.color = color or QColor(255, 165, 0, 180)  # Default orange with transparency
        self.category = category or "default"
        self.description = description or ""
        self.selected = False
    
    def to_dict(self):
        """Convert label to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "color": [self.color.red(), self.color.green(), self.color.blue(), self.color.alpha()],
            "category": self.category,
            "description": self.description
        }
        
    def to_export_dict(self, fps=30.0):
        """Convert label to dictionary for export with timestamps."""
        # Calculate timestamps
        start_time_sec = self.start_frame / fps if fps > 0 else 0
        end_time_sec = self.end_frame / fps if fps > 0 else 0
        duration_sec = end_time_sec - start_time_sec
        
        # Format timestamps as strings
        start_time = self.format_timestamp(start_time_sec)
        end_time = self.format_timestamp(end_time_sec)
        duration = self.format_timestamp(duration_sec)
        
        return {
            "id": self.id,
            "category": self.category,
            "name": self.name,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "description": self.description
        }
    
    @staticmethod
    def format_timestamp(seconds):
        """Format seconds to HH:MM:SS.mmm format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_remainder = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds_remainder:06.3f}"
    
    @classmethod
    def from_dict(cls, data):
        """Create label from dictionary."""
        color = QColor(*data.get("color", [255, 165, 0, 180]))
        return cls(
            label_id=data.get("id"),
            name=data.get("name", "Unnamed"),
            start_frame=data.get("start_frame", 0),
            end_frame=data.get("end_frame", 0),
            color=color,
            category=data.get("category", "default"),
            description=data.get("description", "")
        )
    
    def duration(self):
        """Get label duration in frames."""
        return max(0, self.end_frame - self.start_frame)
    
    def contains_frame(self, frame):
        """Check if the label contains a specific frame."""
        return self.start_frame <= frame <= self.end_frame


class TimelineWidget(QWidget):
    """Widget for displaying video timeline and labeled regions."""
    
    # Signals
    position_changed = Signal(int)  # Current position
    label_selected = Signal(str)  # Label ID
    label_double_clicked = Signal(str)  # Label ID
    
    # States for mouse interactions
    NONE = 0
    DRAGGING_POSITION = 1
    CREATING_LABEL = 2
    MOVING_LABEL = 3
    RESIZING_LABEL_START = 4
    RESIZING_LABEL_END = 5
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Timeline properties
        self.frame_count = 0
        self.current_frame = 0
        self.zoom_level = 1.0
        self.offset = 0
        self.fps = 30.0  # Default FPS
        
        # Labels
        self.labels = []
        
        # Interaction state
        self.state = self.NONE
        self.mouse_down_pos = None
        self.mouse_down_frame = 0
        self.selected_label_idx = -1
        self.hover_label_idx = -1
        self.drag_start_frame = 0
        
        # UI settings
        self.timeline_height = 50
        self.label_track_height = 30
        self.time_marker_height = 15
        self.resize_handle_width = 5
        
        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)
        
        # Set minimum height
        self.setMinimumHeight(100)
        
        # Set focus policy
        self.setFocusPolicy(Qt.StrongFocus)
    
    def get_frame_at_position(self, x_pos):
        """Convert x position to frame number."""
        # Calculate visible timeline width
        timeline_width = self.width()
        
        # Calculate the visible frame range
        visible_frames = self.frame_count / self.zoom_level
        
        # Calculate frame position
        frame = int(self.offset + (x_pos / timeline_width) * visible_frames)
        
        # Clamp to valid range
        return max(0, min(self.frame_count - 1, frame))
    
    def get_position_for_frame(self, frame):
        """Convert frame number to x position."""
        if self.frame_count == 0:
            return 0
            
        # Calculate visible timeline width
        timeline_width = self.width()
        
        # Calculate the visible frame range
        visible_frames = self.frame_count / self.zoom_level
        
        # Calculate x position
        relative_frame = frame - self.offset
        x_pos = (relative_frame / visible_frames) * timeline_width
        
        return x_pos
    
    def paintEvent(self, event):
        """Handle paint event to draw the timeline."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        
        if self.frame_count == 0:
            # No video loaded
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignCenter, "No video loaded")
            return
        
        # Calculate dimensions
        width = self.width()
        height = self.height()
        
        # Draw timeline track background with gradient for better visual appeal
        timeline_rect = QRectF(0, 0, width, self.timeline_height)
        gradient = QLinearGradient(0, 0, 0, self.timeline_height)
        gradient.setColorAt(0, QColor(70, 70, 70))
        gradient.setColorAt(1, QColor(50, 50, 50))
        painter.fillRect(timeline_rect, gradient)
        
        # Draw timeline ruler with grid lines
        self.draw_time_markers(painter, timeline_rect)
        
        # Draw label tracks
        label_area_rect = QRectF(0, self.timeline_height, width, height - self.timeline_height)
        painter.fillRect(label_area_rect, QColor(45, 45, 45))
        
        # Draw thin separator line between timeline and label area
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawLine(0, self.timeline_height, width, self.timeline_height)
        
        # Draw labels
        self.draw_labels(painter, label_area_rect)
        
        # Draw current position marker with enhanced style
        position_x = self.get_position_for_frame(self.current_frame)
        self.draw_timeline_scrubber(painter, position_x, height)
    
    def draw_time_markers(self, painter, rect):
        """Draw time markers on the timeline."""
        if self.frame_count == 0:
            return
            
        width = rect.width()
        
        # Determine appropriate marker interval based on zoom
        visible_frames = self.frame_count / self.zoom_level
        frame_per_pixel = visible_frames / width
        
        # Calculate appropriate interval (aim for markers roughly every 100 pixels)
        intervals = [1, 5, 10, 30, 60, 300, 600, 1800, 3600, 7200, 18000, 36000]
        
        # Use the instance FPS instead of hardcoded value
        fps = self.fps
        
        # Convert frames to seconds for marker intervals
        seconds_per_pixel = frame_per_pixel / fps
        interval_seconds = 1
        
        for interval in intervals:
            if interval / seconds_per_pixel >= 100:
                interval_seconds = interval
                break
        
        # Draw background grid lines (minor ticks)
        painter.setPen(QPen(QColor(70, 70, 70), 1, Qt.DotLine))
        minor_interval = interval_seconds / 5  # Smaller ticks in between major ones
        
        start_second_minor = math.floor(self.offset / fps / minor_interval) * minor_interval
        end_second_minor = math.ceil((self.offset + visible_frames) / fps / minor_interval) * minor_interval
        
        for second in np.arange(start_second_minor, end_second_minor + minor_interval, minor_interval):
            frame = int(second * fps)
            x_pos = self.get_position_for_frame(frame)
            
            if 0 <= x_pos <= width:
                # Draw minor tick line (shorter)
                painter.drawLine(QPointF(x_pos, rect.bottom() - self.time_marker_height / 3), 
                                QPointF(x_pos, rect.bottom()))
        
        # Draw main markers
        painter.setPen(QColor(180, 180, 180))
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        
        # Start from the first visible marker
        start_second = math.floor(self.offset / fps / interval_seconds) * interval_seconds
        end_second = math.ceil((self.offset + visible_frames) / fps / interval_seconds) * interval_seconds
        
        for second in np.arange(start_second, end_second + interval_seconds, interval_seconds):
            frame = int(second * fps)
            x_pos = self.get_position_for_frame(frame)
            
            if 0 <= x_pos <= width:
                # Draw marker line
                painter.setPen(QPen(QColor(120, 120, 120), 1))
                painter.drawLine(QPointF(x_pos, rect.top()), 
                                QPointF(x_pos, rect.bottom()))
                
                painter.setPen(QColor(200, 200, 200))
                
                # Format time
                minutes, seconds = divmod(second, 60)
                hours, minutes = divmod(minutes, 60)
                
                if hours > 0:
                    time_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                else:
                    time_text = f"{minutes:02d}:{seconds:02d}"
                
                # Draw time text
                time_rect = QRectF(x_pos - 50, rect.bottom() - self.time_marker_height - 15, 100, 15)
                painter.drawText(time_rect, Qt.AlignCenter, time_text)
                
                # Draw frame number below (smaller)
                small_font = QFont()
                small_font.setPointSize(7)
                painter.setFont(small_font)
                
                frame_rect = QRectF(x_pos - 50, rect.bottom() - 12, 100, 12)
                painter.setPen(QColor(150, 150, 150))
                painter.drawText(frame_rect, Qt.AlignCenter, f"Frame: {frame}")
                
                painter.setFont(font)  # Restore original font
    
    def draw_labels(self, painter, rect):
        """Draw label tracks and labels."""
        if not self.labels:
            return
            
        # Group labels by category to draw them in separate tracks
        categories = {}
        for label in self.labels:
            if label.category not in categories:
                categories[label.category] = []
            categories[label.category].append(label)
        
        # Calculate track height
        num_tracks = max(1, len(categories))
        track_height = min(self.label_track_height, rect.height() / num_tracks)
        
        # Draw each track
        track_idx = 0
        for category, category_labels in categories.items():
            track_y = rect.top() + track_idx * track_height
            track_rect = QRectF(0, track_y, rect.width(), track_height)
            
            # Draw track background with slight alternating colors
            bg_color = QColor(55, 55, 55) if track_idx % 2 == 0 else QColor(50, 50, 50)
            painter.fillRect(track_rect, bg_color)
            
            # Draw category name
            painter.setPen(QColor(200, 200, 200))
            font = QFont()
            font.setPointSize(8)
            painter.setFont(font)
            text_rect = QRectF(5, track_y, 100, track_height)
            painter.drawText(text_rect, Qt.AlignVCenter, category)
            
            # Draw labels in this track
            for label in category_labels:
                self.draw_label(painter, label, track_rect)
            
            track_idx += 1
    
    def draw_label(self, painter, label, track_rect):
        """Draw a single label on the timeline."""
        start_x = self.get_position_for_frame(label.start_frame)
        end_x = self.get_position_for_frame(label.end_frame)
        
        # Skip if not visible
        if end_x < 0 or start_x > track_rect.width():
            return
            
        # Create label rectangle
        margin = 2
        label_rect = QRectF(
            start_x,
            track_rect.top() + margin,
            max(2, end_x - start_x),  # Ensure minimum width of 2 pixels
            track_rect.height() - 2 * margin
        )
        
        # Create gradient for label background
        gradient = QLinearGradient(
            label_rect.topLeft(),
            label_rect.bottomLeft()
        )
        
        base_color = QColor(label.color)
        if label.selected:
            # Make selected labels brighter
            base_color = base_color.lighter(130)
            
        darker_color = base_color.darker(130)
        gradient.setColorAt(0, base_color)
        gradient.setColorAt(1, darker_color)
        
        # Draw label background
        painter.setBrush(QBrush(gradient))
        
        # Draw border
        if label.selected:
            # Add glow effect for selected labels
            glow_pen = QPen(QColor(255, 255, 255, 100), 3)
            painter.setPen(glow_pen)
            painter.drawRoundedRect(label_rect.adjusted(-1, -1, 1, 1), 4, 4)
            
            painter.setPen(QPen(QColor(255, 255, 255), 2))
        elif self.hover_label_idx >= 0 and self.labels[self.hover_label_idx].id == label.id:
            painter.setPen(QPen(QColor(220, 220, 220), 1))
        else:
            painter.setPen(QPen(darker_color, 1))
            
        painter.drawRoundedRect(label_rect, 4, 4)
        
        # Draw label text if there's enough space
        if label_rect.width() > 30:
            font = QFont()
            font.setPointSize(8)
            painter.setFont(font)
            
            # Format timestamps for start and end frames
            start_time_sec = label.start_frame / self.fps if self.fps > 0 else 0
            end_time_sec = label.end_frame / self.fps if self.fps > 0 else 0
            
            # Format time as MM:SS or HH:MM:SS depending on length
            start_time = self.format_time_compact(start_time_sec)
            end_time = self.format_time_compact(end_time_sec)
            
            # Create label text with timestamps
            if label_rect.width() > 150:
                # Full format with timestamps if there's enough space
                display_text = f"{label.name} [{start_time}-{end_time}]"
            elif label_rect.width() > 80:
                # Shorter label name with timestamps
                name = label.name[:8] + "..." if len(label.name) > 10 else label.name
                display_text = f"{name} [{start_time}]"
            else:
                # Just the label name if space is limited
                display_text = label.name
            
            # Calculate text bounding rect
            font_metrics = QFontMetrics(font)
            text_width = font_metrics.horizontalAdvance(display_text)
            
            if text_width + 10 <= label_rect.width():
                text_rect = label_rect.adjusted(5, 0, -5, 0)
                painter.setPen(QColor(0, 0, 0))
                painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, display_text)
    
    @Slot(int)
    def update_position(self, position):
        """Update the current frame position."""
        if position != self.current_frame:
            self.current_frame = position
            self.update()
    
    def clear(self):
        """Clear all labels and reset timeline."""
        self.labels.clear()
        self.frame_count = 0
        self.current_frame = 0
        self.zoom_level = 1.0
        self.offset = 0
        self.selected_label_idx = -1
        self.hover_label_idx = -1
        self.update()
    
    @Slot(int)
    def set_frame_count(self, count):
        """Set the total frame count."""
        self.frame_count = count
        self.update()
    
    @Slot(dict)
    def add_label(self, label_data):
        """Add a new label."""
        # Create label from data
        label = Label.from_dict(label_data)
        
        # Add to labels list
        self.labels.append(label)
        
        # Clear selection and select the new label
        self.select_label_by_id(label.id)
        
        self.update()
    
    @Slot(str)
    def remove_label(self, label_id):
        """Remove a label by ID."""
        for i, label in enumerate(self.labels):
            if label.id == label_id:
                self.labels.pop(i)
                if self.selected_label_idx == i:
                    self.selected_label_idx = -1
                elif self.selected_label_idx > i:
                    self.selected_label_idx -= 1
                    
                if self.hover_label_idx == i:
                    self.hover_label_idx = -1
                elif self.hover_label_idx > i:
                    self.hover_label_idx -= 1
                    
                self.update()
                return
    
    @Slot(str)
    def select_label(self, label_id):
        """Select a label by ID."""
        self.select_label_by_id(label_id)
        self.update()
    
    def select_label_by_id(self, label_id):
        """Helper to select a label by ID."""
        # Clear current selection
        if self.selected_label_idx >= 0:
            self.labels[self.selected_label_idx].selected = False
        
        # Find and select the label
        self.selected_label_idx = -1
        for i, label in enumerate(self.labels):
            if label.id == label_id:
                label.selected = True
                self.selected_label_idx = i
                break
    
    def find_label_at_position(self, pos):
        """Find the label at the given position."""
        if not self.labels:
            return -1, self.NONE
            
        frame = self.get_frame_at_position(pos.x())
        
        # Check for labels containing this frame
        for i, label in enumerate(self.labels):
            if label.contains_frame(frame):
                start_x = self.get_position_for_frame(label.start_frame)
                end_x = self.get_position_for_frame(label.end_frame)
                
                # Check if clicked near the resize handles
                if abs(pos.x() - start_x) <= self.resize_handle_width:
                    return i, self.RESIZING_LABEL_START
                elif abs(pos.x() - end_x) <= self.resize_handle_width:
                    return i, self.RESIZING_LABEL_END
                else:
                    return i, self.MOVING_LABEL
        
        return -1, self.NONE
    
    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if event.button() == Qt.LeftButton:
            # Calculate frame at click position
            click_frame = self.get_frame_at_position(event.position().x())
            self.mouse_down_pos = event.position()
            self.mouse_down_frame = click_frame
            
            # Check if clicked on a label
            label_idx, action = self.find_label_at_position(event.position())
            
            if label_idx >= 0:
                # Clicked on a label
                self.selected_label_idx = label_idx
                for i, label in enumerate(self.labels):
                    label.selected = (i == label_idx)
                    
                # Emit signal for selected label
                self.label_selected.emit(self.labels[label_idx].id)
                
                # Set the action state
                self.state = action
                
                if self.state == self.MOVING_LABEL:
                    # Store initial position for dragging
                    self.drag_start_frame = self.labels[label_idx].start_frame
            else:
                # Clicked on empty space
                if event.position().y() >= self.timeline_height:
                    # Clicked in label area - start creating a new label
                    self.state = self.CREATING_LABEL
                    
                    # Clear selection
                    if self.selected_label_idx >= 0:
                        self.labels[self.selected_label_idx].selected = False
                        self.selected_label_idx = -1
                else:
                    # Clicked on timeline - change position
                    self.current_frame = click_frame
                    self.position_changed.emit(click_frame)
                    self.state = self.DRAGGING_POSITION
            
            self.update()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events."""
        if not self.mouse_down_pos:
            # Update hover state
            label_idx, _ = self.find_label_at_position(event.position())
            if label_idx != self.hover_label_idx:
                self.hover_label_idx = label_idx
                self.update()
                
                # Change cursor based on what we're hovering over
                if label_idx >= 0:
                    start_x = self.get_position_for_frame(self.labels[label_idx].start_frame)
                    end_x = self.get_position_for_frame(self.labels[label_idx].end_frame)
                    
                    # Change cursor based on position
                    if abs(event.position().x() - start_x) <= self.resize_handle_width:
                        self.setCursor(Qt.SizeHorCursor)  # Left resize
                    elif abs(event.position().x() - end_x) <= self.resize_handle_width:
                        self.setCursor(Qt.SizeHorCursor)  # Right resize
                    else:
                        self.setCursor(Qt.SizeAllCursor)  # Move
                else:
                    self.setCursor(Qt.ArrowCursor)  # Default cursor
                
            return
            
        # Calculate frame at current position
        current_frame = self.get_frame_at_position(event.position().x())
        
        if self.state == self.DRAGGING_POSITION:
            # Update position while dragging
            self.current_frame = current_frame
            self.position_changed.emit(current_frame)
            
        elif self.state == self.CREATING_LABEL:
            # Nothing to do while creating - will create on mouse release
            self.update()
            
        elif self.state == self.MOVING_LABEL and self.selected_label_idx >= 0:
            # Move the label
            label = self.labels[self.selected_label_idx]
            frame_delta = current_frame - self.mouse_down_frame
            
            new_start = self.drag_start_frame + frame_delta
            label_duration = label.duration()
            
            # Ensure label stays within bounds
            if new_start < 0:
                new_start = 0
            elif new_start + label_duration >= self.frame_count:
                new_start = self.frame_count - label_duration - 1
            
            label.start_frame = new_start
            label.end_frame = new_start + label_duration
            
        elif self.state == self.RESIZING_LABEL_START and self.selected_label_idx >= 0:
            # Resize the label start
            label = self.labels[self.selected_label_idx]
            new_start = min(current_frame, label.end_frame - 1)
            label.start_frame = max(0, new_start)
            
        elif self.state == self.RESIZING_LABEL_END and self.selected_label_idx >= 0:
            # Resize the label end
            label = self.labels[self.selected_label_idx]
            new_end = max(current_frame, label.start_frame + 1)
            label.end_frame = min(self.frame_count - 1, new_end)
        
        self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if event.button() == Qt.LeftButton and self.mouse_down_pos:
            # Calculate frame at release position
            release_frame = self.get_frame_at_position(event.position().x())
            
            if self.state == self.CREATING_LABEL:
                # Create a new label
                start_frame = min(self.mouse_down_frame, release_frame)
                end_frame = max(self.mouse_down_frame, release_frame)
                
                # Only create if it spans at least 1 frame
                if end_frame > start_frame:
                    label = Label(
                        name="New Label",
                        start_frame=start_frame,
                        end_frame=end_frame
                    )
                    
                    self.labels.append(label)
                    
                    # Select the new label
                    self.selected_label_idx = len(self.labels) - 1
                    label.selected = True
                    
                    # Emit signal
                    self.label_selected.emit(label.id)
            
            # Reset state
            self.state = self.NONE
            self.mouse_down_pos = None
            self.update()
    
    def mouseDoubleClickEvent(self, event):
        """Handle mouse double click events."""
        if event.button() == Qt.LeftButton:
            # Check if double-clicked on a label
            label_idx, _ = self.find_label_at_position(event.position())
            
            if label_idx >= 0:
                # Emit signal for double-clicked label
                self.label_double_clicked.emit(self.labels[label_idx].id)
    
    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        if self.frame_count == 0:
            return
            
        # Determine zoom direction
        delta = event.angleDelta().y()
        zoom_factor = 1.2 if delta > 0 else 1 / 1.2
        
        # Get mouse position as frame before zoom
        mouse_frame = self.get_frame_at_position(event.position().x())
        
        # Apply zoom
        old_zoom = self.zoom_level
        self.zoom_level = max(1.0, min(50.0, self.zoom_level * zoom_factor))
        
        # Adjust offset to keep mouse position at same visual point
        if old_zoom != self.zoom_level:
            # Calculate mouse position as fraction of width
            width_fraction = event.position().x() / self.width()
            
            # Calculate new visible frame count
            new_visible_frames = self.frame_count / self.zoom_level
            
            # Adjust offset to keep mouse frame at same relative position
            self.offset = mouse_frame - (new_visible_frames * width_fraction)
            
            # Clamp offset
            max_offset = self.frame_count - new_visible_frames
            self.offset = max(0, min(max_offset, self.offset))
            
            self.update()
    
    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_Delete and self.selected_label_idx >= 0:
            # Delete selected label
            label_id = self.labels[self.selected_label_idx].id
            self.remove_label(label_id)
        elif event.key() == Qt.Key_Left:
            # Move one frame back
            if self.current_frame > 0:
                self.current_frame -= 1
                self.position_changed.emit(self.current_frame)
                self.update()
        elif event.key() == Qt.Key_Right:
            # Move one frame forward
            if self.current_frame < self.frame_count - 1:
                self.current_frame += 1
                self.position_changed.emit(self.current_frame)
                self.update()
    
    def get_labels(self):
        """Get all labels as dictionaries for serialization."""
        return [label.to_dict() for label in self.labels]
    
    def get_labels_for_export(self, fps=None):
        """Get all labels formatted for export with timestamps."""
        if fps is None and hasattr(self, 'fps'):
            fps = self.fps
        elif fps is None:
            fps = 30.0  # Default assumption if FPS not provided
            
        return [label.to_export_dict(fps) for label in self.labels]
    
    def set_fps(self, fps):
        """Set the FPS value for the timeline."""
        if fps > 0:
            self.fps = fps
            self.update()
    
    def format_time_compact(self, seconds):
        """Format seconds to MM:SS or HH:MM:SS format depending on duration."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def draw_timeline_scrubber(self, painter, position_x, height):
        """Draw a more intuitive timeline scrubber handle."""
        # Draw the vertical line
        position_pen = QPen(QColor(255, 0, 0), 2)
        painter.setPen(position_pen)
        painter.drawLine(QPointF(position_x, 0), QPointF(position_x, height))
        
        # Draw scrubber handle (circle at top)
        handle_radius = 6
        handle_rect = QRectF(
            position_x - handle_radius,
            0, 
            handle_radius * 2,
            handle_radius * 2
        )
        
        # Draw glow effect
        gradient = QRadialGradient(
            position_x,
            handle_radius,
            handle_radius * 2
        )
        gradient.setColorAt(0, QColor(255, 80, 80, 180))
        gradient.setColorAt(1, QColor(255, 0, 0, 0))
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(handle_rect.adjusted(-handle_radius, -handle_radius/2, handle_radius, handle_radius/2))
        
        # Draw handle
        painter.setBrush(QColor(255, 0, 0))
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.drawEllipse(handle_rect) 