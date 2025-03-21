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
    label_playback_requested = Signal(int, int)  # start_frame, end_frame
    label_created = Signal(object)  # Emit the new label data
    
    # States for mouse interactions
    NONE = 0
    DRAGGING_POSITION = 1
    CREATING_LABEL = 2
    MOVING_LABEL = 3
    RESIZING_LABEL_START = 4
    RESIZING_LABEL_END = 5
    
    # Operation modes
    CHOOSE_MODE = 0  # For scrolling and navigating
    EDIT_MODE = 1    # For creating and editing labels
    
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
        
        # Current operation mode - default to CHOOSE_MODE for safer navigation
        self.current_mode = self.CHOOSE_MODE
        
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
        
        # Track management
        self.tracks = []  # Will store track occupation data
        
        # Debug mode
        self.debug_mode = False  # Set to True to show selection regions
    
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
        
        # Get unique labels and assign them to tracks to avoid overlapping
        self.assign_labels_to_tracks()
        
        # Draw each track background
        track_height = self.label_track_height
        for track_idx in range(len(self.tracks)):
            track_top = rect.top() + track_idx * track_height
            track_rect = QRectF(0, track_top, rect.width(), track_height)
            
            # Draw track background
            bg_color = QColor(55, 55, 55)
            painter.fillRect(track_rect, bg_color)
        
        # Draw labels in their assigned tracks
        for label in self.labels:
            if hasattr(label, 'track'):
                track_idx = label.track
                track_top = rect.top() + track_idx * track_height
                track_rect = QRectF(0, track_top, rect.width(), track_height)
                self.draw_label(painter, label, track_rect)
        
        # Debug drawing
        if self.debug_mode:
            painter.setPen(QPen(QColor(255, 0, 0, 120), 1))
            for label in self.labels:
                if hasattr(label, 'track'):
                    track_idx = label.track
                    track_top = rect.top() + track_idx * track_height
                    
                    # Draw label hit areas
                    label_left = self.get_position_for_frame(label.start_frame)
                    label_right = self.get_position_for_frame(label.end_frame)
                    
                    # Draw selection area
                    debug_rect = QRectF(
                        label_left - 8,
                        track_top,
                        label_right - label_left + 16,
                        track_height
                    )
                    painter.drawRect(debug_rect)
    
    def draw_label(self, painter, label, track_rect):
        """Draw a single label on the timeline."""
        start_x = self.get_position_for_frame(label.start_frame)
        end_x = self.get_position_for_frame(label.end_frame)
        
        # Skip if not visible
        if end_x < 0 or start_x > self.width():
            return
            
        # Create label rectangle
        margin = 2
        label_rect = QRectF(
            start_x,
            track_rect.top() + margin,
            max(5, end_x - start_x),  # Ensure minimum width for visibility
            track_rect.height() - 2 * margin
        )
        
        # Determine if this label is being hovered
        is_hovered = (self.hover_label_idx >= 0 and 
                     self.hover_label_idx < len(self.labels) and 
                     self.labels[self.hover_label_idx].id == label.id)
        
        # Create gradient for label background
        gradient = QLinearGradient(
            label_rect.topLeft(),
            label_rect.bottomLeft()
        )
        
        base_color = QColor(label.color)
        if label.selected:
            # Make selected labels significantly brighter for better visibility
            base_color = base_color.lighter(160)  # Increase from 140 to 160
        elif is_hovered:
            # Make hovered labels more noticeable
            base_color = base_color.lighter(125)  # Increase from 115 to 125
            
        darker_color = base_color.darker(130)
        gradient.setColorAt(0, base_color)
        gradient.setColorAt(1, darker_color)
        
        # Draw label background
        painter.setBrush(QBrush(gradient))
        
        # Draw border
        if label.selected:
            # Add stronger glow effect for selected labels
            glow_pen = QPen(QColor(255, 255, 255, 230), 3)  # Wider, more opaque
            painter.setPen(glow_pen)
            painter.drawRoundedRect(label_rect.adjusted(-1.5, -1.5, 1.5, 1.5), 4, 4)
            
            # Draw actual border
            painter.setPen(QPen(QColor(255, 255, 255), 2))
        elif is_hovered:
            painter.setPen(QPen(QColor(230, 230, 230), 1.5))
        else:
            painter.setPen(QPen(darker_color, 1))
            
        painter.drawRoundedRect(label_rect, 4, 4)
        
        # Draw resize handles
        if label.selected or is_hovered:
            # Draw start handle
            handle_color = QColor(255, 255, 255, 220) if label.selected else QColor(220, 220, 220, 180)
            
            start_handle_rect = QRectF(
                label_rect.left(), 
                label_rect.top(), 
                self.resize_handle_width, 
                label_rect.height()
            )
            painter.setPen(Qt.black)
            painter.setBrush(handle_color)
            painter.drawRect(start_handle_rect)
            
            # Draw end handle
            end_handle_rect = QRectF(
                label_rect.right() - self.resize_handle_width, 
                label_rect.top(), 
                self.resize_handle_width, 
                label_rect.height()
            )
            painter.drawRect(end_handle_rect)
            
            # Draw small arrows in the handles to indicate direction
            painter.setPen(QPen(Qt.black, 1))
            
            # Left handle arrow
            left_arrow_y = start_handle_rect.center().y()
            painter.drawLine(
                start_handle_rect.left() + 4, 
                left_arrow_y,
                start_handle_rect.right() - 2, 
                left_arrow_y
            )
            painter.drawLine(
                start_handle_rect.left() + 4, 
                left_arrow_y - 3,
                start_handle_rect.left() + 4, 
                left_arrow_y + 3
            )
            
            # Right handle arrow
            right_arrow_y = end_handle_rect.center().y()
            painter.drawLine(
                end_handle_rect.left() + 2, 
                right_arrow_y,
                end_handle_rect.right() - 4, 
                right_arrow_y
            )
            painter.drawLine(
                end_handle_rect.right() - 4, 
                right_arrow_y - 3,
                end_handle_rect.right() - 4, 
                right_arrow_y + 3
            )
        
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
            
            # Draw text with contrasting color for better visibility
            font_metrics = QFontMetrics(font)
            text_width = font_metrics.horizontalAdvance(display_text)
            
            if text_width + 10 <= label_rect.width():
                text_rect = label_rect.adjusted(5, 0, -5, 0)
                
                # Choose text color based on background brightness
                # for better contrast
                qcolor = QColor(base_color)
                brightness = (qcolor.red() * 299 + qcolor.green() * 587 + qcolor.blue() * 114) / 1000
                
                if brightness > 128:
                    painter.setPen(QColor(0, 0, 0))  # Black text on light background
                else:
                    painter.setPen(QColor(255, 255, 255))  # White text on dark background
                    
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
        # Always clear all selections first
        for label in self.labels:
            label.selected = False
        
        # Find and select the label
        self.selected_label_idx = -1
        for i, label in enumerate(self.labels):
            if label.id == label_id:
                label.selected = True
                self.selected_label_idx = i
                break
    
    def find_label_at_position(self, pos):
        """Find the label and region at the given position."""
        x, y = pos.x(), pos.y()
        
        # Calculate timeline area
        timeline_rect = QRectF(0, 0, self.width(), self.timeline_height)
        
        # Check if position is in the label tracks area
        if y >= timeline_rect.bottom() and y < self.height():
            # Get visible frame range - fix calculation
            visible_frames = self.frame_count / self.zoom_level
            start_frame = max(0, int(self.offset))
            end_frame = min(self.frame_count, int(self.offset + visible_frames))
            
            # Debug output if enabled
            if self.debug_mode:
                print(f"Mouse at: {x}, {y}, Visible frames: {start_frame}-{end_frame}")
            
            if start_frame >= end_frame:
                return -1, None
            
            # Check labels in reverse order (top-most drawn last)
            for i in range(len(self.labels) - 1, -1, -1):
                label = self.labels[i]
                
                # Skip if label is outside visible range - more lenient check
                if label.end_frame < start_frame - 10 or label.start_frame > end_frame + 10:
                    continue
                
                # Calculate track position
                track_idx = getattr(label, 'track', 0)
                track_top = timeline_rect.bottom() + track_idx * self.label_track_height
                track_rect = QRectF(0, track_top, self.width(), self.label_track_height)
                
                # Check if click is within track height with more tolerance
                if y >= track_rect.top() - 2 and y <= track_rect.bottom() + 2:
                    # Convert label frames to pixel positions
                    label_left = self.get_position_for_frame(label.start_frame)
                    label_right = self.get_position_for_frame(label.end_frame)
                    
                    # Ensure minimum width for better hit detection
                    if (label_right - label_left) < 10:
                        label_right = label_left + 10
                    
                    # Increase margins for easier selection (increase to 10)
                    label_rect = QRectF(
                        label_left - 10, 
                        track_rect.top(), 
                        label_right - label_left + 20, 
                        track_rect.height()
                    )
                    
                    # Debug visualization if enabled
                    if self.debug_mode:
                        print(f"Label {i} ({label.name}): rect={label_rect}, contains={label_rect.contains(x, y)}")
                    
                    if label_rect.contains(x, y):
                        # Detect which part was clicked
                        start_handle_rect = QRectF(
                            label_left - 10, 
                            track_rect.top(), 
                            self.resize_handle_width + 20, 
                            track_rect.height()
                        )
                        if start_handle_rect.contains(x, y):
                            return i, "start_handle"
                        
                        end_handle_rect = QRectF(
                            label_right - self.resize_handle_width - 10, 
                            track_rect.top(), 
                            self.resize_handle_width + 20, 
                            track_rect.height()
                        )
                        if end_handle_rect.contains(x, y):
                            return i, "end_handle"
                        
                        # Must be the body
                        return i, "body"
            
            # No label found
            return -1, None
        
        # Not in label area
        return -1, None
    
    def mousePressEvent(self, event):
        """Handle mouse press events."""
        # Calculate frame at click position
        click_frame = self.get_frame_at_position(event.position().x())
        
        # Store mouse position and frame for potential drag operations
        self.mouse_down_pos = event.position()
        self.mouse_down_frame = click_frame
        
        # Check if clicking on a label first, regardless of mode
        label_idx, region = self.find_label_at_position(event.position())
        
        # LEFT MOUSE BUTTON HANDLING
        if event.button() == Qt.LeftButton:
            # If we clicked on a label, prioritize selection over other actions
            if label_idx >= 0:
                # Select the label in all modes
                self.select_label_at_index(label_idx)
                
                # In CHOOSE_MODE, request playback
                if self.current_mode == self.CHOOSE_MODE:
                    label = self.labels[label_idx]
                    # Don't immediately emit playback signal - defer it
                    # self.label_playback_requested.emit(label.start_frame, label.end_frame)
                    # Don't reset state immediately
                    # self.state = self.NONE  # This prevents drag operations
                    
                    # Allow the selection to be visible first
                    self.state = self.DRAGGING_POSITION
                    # We'll emit the playback signal on mouse release instead
                else:  # EDIT_MODE
                    # Set appropriate state for label manipulation
                    if region == "start_handle":
                        self.state = self.RESIZING_LABEL_START
                        self.setCursor(Qt.SizeHorCursor)
                    elif region == "end_handle":
                        self.state = self.RESIZING_LABEL_END
                        self.setCursor(Qt.SizeHorCursor)
                    elif region == "body":
                        self.state = self.MOVING_LABEL
                        self.drag_start_frame = click_frame
                        self.setCursor(Qt.ClosedHandCursor)
            else:
                # Clicked on empty space - position seeking in both modes
                self.state = self.DRAGGING_POSITION
                self.current_frame = click_frame
                self.position_changed.emit(self.current_frame)
                
                # Deselect any selected label when clicking empty space
                if self.selected_label_idx >= 0:
                    self.labels[self.selected_label_idx].selected = False
                    self.selected_label_idx = -1
        
        # RIGHT MOUSE BUTTON HANDLING - Only for creating new labels in EDIT_MODE
        elif event.button() == Qt.RightButton and self.current_mode == self.EDIT_MODE:
            # Start creating a new label
            self.state = self.CREATING_LABEL
            self.current_frame = click_frame
            self.position_changed.emit(self.current_frame)
        
        self.update()
    
    def select_label_at_index(self, idx):
        """Select the label at the given index."""
        if idx < 0 or idx >= len(self.labels):
            return
        
        # Deselect previously selected label
        if self.selected_label_idx >= 0 and self.selected_label_idx < len(self.labels):
            self.labels[self.selected_label_idx].selected = False
        
        # Select the new label
        self.selected_label_idx = idx
        self.labels[idx].selected = True
        
        # Emit signal with label ID
        label_id = self.labels[idx].id
        self.label_selected.emit(label_id)
        
        self.update()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events."""
        if not self.mouse_down_pos:
            # Just hovering (no mouse button pressed)
            if self.current_mode == self.CHOOSE_MODE:
                # In choose mode, only show hover effects for labels
                hover_idx, _ = self.find_label_at_position(event.position())
                if hover_idx != self.hover_label_idx:
                    self.hover_label_idx = hover_idx
                    self.update()
                
                # Set cursor based on what's under it
                if hover_idx >= 0:
                    self.setCursor(Qt.PointingHandCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)
                return
            
            # In EDIT_MODE, check for hovering over labels or handles
            hover_idx, hover_region = self.find_label_at_position(event.position())
            if hover_idx != self.hover_label_idx:
                self.hover_label_idx = hover_idx
                self.update()
            
            # Set cursor based on what's under it
            if hover_idx >= 0:
                if hover_region == "start_handle" or hover_region == "end_handle":
                    self.setCursor(Qt.SizeHorCursor)
                else:  # Body
                    self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.CrossCursor)
            return
        
        # Get frame at current mouse position
        current_frame = self.get_frame_at_position(event.position().x())
        
        # Add a small drag threshold to prevent accidental moves
        if self.state == self.MOVING_LABEL:
            # Only start moving if we've moved at least 2 frames
            if abs(current_frame - self.mouse_down_frame) < 2:
                return
        
        # Handle different drag states
        if self.state == self.DRAGGING_POSITION:
            # Update current position
            self.current_frame = current_frame
            self.position_changed.emit(self.current_frame)
        
        elif self.state == self.CREATING_LABEL:
            # Update position for label preview during creation
            self.current_frame = current_frame
            self.position_changed.emit(self.current_frame)
        
        elif self.state == self.MOVING_LABEL and self.selected_label_idx >= 0:
            # Move the selected label
            label = self.labels[self.selected_label_idx]
            
            # Calculate frame delta
            frame_delta = current_frame - self.drag_start_frame
            
            # Calculate new start and end with boundary checks
            new_start = max(0, label.start_frame + frame_delta)
            new_end = min(self.frame_count - 1, label.end_frame + frame_delta)
            
            # Ensure we don't change the duration
            duration = label.end_frame - label.start_frame
            if new_end - new_start != duration:
                if new_end >= self.frame_count - 1:
                    # Hit right boundary
                    new_start = new_end - duration
                else:
                    # Hit left boundary or other constraint
                    new_end = new_start + duration
            
            # Update label positions
            label.start_frame = new_start
            label.end_frame = new_end
            
            # Update drag reference point
            self.drag_start_frame = current_frame
            
            # Update current position to follow the label
            self.current_frame = current_frame
            self.position_changed.emit(self.current_frame)
        
        elif self.state == self.RESIZING_LABEL_START and self.selected_label_idx >= 0:
            # Resize label by moving start point
            label = self.labels[self.selected_label_idx]
            new_start = min(current_frame, label.end_frame - 1)
            new_start = max(0, new_start)
            label.start_frame = new_start
            self.current_frame = new_start
            self.position_changed.emit(self.current_frame)
        
        elif self.state == self.RESIZING_LABEL_END and self.selected_label_idx >= 0:
            # Resize label by moving end point
            label = self.labels[self.selected_label_idx]
            new_end = max(current_frame, label.start_frame + 1)
            new_end = min(self.frame_count - 1, new_end)
            label.end_frame = new_end
            self.current_frame = new_end
            self.position_changed.emit(self.current_frame)
        
        self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        # Only process if we have a stored mouse down position
        if not self.mouse_down_pos:
            return
        
        # Calculate frame at release position
        release_frame = self.get_frame_at_position(event.position().x())
        
        # LEFT MOUSE BUTTON RELEASE
        if event.button() == Qt.LeftButton:
            # If we've clicked a label in CHOOSE_MODE, now play it
            if self.current_mode == self.CHOOSE_MODE and self.selected_label_idx >= 0:
                label = self.labels[self.selected_label_idx]
                # Now emit the playback signal
                self.label_playback_requested.emit(label.start_frame, label.end_frame)
        
        # RIGHT MOUSE BUTTON RELEASE
        elif event.button() == Qt.RightButton:
            if self.state == self.CREATING_LABEL:
                # Get the frame range for the new label
                start_frame = min(self.mouse_down_frame, self.current_frame)
                end_frame = max(self.mouse_down_frame, self.current_frame)
                
                # Get the selected template name from the label panel
                template_name = "New Label"
                parent = self.parent()
                while parent:
                    if hasattr(parent, 'label_panel'):
                        template_name = parent.label_panel.selected_template
                        break
                    parent = parent.parent()
                    
                # Create base name from template
                base_name = template_name.split(". ", 1)[-1] if ". " in template_name else template_name
                
                # Get the next number based on existing labels
                existing_numbers = []
                for lbl in self.labels:
                    name_parts = lbl.name.split(". ", 1)
                    if len(name_parts) > 0 and name_parts[0].isdigit():
                        try:
                            existing_numbers.append(int(name_parts[0]))
                        except ValueError:
                            pass
                
                next_number = max(existing_numbers) + 1 if existing_numbers else 1
                
                # Create label with formatted name
                label = Label(
                    name=f"{next_number}. {base_name}",
                    category="default",
                    start_frame=start_frame,
                    end_frame=end_frame
                )
                
                # Add to labels list
                self.labels.append(label)
                
                # Select the new label
                self.select_label_by_id(label.id)
                
                # Emit signal with label data
                self.label_created.emit(label.to_dict())
                
                # Update the display
                self.update()
        
        # Reset cursor to appropriate default for mode
        if self.current_mode == self.EDIT_MODE:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        
        # Reset state for all mouse buttons
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
        elif event.key() == Qt.Key_C:
            # Switch to CHOOSE mode (for scrolling/navigating)
            self.current_mode = self.CHOOSE_MODE
            self.setCursor(Qt.ArrowCursor)
            # Show status message (requires MainWindow connection)
            if hasattr(self, 'parent') and hasattr(self.parent(), 'statusBar'):
                self.parent().statusBar().showMessage("Choose/Scroll Mode: Navigate without affecting labels", 2000)
        elif event.key() == Qt.Key_X:
            # Switch to EDIT mode (for creating/modifying labels)
            self.current_mode = self.EDIT_MODE
            self.setCursor(Qt.CrossCursor)
            # Show status message (requires MainWindow connection)
            if hasattr(self, 'parent') and hasattr(self.parent(), 'statusBar'):
                self.parent().statusBar().showMessage("Edit Mode: Create and adjust labels", 2000)
    
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
        """Format seconds to MM:SS or HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
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
    
    def get_next_label_number(self, category="default"):
        """Get the next sequential number for a label in this category."""
        category_labels = [l for l in self.labels if l.category == category]
        
        # Extract existing numbers from labels
        existing_numbers = []
        for label in category_labels:
            # Try to extract the number from the label name
            # Expected format: "category-number" or just "number"
            name_parts = label.name.split('-')
            if len(name_parts) > 1 and name_parts[-1].isdigit():
                existing_numbers.append(int(name_parts[-1]))
            elif label.name.isdigit():
                existing_numbers.append(int(label.name))
        
        # Find the next available number
        if not existing_numbers:
            return 1
        return max(existing_numbers) + 1
    
    def frame_to_position(self, frame):
        """Convert frame number to x position on the timeline (alias for get_position_for_frame)."""
        return self.get_position_for_frame(frame)
    
    def assign_labels_to_tracks(self):
        """Assign labels to tracks to avoid overlapping."""
        # Sort labels by start frame for optimal track assignment
        sorted_labels = sorted(self.labels, key=lambda l: l.start_frame)
        
        # Clear existing tracks
        self.tracks = []
        
        # Assign each label to a track
        for label in sorted_labels:
            # Try to find a suitable track
            assigned = False
            for track_idx, track_end_frame in enumerate(self.tracks):
                if label.start_frame >= track_end_frame:  # Use >= not just > to prevent touching edges
                    # This track is free, use it
                    self.tracks[track_idx] = label.end_frame
                    label.track = track_idx  # Store the track index on the label
                    assigned = True
                    break
            
            if not assigned:
                # Need to create a new track
                self.tracks.append(label.end_frame)
                label.track = len(self.tracks) - 1
    
    def set_mode(self, mode):
        """Set the current timeline mode.
        
        Args:
            mode: Either CHOOSE_MODE or EDIT_MODE
        """
        if mode == self.CHOOSE_MODE or mode == self.EDIT_MODE:
            self.current_mode = mode
            
            # Reset state and cursor when changing modes
            self.state = self.NONE
            
            # Set appropriate cursor for the mode
            if mode == self.CHOOSE_MODE:
                self.setCursor(Qt.ArrowCursor)
            else:  # EDIT_MODE
                self.setCursor(Qt.CrossCursor)
            
            # Deselect any selected label when changing modes
            if self.selected_label_idx >= 0:
                self.labels[self.selected_label_idx].selected = False
                self.selected_label_idx = -1
            
            # Signal that the mode has changed
            # Find parent main window and update its UI
            parent = self.parent()
            while parent:
                if hasattr(parent, 'update_mode'):
                    parent.update_mode(mode)
                    break
                parent = parent.parent()
            
            self.update()  # Redraw the timeline with new mode settings
        else:
            print(f"Invalid mode: {mode}")
    
    def toggle_debug_mode(self):
        """Toggle debug visualization mode."""
        self.debug_mode = not self.debug_mode
        print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
        self.update()

    @Slot(str, str)
    def update_label_name(self, label_id, new_name):
        """Update the name of a label on the timeline."""
        # Find the label with the given ID and update its name
        for i, label in enumerate(self.labels):
            if label.id == label_id:
                self.labels[i].name = new_name
                # Force a repaint of the timeline
                self.update()
                break
