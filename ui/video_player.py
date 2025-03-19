#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
from pathlib import Path
from collections import OrderedDict

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                              QSlider, QLabel, QSizePolicy)
from PySide6.QtCore import (Qt, Signal, Slot, QThread, QMutex, QMutexLocker, 
                           QSize, QTimer, QThreadPool, QRunnable)
from PySide6.QtWidgets import QStyle
from PySide6.QtGui import QImage, QPixmap, QIcon

class FrameLoader(QRunnable):
    """Runnable for loading frames in the background."""
    
    def __init__(self, video_path, frame_idx, callback):
        super().__init__()
        self.video_path = video_path
        self.frame_idx = frame_idx
        self.callback = callback
        
    def run(self):
        """Load the frame and call the callback with the result."""
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.callback(self.frame_idx, frame)
        else:
            self.callback(self.frame_idx, None)


class FrameCache:
    """LRU cache for video frames to reduce memory usage."""
    
    def __init__(self, max_size=30):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.mutex = QMutex()
        
    def get(self, key):
        """Get a frame from the cache."""
        with QMutexLocker(self.mutex):
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
            
    def put(self, key, value):
        """Add a frame to the cache."""
        with QMutexLocker(self.mutex):
            if key in self.cache:
                # Update existing key
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove oldest item
                self.cache.popitem(last=False)
            self.cache[key] = value
            
    def clear(self):
        """Clear the cache."""
        with QMutexLocker(self.mutex):
            self.cache.clear()


class VideoPlayer(QWidget):
    """Custom video player widget optimized for low memory usage."""
    
    position_changed = Signal(int)  # Current frame position
    duration_changed = Signal(int)  # Total frames
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Video properties
        self.video_path = None
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.duration_sec = 0
        self.current_frame = 0
        self.playing = False
        self.is_playing_flag = False
        
        # Frame cache
        self.frame_cache = FrameCache(max_size=30)
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(1)  # Limit to 1 background thread
        
        # Playback timer
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.next_frame)
        
        # Create UI
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Video display
        self.video_label = QLabel("No video loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label)
        
        # Controls layout
        controls_layout = QHBoxLayout()
        
        # Play/Pause button
        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_button)
        
        # Frame navigation buttons
        self.prev_frame_button = QPushButton("<<")
        self.prev_frame_button.setEnabled(False)
        self.prev_frame_button.clicked.connect(self.previous_frame)
        controls_layout.addWidget(self.prev_frame_button)
        
        self.next_frame_button = QPushButton(">>")
        self.next_frame_button.setEnabled(False)
        self.next_frame_button.clicked.connect(self.next_frame)
        controls_layout.addWidget(self.next_frame_button)
        
        # Time/frame display
        self.time_label = QLabel("00:00:00 / 00:00:00")
        controls_layout.addWidget(self.time_label)
        
        layout.addLayout(controls_layout)
        
        # Seek slider
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setEnabled(False)
        self.seek_slider.valueChanged.connect(self.slider_value_changed)
        layout.addWidget(self.seek_slider)
    
    def load_video(self, video_path):
        """Load a video file."""
        try:
            # Open video file
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                return False
                
            # Get video properties
            self.video_path = video_path
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.duration_sec = self.frame_count / self.fps if self.fps > 0 else 0
            self.current_frame = 0
            
            # Update UI
            self.seek_slider.setRange(0, self.frame_count - 1)
            self.seek_slider.setValue(0)
            self.seek_slider.setEnabled(True)
            
            self.play_button.setEnabled(True)
            self.prev_frame_button.setEnabled(True)
            self.next_frame_button.setEnabled(True)
            
            self.update_time_label()
            
            # Clear cache
            self.frame_cache.clear()
            
            # Load first frame
            self.load_frame(0)
            
            # Signals
            self.duration_changed.emit(self.frame_count)
            self.position_changed.emit(0)
            
            # Share fps with timeline directly through parent window
            parent = self.parent()
            if parent and hasattr(parent, 'timeline'):
                parent.timeline.set_fps(self.fps)
                print(f"Set timeline FPS to {self.fps}")
            
            return True
            
        except Exception as e:
            print(f"Error loading video: {str(e)}")
            return False
            
    def load_frame(self, frame_idx):
        """Load a specific frame."""
        if frame_idx < 0 or frame_idx >= self.frame_count:
            return
            
        # Check if frame is in cache
        cached_frame = self.frame_cache.get(frame_idx)
        if cached_frame is not None:
            self.display_frame(cached_frame)
            return
            
        # Load in background
        task = FrameLoader(self.video_path, frame_idx, self.frame_loaded_callback)
        self.thread_pool.start(task)
    
    def frame_loaded_callback(self, frame_idx, frame):
        """Callback when a frame is loaded."""
        if frame is None:
            return
            
        # Add to cache
        self.frame_cache.put(frame_idx, frame)
        
        # If this is the current frame, display it
        if frame_idx == self.current_frame:
            self.display_frame(frame)
    
    def display_frame(self, frame):
        """Display a frame in the UI."""
        h, w, _ = frame.shape
        
        # Scale the frame to fit the label while maintaining aspect ratio
        label_size = self.video_label.size()
        target_size = QSize(label_size.width(), label_size.height())
        
        # Convert to QImage
        qimg = QImage(frame.data, w, h, frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        # Scale pixmap
        scaled_pixmap = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Set pixmap to label
        self.video_label.setPixmap(scaled_pixmap)
    
    @Slot()
    def toggle_play(self):
        """Toggle play/pause state."""
        if self.playing:
            self.playing = False
            self.play_timer.stop()
            self.play_button.setText("Play")
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        else:
            self.playing = True
            # Calculate interval based on FPS
            interval = int(1000 / self.fps) if self.fps > 0 else 33  # Default to ~30fps
            self.play_timer.start(interval)
            self.play_button.setText("Pause")
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
    
    @Slot()
    def next_frame(self):
        """Go to the next frame."""
        if self.current_frame < self.frame_count - 1:
            self.set_position(self.current_frame + 1)
    
    @Slot()
    def previous_frame(self):
        """Go to the previous frame."""
        if self.current_frame > 0:
            self.set_position(self.current_frame - 1)
    
    @Slot(int)
    def slider_value_changed(self, value):
        """Handle slider value change."""
        if value != self.current_frame:
            self.set_position(value)
    
    @Slot(int)
    def set_position(self, position):
        """Set the current frame position."""
        if 0 <= position < self.frame_count and position != self.current_frame:
            self.current_frame = position
            self.seek_slider.setValue(position)
            self.load_frame(position)
            self.update_time_label()
            self.position_changed.emit(position)
    
    def update_time_label(self):
        """Update the time display label."""
        current_sec = self.current_frame / self.fps if self.fps > 0 else 0
        total_sec = self.duration_sec
        
        current_time = self.format_time(current_sec)
        total_time = self.format_time(total_sec)
        
        self.time_label.setText(f"{current_time} / {total_time} (Frame: {self.current_frame+1}/{self.frame_count})")
    
    def format_time(self, seconds):
        """Format seconds to HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def closeEvent(self, event):
        """Handle close event."""
        if self.cap:
            self.cap.release()
        event.accept()

    def play(self):
        """Start video playback."""
        if not self.cap.isOpened():
            return
        
        self.is_playing_flag = True
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        
        # Start playback timer if not running
        if not self.play_timer.isActive():
            self.play_timer.start(int(1000 / self.fps))

    def pause(self):
        """Pause video playback."""
        self.is_playing_flag = False
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        
        # Stop timer
        if self.play_timer.isActive():
            self.play_timer.stop()

    def is_playing(self):
        """Return whether video is currently playing."""
        return self.is_playing_flag

    def set_playback_range(self, start_frame, end_frame):
        """Set a range of frames to play."""
        self.playback_start_frame = start_frame
        self.playback_end_frame = end_frame
        self.current_frame = start_frame
        
        # Update UI
        self.seek_slider.setValue(start_frame)
        self.update_time_label()
        self.load_frame(start_frame)
        
        # Signal
        self.position_changed.emit(start_frame) 