#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
from pathlib import Path
from collections import OrderedDict

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                              QSlider, QLabel, QSizePolicy, QComboBox, QGridLayout)
from PySide6.QtCore import (Qt, Signal, Slot, QThread, QMutex, QMutexLocker, 
                           QSize, QTimer, QThreadPool, QRunnable)
from PySide6.QtWidgets import QStyle
from PySide6.QtGui import QImage, QPixmap, QIcon

class FrameLoader(QRunnable):
    """Worker to load frames in background."""
    
    def __init__(self, video_path, frame_idx, callback):
        super().__init__()
        self.video_path = video_path
        self.frame_idx = frame_idx
        self.callback = callback
        self.quality_mode = "high"  # "high" or "low"
        
    def set_quality_mode(self, mode):
        """Set the quality mode for frame loading."""
        self.quality_mode = mode
        
    def run(self):
        """Load the frame in background."""
        start_time = time.time()
        
        try:
            cap = cv2.VideoCapture(self.video_path)
            
            # For faster seeking, use hardware acceleration
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            
            # Faster seeking to target frame using grab/retrieve approach
            current_pos = 0
            target_frame = self.frame_idx
            
            # If target is far away, use direct positioning first
            if target_frame > 30:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                current_pos = target_frame
            
            # Fine-tune position using grab() for precision
            if current_pos < target_frame:
                # Skip frames using grab() without decoding them
                for _ in range(target_frame - current_pos):
                    if not cap.grab():
                        break
            
            # Now retrieve the frame (decode only the target frame)
            ret, frame = cap.retrieve()
            
            # Clean up
            cap.release()
            
            if ret:
                # Convert to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # If in low quality mode, downsample the image
                if self.quality_mode == "low":
                    h, w = frame.shape[:2]
                    # Faster resize with NEAREST interpolation
                    scale_factor = 3  # Higher value = lower quality but faster
                    small_w, small_h = w//scale_factor, h//scale_factor
                    frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_NEAREST)
                
                # Calculate load time
                load_time = time.time() - start_time
                
                # Call callback with frame and load time
                self.callback(self.frame_idx, frame, load_time)
            else:
                self.callback(self.frame_idx, None)
        except Exception as e:
            print(f"Error loading frame: {str(e)}")
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
        self.frame_cache = FrameCache(max_size=60)  # Increase cache size
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(1)  # Limit to 1 background thread
        
        # Add separate thread pool for preloading with lower priority
        self.preload_thread_pool = QThreadPool()
        self.preload_thread_pool.setMaxThreadCount(1)
        
        # Playback timer
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.next_frame)
        
        # Playback speed
        self.playback_speed = 1.0
        
        # Add a flag to track if the widget is being closed
        self._is_closing = False
        
        # Add mutex for thread safety
        self.display_mutex = QMutex()
        
        # Add performance metrics
        self.metrics = {
            'load_time': 0,
            'display_time': 0,
            'total_time': 0,
            'frames_processed': 0
        }
        
        # Add a label to display metrics
        self.metrics_label = QLabel("Frame metrics: N/A")
        self.metrics_label.setStyleSheet("color: white; background-color: rgba(0,0,0,120); padding: 2px;")
        self.metrics_label.setAlignment(Qt.AlignRight | Qt.AlignTop)
        
        # Add scrubbing mode control
        self.scrubbing_mode = "high"  # "high" or "low"
        self.scrubbing_active = False
        self.scrubbing_timer = QTimer()
        self.scrubbing_timer.setSingleShot(True)
        self.scrubbing_timer.timeout.connect(self.on_scrubbing_ended)
        
        # Pre-loading flag and behavior
        self.preload_enabled = True
        
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
        
        # Create a grid layout for the video area to overlay metrics
        self.video_layout = QGridLayout()
        self.video_layout.addWidget(self.video_label, 0, 0, 1, 2)
        self.video_layout.addWidget(self.metrics_label, 0, 1, 1, 1, Qt.AlignRight | Qt.AlignTop)
        
        # Add video layout to main layout
        layout.addLayout(self.video_layout)
        
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
        
        # Add speed control buttons
        speed_layout = QHBoxLayout()
        
        speed_label = QLabel("Speed:")
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1.0x", "1.5x", "2.0x", "4.0x", "8.0x"])
        self.speed_combo.setCurrentIndex(2)  # Default to 1.0x
        self.speed_combo.currentTextChanged.connect(self.set_playback_speed)
        
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.speed_combo)
        speed_layout.addStretch()
        
        # Add to controls layout
        controls_layout.addLayout(speed_layout)
        
        # Add fast skip buttons (5 seconds forward/backward)
        self.fast_backward_button = QPushButton("<<<")
        self.fast_backward_button.setEnabled(False)
        self.fast_backward_button.clicked.connect(self.fast_backward)
        controls_layout.addWidget(self.fast_backward_button)
        
        self.fast_forward_button = QPushButton(">>>")
        self.fast_forward_button.setEnabled(False)
        self.fast_forward_button.clicked.connect(self.fast_forward)
        controls_layout.addWidget(self.fast_forward_button)
        
        # Add metrics toggle button
        self.metrics_button = QPushButton("Show Metrics")
        self.metrics_button.setCheckable(True)
        self.metrics_button.setChecked(True)
        self.metrics_button.clicked.connect(self.toggle_metrics)
        controls_layout.addWidget(self.metrics_button)
        
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
            
            # Set quality mode based on FPS
            if self.fps < 25:
                self.scrubbing_mode = "low"
                print(f"Setting low quality mode by default (FPS: {self.fps})")
            else:
                self.scrubbing_mode = "high"
            
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
            
            self.fast_backward_button.setEnabled(True)
            self.fast_forward_button.setEnabled(True)
            
            return True
            
        except Exception as e:
            print(f"Error loading video: {str(e)}")
            return False
            
    def load_frame(self, frame_idx):
        """Load a specific frame with performance metrics."""
        if frame_idx < 0 or frame_idx >= self.frame_count:
            return
            
        # Check if frame is in cache
        cached_frame = self.frame_cache.get(frame_idx)
        if cached_frame is not None:
            self.display_frame(cached_frame)
            
            # Preload next frames if enabled and we're not scrubbing
            if self.preload_enabled and not self.scrubbing_active:
                self.preload_frames(frame_idx)
            
            return
            
        # Start timing
        start_time = time.time()
        
        # Load in background
        task = FrameLoader(self.video_path, frame_idx, self.frame_loaded_callback)
        task.set_quality_mode(self.scrubbing_mode)
        self.thread_pool.start(task)
        
        # Update metrics for loading request
        self.metrics['frames_processed'] += 1
    
    def frame_loaded_callback(self, frame_idx, frame, load_time=None):
        """Callback when a frame is loaded."""
        if frame is None or self._is_closing:
            return
            
        # Add to cache
        self.frame_cache.put(frame_idx, frame)
        
        # If this is the current frame, display it
        if frame_idx == self.current_frame and not self._is_closing:
            # Update load time metric if provided by the loader
            if load_time is not None:
                self.metrics['load_time'] = load_time
            
            # Display the frame and measure time
            display_start = time.time()
            self.display_frame(frame)
            display_time = time.time() - display_start
            
            # Update display time metric
            self.metrics['display_time'] = display_time
            self.metrics['total_time'] = load_time + display_time if load_time else display_time
            
            # Update metrics display
            self.update_metrics_display()
    
    def display_frame(self, frame):
        """Display a frame in the UI."""
        # Use mutex to protect this function from concurrent access
        with QMutexLocker(self.display_mutex):
            # Check if widget is still valid
            if self._is_closing:
                return
            
            # Start timing
            display_start = time.time()
            
            # Check if we are in a valid state to display the frame
            try:
                h, w, _ = frame.shape
                
                # Scale the frame to fit the label while maintaining aspect ratio
                label_size = self.video_label.size()
                target_size = QSize(label_size.width(), label_size.height())
                
                # Convert to QImage
                qimg = QImage(frame.data, w, h, frame.strides[0], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                
                # Scale pixmap - use faster transformation when scrubbing
                if self.scrubbing_active:
                    scaled_pixmap = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.FastTransformation)
                else:
                    scaled_pixmap = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
                # Set pixmap to label
                self.video_label.setPixmap(scaled_pixmap)
                
                # Calculate display time
                display_time = time.time() - display_start
                self.metrics['display_time'] = display_time
                
                # Update time label with current frame
                self.update_time_label()
                
                # Update metrics display
                self.update_metrics_display()
                
            except (RuntimeError, AttributeError) as e:
                # Handle the case where the widget is in an invalid state
                print(f"Error displaying frame: {e}")
    
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
            # Calculate interval based on FPS and playback speed
            interval = int(1000 / (self.fps * self.playback_speed)) if self.fps > 0 else 33
            # Ensure minimum interval to prevent UI freezing
            interval = max(10, interval)
            self.play_timer.start(interval)
            self.play_button.setText("Pause")
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
    
    @Slot()
    def next_frame(self):
        """Move to the next frame."""
        if self.current_frame < self.frame_count - 1:
            # For higher speeds, skip appropriate number of frames
            if self.playback_speed > 1.0:
                frames_to_skip = round(self.playback_speed) - 1
                next_frame = min(self.frame_count - 1, self.current_frame + 1 + frames_to_skip)
            else:
                next_frame = self.current_frame + 1
            
            # Update current frame
            self.current_frame = next_frame
            
            # Update UI without triggering another set_position call
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(next_frame)
            self.seek_slider.blockSignals(False)
            
            # Load the frame
            self.load_frame(next_frame)
            
            # Emit position changed signal
            self.position_changed.emit(next_frame)
            
            # If we're running at high speed, pre-compute several frames ahead
            if self.playback_speed > 2.0:
                for i in range(1, min(int(self.playback_speed * 2), 10)):
                    future_frame = next_frame + i
                    if future_frame < self.frame_count and self.frame_cache.get(future_frame) is None:
                        # Queue pre-loading with low quality for speed
                        preload_task = FrameLoader(self.video_path, future_frame, self.preload_callback)
                        preload_task.set_quality_mode("low")  # Always use low quality for preloaded frames
                        self.preload_thread_pool.start(preload_task)
    
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
    def set_position(self, frame):
        """Set the current position to the given frame."""
        if frame == self.current_frame:
            return
            
        # Calculate jump size
        jump_size = abs(frame - self.current_frame)
        
        # For large jumps, clear the cache to avoid keeping irrelevant frames
        if jump_size > 30:
            self.frame_cache.clear()
        
        if not self.scrubbing_active:
            self.scrubbing_active = True
            self.scrubbing_mode = "low"  # Switch to low quality during scrubbing
            
        # Restart the timer - if no scrubbing for 500ms, switch back to high quality
        self.scrubbing_timer.start(500)
        
        # Update current frame
        self.current_frame = frame
        
        # Load the frame
        self.load_frame(frame)
        
        # Emit position changed signal
        self.position_changed.emit(frame)
    
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
        # Set the closing flag
        self._is_closing = True
        
        # Wait for any pending frame loaders to finish
        self.thread_pool.clear()  # Cancel pending tasks
        self.thread_pool.waitForDone(1000)  # Wait up to a second
        
        if self.cap:
            self.cap.release()
        event.accept()

    def play(self):
        """Start video playback."""
        if not self.cap or not self.cap.isOpened():
            return
        
        self.playing = True
        self.is_playing_flag = True
        self.play_button.setText("Pause")
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        
        # Start playback timer if not running
        if not self.play_timer.isActive():
            # Calculate interval based on FPS and playback speed
            interval = int(1000 / (self.fps * self.playback_speed)) if self.fps > 0 else 33
            # Ensure minimum interval to prevent UI freezing
            interval = max(10, interval)
            self.play_timer.start(interval)

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

    def set_playback_speed(self, speed_text):
        """Set the playback speed."""
        try:
            # Parse the speed value from the text (remove the 'x')
            speed = float(speed_text.rstrip('x'))
            self.playback_speed = speed
            
            # Update timer interval if playing
            if self.is_playing_flag and self.play_timer.isActive():
                self.play_timer.stop()
                # Adjust timer interval based on speed
                interval = int(1000 / (self.fps * self.playback_speed))
                # Ensure we have at least 10ms interval to prevent UI freeze
                interval = max(10, interval)
                self.play_timer.start(interval)
        except ValueError:
            # If speed text couldn't be converted to float
            pass

    def update_metrics_display(self):
        """Update the metrics display label."""
        load_ms = self.metrics['load_time'] * 1000
        display_ms = self.metrics['display_time'] * 1000
        total_ms = (self.metrics['load_time'] + self.metrics['display_time']) * 1000
        fps_approx = 1000 / total_ms if total_ms > 0 else 0
        
        metrics_text = (
            f"Load: {load_ms:.1f}ms | "
            f"Display: {display_ms:.1f}ms | "
            f"Total: {total_ms:.1f}ms | "
            f"~{fps_approx:.1f} FPS | "
            f"Mode: {'Low' if self.scrubbing_mode == 'low' else 'High'}"
        )
        
        # Make the text more visible with styling
        self.metrics_label.setText(metrics_text)
        self.metrics_label.setStyleSheet(
            "color: white; background-color: rgba(0,0,0,150); "
            "padding: 5px; border-radius: 3px; font-weight: bold;"
        )

    def on_scrubbing_ended(self):
        """Called when scrubbing ends (timer expired)."""
        self.scrubbing_active = False
        self.scrubbing_mode = "high"
        
        # Reload current frame in high quality
        self.load_frame(self.current_frame)

    def set_scrubbing_quality(self, mode):
        """Set the scrubbing quality mode."""
        if mode in ["high", "low"]:
            self.scrubbing_mode = mode
            # Clear cache to ensure frames are reloaded with new quality
            self.frame_cache.clear()
            # Reload current frame with new quality
            self.load_frame(self.current_frame)

    def preload_frames(self, current_idx):
        """Preload a few future frames to improve playback smoothness."""
        # If we're playing forward, preload next 3 frames
        if self.playing:
            for i in range(1, 4):
                next_idx = current_idx + i
                # Fixed comparison by checking if the key exists in cache
                if isinstance(next_idx, (int, float)) and next_idx < self.frame_count and self.frame_cache.get(next_idx) is None:
                    # Lower priority preload task
                    task = FrameLoader(self.video_path, next_idx, self.preload_callback)
                    task.set_quality_mode(self.scrubbing_mode)
                    self.thread_pool.start(task)
                    # Only queue one preload at a time to avoid overloading
                    break

    def preload_callback(self, frame_idx, frame, load_time=None):
        """Callback for preloaded frames - just add to cache."""
        if frame is not None and not self._is_closing:
            self.frame_cache.put(frame_idx, frame)

    def fast_forward(self):
        """Skip forward 5 seconds."""
        frames_to_skip = int(5 * self.fps)
        new_frame = min(self.frame_count - 1, self.current_frame + frames_to_skip)
        self.set_position(new_frame)

    def fast_backward(self):
        """Skip backward 5 seconds."""
        frames_to_skip = int(5 * self.fps)
        new_frame = max(0, self.current_frame - frames_to_skip)
        self.set_position(new_frame)

    def toggle_metrics(self, checked):
        """Toggle display of performance metrics."""
        self.metrics_label.setVisible(checked)
        self.metrics_button.setText("Hide Metrics" if checked else "Show Metrics") 