#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
import threading
from pathlib import Path
from collections import OrderedDict
import platform
import logging
import subprocess
import os
import attr
import multiprocessing as mp
from typing import Text
import shutil

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                              QSlider, QLabel, QSizePolicy, QComboBox, QGridLayout)
from PySide6.QtCore import (Qt, Signal, Slot, QThread, QMutex, QMutexLocker, 
                           QSize, QTimer, QThreadPool, QRunnable)
from PySide6.QtWidgets import QStyle
from PySide6.QtGui import QImage, QPixmap, QIcon
from PySide6.QtWidgets import QApplication

# Add these constants at the top of the file
LOW_QUALITY_SCALE = 4  # Increased from 3 to 4 for maximum speed
RESIZE_METHOD = cv2.INTER_NEAREST  # Force fastest method everywhere

def check_system_capabilities():
    """Check and report system capabilities for video processing."""
    print("\n===== Video Processing System Capabilities =====")
    
    # Check OS
    print(f"Operating System: {platform.system()} {platform.release()}")
    
    # Check CPU information
    try:
        if platform.system() == 'Windows':
            cpu_info = platform.processor()
        elif platform.system() == 'Linux':
            import subprocess
            cpu_info = subprocess.check_output("cat /proc/cpuinfo | grep 'model name' | uniq", shell=True).decode().strip()
            cpu_info = cpu_info.split(':')[1].strip() if ':' in cpu_info else cpu_info
        elif platform.system() == 'Darwin':
            import subprocess
            cpu_info = subprocess.check_output("sysctl -n machdep.cpu.brand_string", shell=True).decode().strip()
        else:
            cpu_info = "Unknown CPU"
        print(f"CPU: {cpu_info}")
    except:
        print("CPU: Detection failed")
    
    # Check OpenCV version and build information
    print(f"OpenCV Version: {cv2.__version__}")
    
    # Check for GPU capabilities
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print("CUDA Support: Available")
            for i in range(cv2.cuda.getCudaEnabledDeviceCount()):
                print(f"  CUDA Device {i}: {cv2.cuda.getDevice()}")
        else:
            print("CUDA Support: Not available")
    except:
        print("CUDA Support: Not available")
    
    # Check available backends with details
    backends = []
    if hasattr(cv2, 'CAP_FFMPEG'):
        ffmpeg_details = "Enabled" if cv2.getBuildInformation().find('FFMPEG:                      YES') >= 0 else "Disabled"
        backends.append(f"FFMPEG ({ffmpeg_details})")
    if hasattr(cv2, 'CAP_GSTREAMER'):
        gstreamer_details = "Enabled" if cv2.getBuildInformation().find('GStreamer:                  YES') >= 0 else "Disabled"
        backends.append(f"GStreamer ({gstreamer_details})")
    if hasattr(cv2, 'CAP_MSMF') and platform.system() == 'Windows':
        backends.append("Microsoft Media Foundation")
    if hasattr(cv2, 'CAP_AVFOUNDATION') and platform.system() == 'Darwin':
        backends.append("AVFoundation")
    
    print(f"Available Backends: {', '.join(backends) if backends else 'None detected'}")
    
    # Check for hardware acceleration support with comprehensive detection
    hw_accel = []
    try:
        # Check common acceleration methods
        if hasattr(cv2, 'VIDEO_ACCELERATION_D3D11'):
            hw_accel.append("Direct3D 11 (Windows)")
        if hasattr(cv2, 'VIDEO_ACCELERATION_VA'):
            hw_accel.append("VA-API (Linux)")
        if hasattr(cv2, 'VIDEO_ACCELERATION_MFX'):
            hw_accel.append("Intel Media SDK")
        if hasattr(cv2, 'VIDEO_ACCELERATION_ANY'):
            hw_accel.append("Auto-select")
            
        # Additional platform-specific checks
        if platform.system() == 'Windows':
            try:
                import ctypes
                dxgi = ctypes.windll.LoadLibrary("dxgi.dll")
                hw_accel.append("DirectX Graphics Infrastructure: Available")
            except:
                pass
        elif platform.system() == 'Linux':
            import subprocess
            try:
                vaapi = subprocess.call("ldconfig -p | grep -q libva", shell=True)
                if vaapi == 0:
                    hw_accel.append("VAAPI: Available")
            except:
                pass
        
        print(f"HW Acceleration Methods: {', '.join(hw_accel) if hw_accel else 'None detected'}")
    except Exception as e:
        print(f"Error detecting acceleration methods: {e}")
    
    print("===============================================\n")

def get_codec_info(cap):
    """Get codec information from a VideoCapture object."""
    # Get the fourcc code (codec identifier)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    
    # Convert fourcc to readable string
    fourcc_chars = bytes([
        fourcc_int & 0xFF,
        (fourcc_int >> 8) & 0xFF,
        (fourcc_int >> 16) & 0xFF,
        (fourcc_int >> 24) & 0xFF
    ]).decode('ascii', errors='replace')
    
    # Identify common codecs
    codec_name = "Unknown"
    if fourcc_chars == 'avc1' or fourcc_chars == 'H264':
        codec_name = "H.264/AVC"
    elif fourcc_chars == 'hev1' or fourcc_chars == 'HEVC':
        codec_name = "H.265/HEVC"
    elif fourcc_chars == 'vp09':
        codec_name = "VP9"
    elif fourcc_chars == 'av01':
        codec_name = "AV1"
    elif fourcc_chars == 'mp4v':
        codec_name = "MPEG-4 Visual"
    elif fourcc_chars == 'DIVX' or fourcc_chars == 'divx':
        codec_name = "DivX"
    elif fourcc_chars == 'XVID' or fourcc_chars == 'xvid':
        codec_name = "Xvid"
    elif fourcc_chars == 'mjpg' or fourcc_chars == 'MJPG':
        codec_name = "Motion JPEG"
    
    return {
        'fourcc': fourcc_chars,
        'name': codec_name
    }

class FrameLoader(QRunnable):
    """Improved frame loader with better error handling and metrics."""
    
    def __init__(self, video_reader, frame_idx, callback):
        super().__init__()
        self.video_reader = video_reader
        self.frame_idx = frame_idx
        self.callback = callback
        self.quality_mode = "low"
        # Need to track if this is being closed
        self._is_closing = False
    
    def set_quality_mode(self, mode):
        """Set the quality mode for frame loading."""
        # Always use low quality regardless of requested mode
        self.quality_mode = "low"
        
    def run(self):
        """Run the frame loading task with performance tracking."""
        start_time = time.time()
        try:
            # FIXED: Remove parent reference that doesn't exist
            # No need for mutex lock here since VideoReader has its own thread safety
            
            frame = self.video_reader.get_frame(self.frame_idx)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use EXACT same downscaling as other methods
            h, w = frame.shape[:2]
            small_w, small_h = w//LOW_QUALITY_SCALE, h//LOW_QUALITY_SCALE
            frame = cv2.resize(frame, (small_w, small_h), interpolation=RESIZE_METHOD)
            
            # Track performance - removed reference to self.parent
            end_time = time.time()
            load_time_ms = (end_time - start_time) * 1000
            
            # Simply call the callback with the loaded frame
            if self.callback and not self._is_closing:
                self.callback(self.frame_idx, frame, load_time_ms)
            
        except Exception as e:
            print(f"Error in FrameLoader for frame {self.frame_idx}: {e}")


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


class EnhancedVideoLoader:
    """Alternative video loader using imageio if available."""
    
    def __init__(self):
        self.use_imageio = False
        try:
            import imageio
            self.use_imageio = True
            self.imageio = imageio
            print("Using imageio for enhanced video loading")
        except ImportError:
            print("imageio not available, falling back to OpenCV")
            
    def load_video(self, video_path):
        """Try to load video with imageio first, fall back to OpenCV."""
        if self.use_imageio:
            try:
                reader = self.imageio.get_reader(video_path)
                return reader, True
            except Exception as e:
                print(f"imageio loading failed: {e}, falling back to OpenCV")
                return None, False
        return None, False
    
    def get_frame_count(self, reader):
        """Safely get frame count from reader with timeout protection."""
        import threading
        import time
        
        # Use a list to store results from the worker thread
        result = [None]
        error = [None]
        completed = [False]
        
        def _get_count_worker():
            try:
                # Try direct length first
                try:
                    count = len(reader)
                    
                    # Validate the count
                    if count <= 0 or count > 1000000000:
                        # Try metadata
                        meta_data = reader.get_meta_data()
                        if 'duration' in meta_data and 'fps' in meta_data and meta_data['fps'] > 0:
                            estimated = int(meta_data['duration'] * meta_data['fps'])
                            if estimated > 0:
                                result[0] = estimated
                                completed[0] = True
                                return
                            
                        # Length invalid but no metadata, use binary search
                        result[0] = self._binary_search_frame_count(reader)
                    else:
                        # Length seems valid
                        result[0] = count
                except Exception as e:
                    # Length access failed, try binary search
                    result[0] = self._binary_search_frame_count(reader)
                
                completed[0] = True
            except Exception as e:
                error[0] = str(e)
                completed[0] = True
        
        # Start worker thread
        worker = threading.Thread(target=_get_count_worker)
        worker.daemon = True
        worker.start()
        
        # Wait with timeout
        timeout = 5.0  # 5 seconds max
        start_time = time.time()
        
        while not completed[0] and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        if not completed[0]:
            print("Frame count estimation timed out")
            return 300  # Default fallback
        
        if error[0]:
            print(f"Error estimating frame count: {error[0]}")
            return 300  # Default fallback
        
        return result[0]

    def _binary_search_frame_count(self, reader, max_frames=100000):
        """Estimate frame count using binary search."""
        low = 0
        high = 1000  # Start with a reasonable guess
        
        # Test if we can access the initial high value
        try:
            reader.get_data(high-1)
            # If successful, double until we find a frame that doesn't exist
            while True:
                try:
                    reader.get_data(high-1)
                    low = high
                    high = high * 2
                    if high > max_frames:
                        # Cap at max_frames to prevent excessive searching
                        return max_frames
                except:
                    break
        except:
            # Initial high is too high, use smaller value
            high = 100
        
        # Binary search between low (valid) and high (invalid)
        while high - low > 1:
            mid = (low + high) // 2
            try:
                reader.get_data(mid-1)  # Adjust for 0-indexing
                low = mid  # This frame exists
            except:
                high = mid  # This frame doesn't exist
        
        return low  # Return highest valid frame count
    
    def get_frame(self, reader, frame_idx):
        """Get a specific frame from the video."""
        try:
            # Try to get the frame directly
            frame = reader.get_data(frame_idx)
            return frame, True
        except (IndexError, KeyError) as e:
            # Handle edge cases by trying adjacent frames
            try:
                if frame_idx > 0:
                    frame = reader.get_data(frame_idx - 1)
                    return frame, True
            except:
                pass
            return None, False


@attr.s(auto_attribs=True, eq=False, order=False)
class MediaVideo:
    """
    Video data reader optimized for reliable frame access with multiprocessing support.
    This provides a robust read-only interface on top of OpenCV's VideoCapture.
    """
    
    filename: str = attr.ib()  
    grayscale: bool = attr.ib(default=False)
    bgr: bool = attr.ib(default=True)

    _reader_ = None
    _test_frame_ = None
    _lock = None

    @property
    def __lock(self):
        if self._lock is None:
            self._lock = mp.RLock()
        return self._lock

    @property
    def __reader(self):
        # Load if not already loaded
        if self._reader_ is None:
            if not os.path.isfile(self.filename):
                raise FileNotFoundError(
                    f"Could not find video file: {self.filename}"
                )

            # Open the file with OpenCV
            self._reader_ = cv2.VideoCapture(self.filename)
            
            # Try hardware acceleration if available
            if hasattr(cv2, 'VIDEO_ACCELERATION_ANY'):
                try:
                    self._reader_.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                except:
                    pass

        # Return cached reader
        return self._reader_

    @property
    def test_frame(self):
        # Load if not already loaded
        if self._test_frame_ is None:
            # Grab a test frame to help figure things out about the video
            self._test_frame_ = self.get_frame(0, grayscale=False)

        # Return stored test frame
        return self._test_frame_

    @property
    def fps(self) -> float:
        """Returns frames per second of video."""
        fps = self.__reader.get(cv2.CAP_PROP_FPS)
        return 25.0 if fps <= 0 else fps

    @property
    def frames(self):
        """Returns total frame count."""
        frames = int(self.__reader.get(cv2.CAP_PROP_FRAME_COUNT))
        if frames <= 0:
            # Estimate frame count from duration
            self.__reader.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
            duration_ms = self.__reader.get(cv2.CAP_PROP_POS_MSEC)
            self.__reader.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frames = int((duration_ms / 1000.0) * self.fps) if duration_ms > 0 else 300
        return frames

    @property
    def channels(self):
        """Returns number of channels (1 for grayscale, 3 for RGB)."""
        if self.grayscale:
            return 1
        else:
            return self.test_frame.shape[2]

    @property
    def width(self):
        """Returns frame width."""
        return self.test_frame.shape[1]

    @property
    def height(self):
        """Returns frame height."""
        return self.test_frame.shape[0]

    @property
    def dtype(self):
        """Returns frame data type."""
        return self.test_frame.dtype

    def reset(self):
        """Reloads the video."""
        with self.__lock:
            if self._reader_ is not None:
                self._reader_.release()
            self._reader_ = None
            self._test_frame_ = None

    def get_frame(self, idx: int, grayscale: bool = None) -> np.ndarray:
        """Get a specific frame by index with reliable seeking."""
        with self.__lock:
            # Position the reader to the requested frame
            if self.__reader.get(cv2.CAP_PROP_POS_FRAMES) != idx:
                self.__reader.set(cv2.CAP_PROP_POS_FRAMES, idx)

            # Read the frame
            success, frame = self.__reader.read()

        if not success or frame is None:
            # If frame read fails, try a more robust method
            with self.__lock:
                # Reposition explicitly to the frame before
                self.__reader.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx-1))
                # Read that frame (which should work)
                _, _ = self.__reader.read()
                # Now read the actual frame we want
                success, frame = self.__reader.read()
                
                # If still failed, raise an error
                if not success or frame is None:
                    raise KeyError(f"Unable to load frame {idx}")

        # Apply grayscale conversion if needed
        if grayscale is None:
            grayscale = self.grayscale

        if grayscale:
            frame = frame[..., 0][..., None]
        elif not self.bgr:
            # Convert BGR to RGB if needed
            frame = frame[..., ::-1]

        return frame

class VideoPlayer(QWidget):
    """Custom video player widget optimized for low memory usage."""
    
    position_changed = Signal(int)  # Current frame position
    duration_changed = Signal(int)  # Total frames
    
    def __init__(self, parent=None):
        """Initialize with improved thread and cache management."""
        super().__init__(parent)
        
        # Video properties
        self.video_path = None
        self.cap = None
        self.frame_count = 0
        self.fps = 25.0  # Set default FPS to 25
        self.duration_sec = 0
        self.current_frame = 0
        self.playing = False
        self.is_playing_flag = False
        
        # Add quality lock to prevent any mode changes
        self.quality_lock = True  # New property
        
        # Always use low quality for better performance
        self.scrubbing_mode = "low"  
        
        # Frame cache
        self.frame_cache = FrameCache(max_size=200)  # Increased from 120 to 200
        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(2)  # Allow 2 parallel loads for main frames
        
        # Add separate thread pool for preloading with lower priority
        self.preload_thread_pool = QThreadPool()
        self.preload_thread_pool.setMaxThreadCount(2)  # Limit preloading to 2 threads
        
        # Playback timer
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.next_frame)
        
        # Playback speed
        self.playback_speed = 1.0
        
        # Add a flag to track if the widget is being closed
        self._is_closing = False
        
        # Add mutex protection
        self.cache_mutex = QMutex()  # For protecting frame cache access
        self.display_mutex = QMutex()  # Already exists in code
        self.loading_mutex = QMutex()  # For protecting frame loading state
        
        # Add performance metrics
        self.metrics = {
            'frames_processed': 0,
            'last_update_time': time.time(),
            'last_load_time': 0,
            'cache_hits': 0,
            'system_load': 0,
            'display_time': 0  # Add missing key
        }
        
        # Add a label to display metrics
        self.metrics_label = QLabel("Frame metrics: N/A")
        self.metrics_label.setStyleSheet("color: white; background-color: rgba(0,0,0,120); padding: 2px;")
        self.metrics_label.setAlignment(Qt.AlignRight | Qt.AlignTop)
        
        # Add scrubbing mode control
        self.scrubbing_active = False
        self.scrubbing_timer = QTimer()
        self.scrubbing_timer.setSingleShot(True)
        self.scrubbing_timer.timeout.connect(self.on_scrubbing_ended)
        
        # Pre-loading flag and behavior
        self.preload_enabled = True
        
        # Add cache cleanup timer to periodically trim it if needed
        self.cache_cleanup_timer = QTimer()
        self.cache_cleanup_timer.timeout.connect(self.cleanup_cache)
        self.cache_cleanup_timer.start(10000)  # Every 10 seconds
        
        # Add dedicated FPS label to UI
        self.fps_label = QLabel("FPS: N/A")
        self.fps_label.setStyleSheet("color: white; background-color: rgba(0,0,0,150); padding: 3px; border-radius: 3px;")
        self.fps_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        
        # Unified playback state management
        self.playback_state = {
            'playing': False,
            'segment_mode': False,
            'continuous_mode': False,
            'last_playback_mode': 'normal'  # 'normal', 'segment'
        }
        
        # Add this flag
        self.continue_after_segment = True
        
        # Create UI
        self.setup_ui()
        
        # Setup logger
        self.logger = self._setup_logger()
        
        # Adaptive cache size based on video length and available memory
        self.adaptive_cache_size = 200  # Default value
        
        # Configure improved thread pools
        self.configure_thread_pools()
        
        # Initialize metrics
        self.metrics = {
            'frames_processed': 0,
            'last_update_time': time.time(),
            'last_load_time': 0,
            'cache_hits': 0,
            'system_load': 0,
            'display_time': 0  # Add missing key
        }
        
        # Adaptive quality management
        self.performance_tracker = {
            'recent_load_times': [],  # Store recent load times
            'load_threshold': 100,    # Threshold in ms
            'quality_lock': False
        }
        
        # Setup logging
        self.logger.info("Video player initialized")
        
        # Add metrics update timer that was missing
        self.metrics_update_timer = QTimer(self)
        self.metrics_update_timer.timeout.connect(self.update_metrics)
        self.metrics_update_timer.start(500)  # Update every 500ms
    
    def _setup_logger(self):
        """Set up detailed logging for video player."""
        logger = logging.getLogger('video_player')
        logger.setLevel(logging.DEBUG)
        
        # Create file handler
        file_handler = logging.FileHandler('video_player.log')
        file_handler.setLevel(logging.WARNING)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
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
        self.video_layout.addWidget(self.fps_label, 0, 0, 1, 1, Qt.AlignLeft | Qt.AlignTop)
        
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
    
    def configure_thread_pools(self):
        """Configure thread pools with adaptive sizing."""
        # Get physical CPU count for better thread management
        import multiprocessing
        cpu_count = max(2, multiprocessing.cpu_count())
        
        # Main frame loading pool - higher priority, smaller size
        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(min(4, cpu_count // 2))
        
        # Dedicated preloading pool with lower priority
        self.preload_pool = QThreadPool()
        self.preload_pool.setMaxThreadCount(min(4, cpu_count // 2))
        
        # Set expiry timeout to keep threads alive longer
        self.thread_pool.setExpiryTimeout(10000)  # 10 seconds
        self.preload_pool.setExpiryTimeout(10000)  # 10 seconds
        
        self.logger.info(f"Thread pools configured: main={self.thread_pool.maxThreadCount()}, "
                        f"preload={self.preload_pool.maxThreadCount()}, "
                        f"CPU cores={cpu_count}")

    def calculate_adaptive_cache_size(self):
        """Calculate optimal cache size based on video length and available memory."""
        # Default cache size
        default_size = 200
        
        if not hasattr(self, 'frame_count') or self.frame_count <= 0:
            return default_size
        
        # Get available system memory
        try:
            import psutil
            avail_mem = psutil.virtual_memory().available
            # Calculate rough frame size based on dimensions and bit depth
            frame_size = (self.frame_width * self.frame_height * 3) if hasattr(self, 'frame_width') else 1920*1080*3
            
            # Use 20% of available memory at most
            max_frames = int(avail_mem * 0.2 / frame_size)
            
            # Cap at reasonable limits
            cache_size = min(max(50, max_frames), 1000)
            
            self.logger.info(f"Adaptive cache size: {cache_size} frames (based on {frame_size/1024/1024:.1f}MB per frame)")
            return cache_size
        except Exception as e:
            self.logger.warning(f"Error calculating cache size: {e}, using default={default_size}")
            return default_size

    def load_video(self, video_path):
        """Load a video file with improved caching."""
        try:
            print(f"\n===== Loading Video: {Path(video_path).name} =====")
            
            # First validate the video format
            valid, message = self.validate_video_format(video_path)
            if not valid:
                print(f"Invalid video: {message}")
                return False
            
            # Create MediaVideo reader
            try:
                self.video_reader = MediaVideo(filename=video_path, grayscale=False, bgr=True)
                print("Created MediaVideo reader successfully")
            except Exception as e:
                print(f"Error creating MediaVideo reader: {e}")
                return False
            
            # Get video properties
            self.video_path = video_path
            self.frame_count = self.video_reader.frames
            self.fps = self.video_reader.fps
            if self.fps <= 0:
                # Try additional FPS detection methods if primary fails
                self.fps = self._detect_fallback_fps(video_path)
            self.duration_sec = self.frame_count / self.fps
            self.current_frame = 0
            
            # Always use low quality mode for better performance
            self.scrubbing_mode = "low"
            
            print("\n----- Video Information -----")
            print(f"Resolution: {self.video_reader.width}x{self.video_reader.height}")
            print(f"FPS: {self.fps}")
            print(f"Frame Count: {self.frame_count}")
            print(f"Duration: {self.format_time(self.duration_sec)}")
            print("-----------------------------\n")
            
            # Update UI
            self.seek_slider.setRange(0, self.frame_count - 1)
            self.seek_slider.setValue(0)
            self.seek_slider.setEnabled(True)
            
            self.play_button.setEnabled(True)
            self.prev_frame_button.setEnabled(True)
            self.next_frame_button.setEnabled(True)
            
            self.update_time_label()
            
            # Update FPS display with actual video metadata
            self.fps_label.setText(f"Video FPS: {self.fps:.2f}")
            
            # Clear existing cache before recreating
            if hasattr(self, 'frame_cache'):
                try:
                    self.frame_cache.clear()
                except:
                    pass
            
            # Create new LRUCache with proper size
            self.adaptive_cache_size = self.calculate_adaptive_cache_size()
            self.frame_cache = LRUCache(self.adaptive_cache_size)
            self.logger.info(f"Created new frame cache with capacity: {self.adaptive_cache_size}")
            
            # Set up UI with a short delay to ensure UI is updated
            QApplication.processEvents()
            
            # Load first frame with error handling
            try:
                print("Loading first frame...")
                self.load_frame(0)
                print("First frame loaded")
            except Exception as e:
                print(f"Error loading first frame: {e}")
                import traceback
                traceback.print_exc()
            
            # Emit signals with short delay to ensure timeline has processed frame count
            self.duration_changed.emit(self.frame_count)
            QTimer.singleShot(100, lambda: self.position_changed.emit(0))
            
            # Share fps with timeline
            parent = self.parent()
            if parent and hasattr(parent, 'timeline'):
                parent.timeline.set_fps(self.fps)
                parent.timeline.set_frame_count(self.frame_count)
                print(f"Set timeline FPS to {self.fps} and frame count to {self.frame_count}")
            
            self.fast_backward_button.setEnabled(True)
            self.fast_forward_button.setEnabled(True)
            
            # Preload more frames on startup for smoother initial playback
            QTimer.singleShot(200, lambda: self.preload_frames(0, preload_count=10))
            
            return True
            
        except Exception as e:
            print(f"Error loading video: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def load_frame(self, frame_idx):
        """Load a frame with improved error handling and metrics."""
        if frame_idx < 0 or frame_idx >= self.frame_count:
            return
        
        start_time = time.time()
        
        # Check if frame is in cache first
        if frame_idx in self.frame_cache:
            frame = self.frame_cache[frame_idx]
            # Update metrics for cache hits
            if hasattr(self, 'metrics'):
                self.metrics.setdefault('cache_hits', 0)
                self.metrics['cache_hits'] += 1
            
            # Set the frame and update UI
            self.display_frame(frame)
            
            # Even for cache hits, track load time for metrics
            end_time = time.time()
            load_time_ms = (end_time - start_time) * 1000
            if hasattr(self, 'metrics'):
                self.metrics['last_load_time'] = end_time - start_time
            
            # Return success
            return True
        
        try:
            # Load and process frame
            frame = self.video_reader.get_frame(frame_idx)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Single downscale pass, no upscaling
            h, w = frame.shape[:2]
            small_w, small_h = w//LOW_QUALITY_SCALE, h//LOW_QUALITY_SCALE
            frame = cv2.resize(frame, (small_w, small_h), interpolation=RESIZE_METHOD)
            
            # Track performance
            end_time = time.time()
            load_time_ms = (end_time - start_time) * 1000
            self.track_frame_load_performance(load_time_ms)
            
            # Add frame to cache using LRUCache dictionary style
            self.frame_cache[frame_idx] = frame
            
            # Display the frame
            self.display_frame(frame)
            
            # Update metrics for direct loads
            end_time = time.time()
            load_time_ms = (end_time - start_time) * 1000
            if hasattr(self, 'metrics'):
                self.metrics['last_load_time'] = end_time - start_time
                self.metrics['frames_processed'] += 1
            
            return True
            
        except Exception as e:
            print(f"Error loading frame {frame_idx}: {e}")
            return False

    def display_frame(self, frame):
        """Display a frame using consistent upscaling."""
        with QMutexLocker(self.display_mutex):
            if self._is_closing:
                return
            
            try:
                # Start timing display operation
                display_start_time = time.time()
                
                # All frames are stored at reduced size, upscale for display
                h, w = frame.shape[:2]
                
                # Convert to QImage at the current size
                qimg = QImage(frame.data, w, h, frame.strides[0], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                
                # Always use fast transformation regardless of scrubbing state
                target_size = self.video_label.size()
                scaled_pixmap = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.FastTransformation)
                
                # Set pixmap to label
                self.video_label.setPixmap(scaled_pixmap)
                
                # Calculate and store display time
                self.metrics['display_time'] = time.time() - display_start_time
                
                # Update metrics and time label
                self.update_time_label()
                self.update_metrics_display()
                
            except Exception as e:
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
                    # FIXED: Use 'in' operator for LRUCache compatibility
                    if future_frame < self.frame_count and future_frame not in self.frame_cache:
                        try:
                            # FIXED: Use direct loading instead of FrameLoader
                            # This is a simpler approach that avoids thread pool issues
                            if self.preload_enabled:
                                QTimer.singleShot(0, lambda f=future_frame: self.preload_frame_direct(f))
                        except Exception as e:
                            print(f"Error scheduling preload for frame {future_frame}: {e}")
    
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
        
        # If we were playing before, resume playback after position change
        if self.playback_state['playing'] and not self.scrubbing_active:
            # Use short delay to allow UI to update
            QTimer.singleShot(50, self.play)
    
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
        """Handle widget close event."""
        # Mark that we're closing to prevent callbacks from accessing deleted objects
        self._is_closing = True
        
        # Clear caches
        self.frame_cache.clear()
        
        # Release MediaVideo reader
        if hasattr(self, 'video_reader') and self.video_reader is not None:
            try:
                self.video_reader.reset()
                self.video_reader = None
            except:
                pass
        
        # Call parent method
        super().closeEvent(event)

    def play(self):
        """Start playback with improved handling."""
        if self.is_playing():
            return
        
        # Always force low quality mode during playback
        self.scrubbing_mode = "low"
        
        # Set unified playback state
        self.playback_state['playing'] = True
        self.playing = True
        self.is_playing_flag = True
        
        # Update UI
        self.play_button.setText("Pause")
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        
        # Add frame processing time tracking
        self.last_frame_time = time.time()
        self.consecutive_late_frames = 0
        
        # Start playback timer if not running
        if not self.play_timer.isActive():
            # Double-check that playback speed is reasonable
            if self.playback_speed <= 0:
                self.playback_speed = 1.0
            
            # Calculate interval based on FPS and playback speed
            interval = int(1000 / (self.fps * self.playback_speed))
            
            # Ensure minimum interval to prevent UI freezing
            interval = max(10, interval)
            
            # Log the playback settings for debugging
            self.logger.info(f"Starting playback: speed={self.playback_speed}, interval={interval}ms, quality={self.scrubbing_mode}")
            
            # Create modified next_frame function with adaptive timing and segment handling
            def adaptive_next_frame():
                now = time.time()
                elapsed = now - self.last_frame_time
                target_interval = 1.0/(self.fps * self.playback_speed) if self.fps > 0 else 0.033
                
                # Check if we're running behind schedule
                if elapsed > target_interval * 1.5:
                    # Calculate how many frames to skip to catch up
                    skip_count = int(elapsed // target_interval)
                    
                    # Limit maximum skip to avoid huge jumps
                    skip_count = min(skip_count, 5)
                    
                    # Calculate next frame with skip
                    next_frame = self.current_frame + skip_count
                    
                    # Update consecutive late frames counter
                    self.consecutive_late_frames += 1
                    
                    # If consistently behind, reduce effective playback speed
                    if self.consecutive_late_frames > 5:
                        self.logger.warning("Performance issue: Reducing playback timer interval")
                        new_interval = self.play_timer.interval() * 1.2
                        self.play_timer.setInterval(int(new_interval))
                        self.consecutive_late_frames = 0
                else:
                    # Running on schedule, reset counter
                    self.consecutive_late_frames = 0
                    next_frame = self.current_frame + 1
                
                # Check for segment end condition with improved state handling
                playing_segment = self.playback_state['segment_mode'] and hasattr(self, 'playback_end_frame') and self.playback_end_frame is not None
                
                if playing_segment and next_frame >= self.playback_end_frame:
                    # We've reached the end of the segment
                    next_frame = self.playback_end_frame
                    
                    # Just clear playback range but keep playing
                    self.playback_start_frame = None
                    self.playback_end_frame = None
                    
                    # Only pause if configured not to continue
                    if not self.continue_after_segment:
                        self.pause()
                        if self.play_timer.isActive():
                            self.play_timer.stop()
                        return
                
                # Otherwise, stay within normal bounds
                next_frame = min(next_frame, self.frame_count - 1)
                
                # Update current frame only if it changed
                if next_frame != self.current_frame:
                    self.current_frame = next_frame
                    
                    # Update UI without triggering another set_position call
                    self.seek_slider.blockSignals(True)
                    self.seek_slider.setValue(next_frame)
                    self.seek_slider.blockSignals(False)
                    
                    # Load the frame (force to low quality for segment playback)
                    self.scrubbing_mode = "low"
                    self.load_frame(next_frame)
                    
                    # Emit position changed signal
                    self.position_changed.emit(next_frame)
                
                # Record time after processing frame for next calculation
                self.last_frame_time = time.time()
            
            # Properly disconnect any existing connections first
            try:
                self.play_timer.timeout.disconnect()
            except RuntimeError:
                # It's okay if there was no connection
                pass
            
            # Connect our adaptive handler
            self.play_timer.timeout.connect(adaptive_next_frame)
            
            # Start the timer
            self.play_timer.start(interval)

    def pause(self):
        """Pause video playback with state preservation."""
        # Update unified state
        self.playback_state['playing'] = False
        self.is_playing_flag = False
        self.playing = False
        
        # Update UI
        self.play_button.setText("Play")
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        
        # Stop timer
        if self.play_timer.isActive():
            self.play_timer.stop()

    def is_playing(self):
        """Return whether video is currently playing."""
        return self.is_playing_flag

    def set_playback_range(self, start_frame, end_frame):
        """Set a range of frames to play."""
        # Always ensure we're in low quality mode for smooth playback
        self.scrubbing_mode = "low"
        
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
        load_ms = self.metrics['last_load_time'] * 1000
        display_ms = self.metrics['display_time'] * 1000
        total_ms = (self.metrics['last_load_time'] + self.metrics['display_time']) * 1000
        fps_approx = 1000 / total_ms if total_ms > 0 else 0
        
        metrics_text = (
            f"Video: {self.fps:.1f} FPS | "  # Add the actual video FPS
            f"Processing: {fps_approx:.1f} FPS | "  # Clarify this is processing speed
            f"Load: {load_ms:.1f}ms | "
            f"Display: {display_ms:.1f}ms | "
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
        # Always keep low quality mode even after scrubbing ends
        self.scrubbing_mode = "low"
        
        # No need to reload the current frame in high quality
        # Just keep the current frame as is

    def set_scrubbing_quality(self, quality):
        """Set scrubbing quality with adaptive adjustment."""
        old_quality = self.scrubbing_mode
        self.scrubbing_mode = quality
        
        # Check if we should override quality based on system performance
        if hasattr(self, 'performance_tracker'):
            if len(self.performance_tracker['recent_load_times']) >= 5:
                avg_load_time = sum(self.performance_tracker['recent_load_times']) / \
                               len(self.performance_tracker['recent_load_times'])
                
                # If average load time is too high, force low quality
                if avg_load_time > self.performance_tracker['load_threshold']:
                    self.scrubbing_mode = "low"
                    self.logger.info(f"Adaptive quality override: {quality} → low (avg load: {avg_load_time:.1f}ms)")
                elif self.scrubbing_mode == "low" and avg_load_time < (self.performance_tracker['load_threshold'] * 0.7):
                    # If performance is good, allow higher quality
                    self.scrubbing_mode = quality
        
        self.logger.info(f"Scrubbing quality requested: {quality}, using: {self.scrubbing_mode}")
        return old_quality != self.scrubbing_mode

    def preload_frames(self, current_idx, preload_count=None):
        """Improved adaptive preloading strategy."""
        if not hasattr(self, 'cap') or self.cap is None:
            return
        
        # Adaptive preload window based on playback state and recent performance
        if preload_count is None:
            if self.is_playing():
                # When playing, preload more in the playback direction
                direction = 1 if not self.is_reversed else -1
                preload_range = range(current_idx + direction, 
                                     current_idx + direction * 15, 
                                     direction)
            else:
                # When paused, preload evenly around current position
                preload_range = list(range(current_idx - 10, current_idx)) + \
                               list(range(current_idx + 1, current_idx + 11))
        else:
            # Use specified preload count
            preload_range = range(current_idx + 1, current_idx + preload_count + 1)
        
        # Filter out invalid frame indices
        valid_preload = [i for i in preload_range if 0 <= i < self.frame_count and i not in self.frame_cache]
        
        # Prioritize frames - closer frames get higher priority
        valid_preload.sort(key=lambda x: abs(x - current_idx))
        
        # Limit the number of simultaneous preload requests
        max_preload = min(10, len(valid_preload))
        for idx in valid_preload[:max_preload]:
            try:
                # Lower priority for preloading
                loader = FrameLoader(self, idx, quality=self.scrubbing_mode, is_preload=True)
                loader.signals.result.connect(self.preload_worker)
                self.preload_pool.start(loader)
            except Exception as e:
                self.logger.warning(f"Preload error at frame {idx}: {e}")

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

    def update_from_timeline(self, frame):
        """Update player position from timeline scrubber."""
        # Ensure frame is within valid range 
        frame = max(0, min(frame, self.frame_count - 1))
        
        # Set current frame
        self.current_frame = frame
        
        # Block signals to avoid recursion
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(frame)
        self.seek_slider.blockSignals(False)
        
        # Load the frame
        self.load_frame(frame)
        
        # Update time label
        self.update_time_label()
        
        # Emit position changed signal, but only if not triggered by position_changed
        if not hasattr(self, '_responding_to_position_change') or not self._responding_to_position_change:
            self.position_changed.emit(frame) 

    def validate_frame_count(self):
        """Validate and correct frame count using multiple methods."""
        if self.frame_count <= 0 or self.frame_count > 1000000000:
            print(f"Invalid frame count detected: {self.frame_count}, attempting to validate")
            methods = []
            
            # Method 1: Try using CAP_PROP_FRAME_COUNT directly
            count1 = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if count1 > 0:
                methods.append(("CAP_PROP_FRAME_COUNT", count1))
            
            # Method 2: Estimate using duration and FPS
            self.cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)  # Seek to end
            duration_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Back to start
            
            if duration_ms > 0 and self.fps > 0:
                count2 = int((duration_ms / 1000.0) * self.fps)
                methods.append(("Duration-based", count2))
            
            # Method 3: Binary search for last valid frame
            try:
                count3 = self._estimate_frame_count_binary()
                if count3 > 0:
                    methods.append(("Binary search", count3))
            except Exception as e:
                print(f"Binary search estimation failed: {e}")
            
            # Choose the most reasonable count from available methods
            if methods:
                # Sort by count value (ascending) and choose the median if we have multiple values
                methods.sort(key=lambda x: x[1])
                if len(methods) >= 3:
                    # Use median for 3+ methods
                    selected = methods[len(methods)//2]
                else:
                    # With fewer methods, prefer non-zero values
                    non_zero = [m for m in methods if m[1] > 0]
                    if non_zero:
                        selected = non_zero[0]
                    else:
                        selected = ("Default", 300)  # Fallback
                    
            print(f"Selected frame count: {selected[1]} (method: {selected[0]})")
            self.frame_count = selected[1]
        else:
            # Last resort fallback
            self.frame_count = 300
            print(f"Using default frame count: {self.frame_count}")
        
        # Ensure frame count is reasonable
        self.frame_count = max(1, min(self.frame_count, 1000000))
            
        # Recalculate duration
        self.duration_sec = self.frame_count / self.fps if self.fps > 0 else 0
        
        return self.frame_count

    def _estimate_frame_count_binary(self, timeout_sec=5):
        """Estimate frame count using binary search with timeout."""
        start_time = time.time()
        
        # Start with reasonable bounds
        low = 0
        high = 100000  # Initial high guess
        
        # Test if high bound is accessible
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, high)
        ret = self.cap.grab()
        
        # If high is valid, double until we find an invalid frame
        while ret:
            low = high
            high = high * 2
            
            # Check for timeout
            if time.time() - start_time > timeout_sec:
                print("Binary search timed out, using current lower bound")
                return low
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, high)
            ret = self.cap.grab()
        
        # Binary search between low (valid) and high (invalid)
        while high - low > 1:
            mid = (low + high) // 2
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
                ret = self.cap.grab()
                
                if ret:
                    low = mid  # This frame exists
                else:
                    high = mid  # This frame doesn't exist
            except Exception as e:
                print(f"Error accessing frame: {e}")
                return low  # Default fallback
        
        # Reset to start
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Return the highest valid frame + 1 (frame count)
        return low + 1 

    def validate_video_format(self, video_path):
        """Validate video file format and codec compatibility."""
        print(f"\n----- Validating Video Format: {os.path.basename(video_path)} -----")
        
        # Step 1: Check if file exists and has non-zero size
        if not os.path.exists(video_path):
            print("Error: File does not exist")
            return False, "File does not exist"
        
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            print("Error: File is empty (0 bytes)")
            return False, "File is empty"
        
        print(f"File size: {file_size/1024/1024:.2f} MB")
        
        # Step 2: Try to open with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: OpenCV cannot open this file")
            return False, "OpenCV cannot open this file"
        
        # Step 3: Get basic properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Step 4: Check if properties are valid
        if width <= 0 or height <= 0:
            print(f"Warning: Invalid dimensions: {width}x{height}")
        else:
            print(f"Dimensions: {width}x{height}")
        
        if fps <= 0:
            print(f"Warning: Invalid FPS: {fps}")
        else:
            print(f"FPS: {fps}")
        
        if frame_count <= 0:
            print(f"Warning: Invalid frame count: {frame_count}")
        else:
            print(f"Frame count: {frame_count}")
        
        # Step 5: Try to read the first frame
        ret, frame = cap.read()
        first_frame_ok = ret and frame is not None and frame.size > 0
        if first_frame_ok:
            print("First frame read successfully")
        else:
            print("Error: Failed to read first frame")
        
        # Step 6: Try to read a frame from the middle
        if frame_count > 10:
            mid_point = frame_count // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_point)
            ret, frame = cap.read()
            mid_frame_ok = ret and frame is not None and frame.size > 0
            if mid_frame_ok:
                print("Middle frame read successfully")
            else:
                print("Warning: Failed to read middle frame")
        else:
            mid_frame_ok = first_frame_ok
        
        # Step 7: Get codec information
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_chars = ''.join([chr((fourcc_int >> 8*i) & 0xFF) for i in range(4)])
        print(f"Codec: {fourcc_chars}")
        
        # Step 8: Determine result
        cap.release()
        
        is_valid = first_frame_ok
        message = "Valid video file" if is_valid else "Invalid video file"
        
        if not mid_frame_ok and frame_count > 10:
            message += " (seeking may be unreliable)"
        
        print(f"Validation result: {message}")
        print("-------------------------------------\n")
        
        return is_valid, message 

    def cleanup_cache(self):
        """Periodically clean up the cache to prevent memory bloat."""
        with QMutexLocker(self.cache_mutex):
            # FIX: Use capacity for LRUCache instead of max_size
            if hasattr(self.frame_cache, 'capacity'):
                # For LRUCache
                if len(self.frame_cache.cache) > self.frame_cache.capacity * 0.9:
                    # Only keep the most recently used frames
                    oldest_keys = sorted(self.frame_cache.lru.items(), key=lambda x: x[1])[:len(self.frame_cache.cache)//2]
                    for k, _ in oldest_keys:
                        if k in self.frame_cache.cache:
                            del self.frame_cache.cache[k]
                            del self.frame_cache.lru[k]
            elif hasattr(self.frame_cache, 'max_size'):
                # For original FrameCache if still in use
                if len(self.frame_cache.cache) > self.frame_cache.max_size * 0.9:
                    # Only keep the most recently used frames
                    self.frame_cache.cache = OrderedDict(
                        list(self.frame_cache.cache.items())[-self.frame_cache.max_size:]
                    )

    def _detect_fallback_fps(self, video_path):
        """Try multiple methods to detect FPS if main method fails."""
        try:
            # Method 1: Try FFmpeg/FFprobe directly
            if shutil.which('ffprobe'):
                cmd = [
                    'ffprobe', '-v', '0', '-of', 'csv=p=0', 
                    '-select_streams', 'v:0', '-show_entries', 
                    'stream=r_frame_rate', video_path
                ]
                output = subprocess.check_output(cmd).decode().strip()
                if '/' in output:
                    num, den = map(int, output.split('/'))
                    if den > 0:
                        return num / den
            
            # Method 2: Try alternate OpenCV approach
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                # Try to read several frames and calculate FPS
                frame_times = []
                for _ in range(10):
                    start_time = time.time()
                    ret = cap.grab()
                    if not ret:
                        break
                    frame_times.append(time.time() - start_time)
                cap.release()
                
                if frame_times:
                    avg_time = sum(frame_times) / len(frame_times)
                    return 1.0 / avg_time if avg_time > 0 else 25.0
        
        except Exception as e:
            print(f"Error detecting FPS: {e}")
        
        # Default fallback
        return 25.0 

    def on_segment_completed(self):
        """Handle completion of a segment without stopping playback."""
        # Only clear segment boundaries
        self.playback_start_frame = None
        self.playback_end_frame = None
        self.playback_state['segment_mode'] = False
        
        # DON'T stop playback - continue from current position
        if self.playback_state['last_playback_mode'] == 'continuous':
            # Continue playing in continuous mode
            self.playback_state['continuous_mode'] = True
        else:
            # Stop only if we were in a dedicated segment playback
            self.pause()
        
        # Log completion
        self.logger.info(f"Completed segment playback, continuing in {self.playback_state['last_playback_mode']} mode")

    def set_position(self, frame):
        """Set position with playback state preservation."""
        previous_state = self.playback_state['playing']
        
        # Calculate jump size 
        jump_size = abs(frame - self.current_frame)
        
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
        
        # If we were playing before, resume playback after position change
        if previous_state and not self.scrubbing_active:
            # Use short delay to allow UI to update
            QTimer.singleShot(50, self.play)

    def set_playback_range(self, start_frame, end_frame):
        """Set a range of frames to play."""
        # Always ensure we're in low quality mode for smooth playback
        self.scrubbing_mode = "low"
        
        self.playback_start_frame = start_frame
        self.playback_end_frame = end_frame
        self.current_frame = start_frame
        
        # Update UI
        self.seek_slider.setValue(start_frame)
        self.update_time_label()
        self.load_frame(start_frame)
        
        # Signal
        self.position_changed.emit(start_frame)

    def update_metrics(self):
        """Update performance metrics with improved cache hit tracking."""
        current_time = time.time()
        elapsed = current_time - self.metrics['last_update_time']
        
        # Only update if enough time has passed to get meaningful data
        if elapsed >= 0.5:  # Update metrics every half second
            frames_processed = self.metrics['frames_processed']
            cache_hits = self.metrics.get('cache_hits', 0)  # Track cache hits
            total_frames = frames_processed + cache_hits
            
            # Calculate processing FPS with error handling
            if elapsed > 0 and total_frames > 0:
                fps = total_frames / elapsed
                # Update display with both metrics
                self.metrics_label.setText(
                    f"Load: {self.metrics['last_load_time']*1000:.1f}ms | "
                    f"Processing: {fps:.1f} FPS | "
                    f"Cache hit: {(cache_hits/total_frames)*100:.1f}% ({cache_hits}/{total_frames})"
                )
            else:
                # Safe fallback when no frames processed
                self.metrics_label.setText("Load: --ms | Processing: --FPS")
            
            # Reset counters
            self.metrics['last_update_time'] = current_time
            self.metrics['frames_processed'] = 0
            self.metrics['cache_hits'] = 0

    def track_frame_load_performance(self, load_time_ms):
        """Track frame loading performance for adaptive quality decisions."""
        if not hasattr(self, 'performance_tracker'):
            return
        
        # Keep a rolling window of recent load times
        self.performance_tracker['recent_load_times'].append(load_time_ms)
        if len(self.performance_tracker['recent_load_times']) > 10:
            self.performance_tracker['recent_load_times'].pop(0)
        
        # Check if we need to adjust quality based on load times
        if len(self.performance_tracker['recent_load_times']) >= 5:
            avg_load_time = sum(self.performance_tracker['recent_load_times']) / \
                           len(self.performance_tracker['recent_load_times'])
            
            # Log significant changes in performance
            if avg_load_time > self.performance_tracker['load_threshold'] * 1.5:
                self.logger.warning(f"Performance degrading: avg load time {avg_load_time:.1f}ms")
            elif avg_load_time < self.performance_tracker['load_threshold'] * 0.5:
                self.logger.info(f"Performance improving: avg load time {avg_load_time:.1f}ms")

    def preload_worker(self, frame_idx):
        """Handle preloaded frames with LRUCache compatibility."""
        try:
            if not self._is_closing and frame_idx is not None:
                # Use direct dictionary-style assignment for LRUCache
                if hasattr(self, 'frame_cache') and frame_idx not in self.frame_cache:
                    frame = self.video_reader.get_frame(frame_idx)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Use EXACT same downscaling as other methods
                    h, w = frame.shape[:2]
                    small_w, small_h = w//LOW_QUALITY_SCALE, h//LOW_QUALITY_SCALE
                    frame = cv2.resize(frame, (small_w, small_h), interpolation=RESIZE_METHOD)
                    
                    # Add to cache using LRUCache dictionary style
                    self.frame_cache[frame_idx] = frame
        except Exception as e:
            self.logger.warning(f"Error in preload worker for frame {frame_idx}: {e}")

    def preload_frame_direct(self, frame_idx):
        """Preload a frame directly without using thread pool."""
        try:
            if frame_idx not in self.frame_cache and frame_idx < self.frame_count:
                frame = self.video_reader.get_frame(frame_idx)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Use EXACT same downscaling as other methods
                h, w = frame.shape[:2]
                small_w, small_h = w//LOW_QUALITY_SCALE, h//LOW_QUALITY_SCALE
                frame = cv2.resize(frame, (small_w, small_h), interpolation=RESIZE_METHOD)
                
                # Add to cache using LRUCache dictionary style
                self.frame_cache[frame_idx] = frame
        except Exception as e:
            print(f"Error in direct preload for frame {frame_idx}: {e}")


# Add a new LRUCache class with better performance
class LRUCache:
    """Improved LRU cache with performance optimizations."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.lru = {}
        self.counter = 0
        self._lock = QMutex()
    
    def __contains__(self, key):
        return key in self.cache
    
    def __getitem__(self, key):
        with QMutexLocker(self._lock):
            self.lru[key] = self.counter
            self.counter += 1
            return self.cache[key]
    
    def __setitem__(self, key, value):
        with QMutexLocker(self._lock):
            if len(self.cache) >= self.capacity and key not in self.cache:
                # Find the least recently used item
                oldest_key = min(self.lru.items(), key=lambda x: x[1])[0]
                del self.cache[oldest_key]
                del self.lru[oldest_key]
            
            self.cache[key] = value
            self.lru[key] = self.counter
            self.counter += 1
    
    def clear(self):
        with QMutexLocker(self._lock):
            self.cache.clear()
            self.lru.clear()
            self.counter = 0
    
    def __len__(self):
        return len(self.cache)

