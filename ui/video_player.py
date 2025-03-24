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

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                              QSlider, QLabel, QSizePolicy, QComboBox, QGridLayout)
from PySide6.QtCore import (Qt, Signal, Slot, QThread, QMutex, QMutexLocker, 
                           QSize, QTimer, QThreadPool, QRunnable)
from PySide6.QtWidgets import QStyle
from PySide6.QtGui import QImage, QPixmap, QIcon
from PySide6.QtWidgets import QApplication

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
    """Worker to load frames in background."""
    
    def __init__(self, video_path, frame_idx, callback):
        super().__init__()
        self.video_path = video_path
        self.frame_idx = frame_idx
        self.callback = callback
        self.quality_mode = "low"  # Always use low quality mode for better performance
        
    def set_quality_mode(self, mode):
        """Set the quality mode for frame loading."""
        # Always maintain low quality mode regardless of requested mode
        self.quality_mode = "low"
        
    def run(self):
        """Load the frame in background."""
        start_time = time.time()
        
        try:
            # Use FFMPEG backend explicitly for better codec support
            backend = cv2.CAP_FFMPEG if hasattr(cv2, 'CAP_FFMPEG') else 0
            cap = cv2.VideoCapture(self.video_path, backend)
            
            # Try to enable hardware acceleration for decoding if available
            hw_accel_success = False
            hw_accel_methods = []
            
            # Only try acceleration methods that are available in this OpenCV build
            if hasattr(cv2, 'VIDEO_ACCELERATION_ANY'):
                hw_accel_methods.append((cv2.VIDEO_ACCELERATION_ANY, "Auto-select"))
            if hasattr(cv2, 'VIDEO_ACCELERATION_D3D11'):
                hw_accel_methods.append((cv2.VIDEO_ACCELERATION_D3D11, "D3D11"))
            if hasattr(cv2, 'VIDEO_ACCELERATION_VA'):
                hw_accel_methods.append((cv2.VIDEO_ACCELERATION_VA, "VA-API"))
            if hasattr(cv2, 'VIDEO_ACCELERATION_MFX'):
                hw_accel_methods.append((cv2.VIDEO_ACCELERATION_MFX, "MFX"))
            
            # Try each method until one works
            for method, name in hw_accel_methods:
                try:
                    cap.set(cv2.CAP_PROP_HW_ACCELERATION, method)
                    cap.set(cv2.CAP_PROP_HW_DEVICE, 0)  # Use first available device
                    if cap.isOpened():
                        hw_accel_success = True
                        # print(f"Frame loader using hardware acceleration: {name}")
                        break
                except Exception as e:
                    pass
                
            if not hw_accel_success:
                # print("Hardware acceleration not available for frame loading, using software decoding")
                pass
            
            if not cap.isOpened():
                # Try again with default backend
                cap = cv2.VideoCapture(self.video_path)
                if not cap.isOpened():
                    raise Exception(f"Could not open video file: {self.video_path}")
            
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
                print(f"Failed to retrieve frame {self.frame_idx}")
                self.callback(self.frame_idx, None)
        except Exception as e:
            print(f"Error loading frame {self.frame_idx}: {str(e)}")
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
        self.frame_cache = FrameCache(max_size=60)  # Increase cache size
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
        self.scrubbing_active = False
        self.scrubbing_timer = QTimer()
        self.scrubbing_timer.setSingleShot(True)
        self.scrubbing_timer.timeout.connect(self.on_scrubbing_ended)
        
        # Pre-loading flag and behavior
        self.preload_enabled = True
        
        # Create UI
        self.setup_ui()
        
        # Setup logging
        self.logger = self.setup_logging()
        self.logger.info("Video player initialized")
    
    def setup_logging(self):
        """Set up detailed logging for video player."""
        logger = logging.getLogger('video_player')
        logger.setLevel(logging.DEBUG)
        
        # Create file handler
        file_handler = logging.FileHandler('video_player.log')
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
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
            
            # Clear cache
            self.frame_cache.clear()
            
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
            
            return True
            
        except Exception as e:
            print(f"Error loading video: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def load_frame(self, frame_idx):
        """Load a specific frame."""
        if frame_idx < 0 or frame_idx >= self.frame_count:
            return
        
        # Always force low quality mode for better performance
        self.scrubbing_mode = "low"
        
        # Check if frame is in cache
        cached_frame = self.frame_cache.get(frame_idx)
        if cached_frame is not None:
            self.display_frame(cached_frame)
            
            # Preload next frames if enabled and we're not scrubbing
            if self.preload_enabled and not self.scrubbing_active:
                self.preload_frames(frame_idx)
            
            return
        
        # Load frame directly (in main thread for reliability)
        try:
            start_time = time.time()
            
            # Get frame from MediaVideo reader
            frame = self.video_reader.get_frame(frame_idx)
            
            # Convert BGR to RGB if needed
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply quality reduction for scrubbing if needed
            if self.scrubbing_mode == "low":
                h, w = frame.shape[:2]
                # Fast downsampling and upsampling for speed
                scale_factor = 3  # Higher value = lower quality but faster
                small_w, small_h = w//scale_factor, h//scale_factor
                frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Calculate load time
            load_time = time.time() - start_time
            
            # Add to cache
            self.frame_cache.put(frame_idx, frame)
            
            # Display
            self.display_frame(frame)
            
            # Update metrics
            self.metrics['load_time'] = load_time
            self.update_metrics_display()
            
            # Preload next frames
            if self.preload_enabled and not self.scrubbing_active:
                self.preload_frames(frame_idx)
            
        except Exception as e:
            print(f"Error loading frame {frame_idx}: {e}")
            if frame_idx > 0:
                # Try previous frame as fallback
                try:
                    self.load_frame(frame_idx - 1)
                except:
                    pass
    
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
        """Start video playback."""
        if not hasattr(self, 'video_reader') or self.video_reader is None:
            return
        
        # Always force low quality mode during playback for better performance
        self.scrubbing_mode = "low"
        
        # Set playback state
        self.playing = True
        self.is_playing_flag = True
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
            interval = int(1000 / (self.fps * self.playback_speed)) if self.fps > 0 else 33
            
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
                
                # Check for segment end condition - FIXED to handle None values safely
                playing_segment = hasattr(self, 'playback_end_frame') and self.playback_end_frame is not None
                
                if playing_segment and next_frame >= self.playback_end_frame:
                    # We've reached the end of the segment
                    self.pause()
                    # Make sure we display the exact end frame
                    next_frame = self.playback_end_frame
                    
                    # Log segment completion
                    self.logger.info(f"Completed segment playback from {self.playback_start_frame} to {self.playback_end_frame}")
                    
                    # Clear playback range settings
                    self.playback_start_frame = None
                    self.playback_end_frame = None
                    
                    # IMPORTANT: Stop the timer to prevent further callbacks
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
        # Always keep low quality mode even after scrubbing ends
        self.scrubbing_mode = "low"
        
        # No need to reload the current frame in high quality
        # Just keep the current frame as is

    def set_scrubbing_quality(self, mode):
        """Set scrubbing quality mode - always uses low quality for better performance."""
        # Force low quality mode for better performance
        actual_mode = "low"
        
        # Update log for debugging purposes
        self.logger.info(f"Scrubbing quality requested: {mode}, but using: {actual_mode}")
        
        # If we have a thread pool, update all workers
        if hasattr(self, 'thread_pool') and self.thread_pool:
            # This implementation depends on how workers are managed
            pass  # Any worker-updating code would go here

    def preload_frames(self, current_idx):
        """Preload frames to improve playback smoothness."""
        # Don't preload during scrubbing
        if self.scrubbing_active:
            return
        
        # Always ensure scrubbing_mode is low for preloading
        self.scrubbing_mode = "low"
        
        # Calculate frames to preload
        if self.playing:
            # When playing, preload frames ahead
            frames_to_preload = [i for i in range(
                current_idx + 1,
                min(current_idx + 1 + int(4 * self.playback_speed), self.frame_count)
            )]
        else:
            # When paused, preload frames in both directions
            frames_to_preload = [
                i for i in range(
                    max(0, current_idx - 5),
                    min(current_idx + 5, self.frame_count)
                ) if i != current_idx
            ]
        
        # Start a low-priority thread to preload frames
        if frames_to_preload:
            def preload_worker():
                for idx in frames_to_preload:
                    # Skip if already in cache or player is closing
                    if self._is_closing or self.frame_cache.get(idx) is not None:
                        continue
                    
                    try:
                        # Load frame and add to cache
                        frame = self.video_reader.get_frame(idx)
                        
                        # Convert BGR to RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Apply low quality for preloaded frames
                        h, w = frame.shape[:2]
                        scale_factor = 3  # Use consistent scale factor of 3 throughout
                        small_w, small_h = w//scale_factor, h//scale_factor
                        frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
                        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_NEAREST)
                        
                        self.frame_cache.put(idx, frame)
                    except:
                        # Silently ignore preload errors
                        pass
            
            # Start preload thread
            preload_task = threading.Thread(target=preload_worker)
            preload_task.daemon = True
            preload_task.start()

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