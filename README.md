# Video Label Tool

A video labeling tool built with PySide6 that allows you to mark and annotate periods of action in videos. This tool is designed for efficient video loading and low memory usage.

## Features

- Fast video loading with minimal memory footprint
- Frame-by-frame navigation and playback with adjustable speed
- Timeline visualization with zooming support
- Create, edit, and delete labels for time periods in videos
- Support for overlapping labels with multi-track display
- Customize label colors
- Export labels to JSON format

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/video-label-tool.git
   cd video-label-tool
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python main.py
   ```

2. Open a video file using the "Open Video" button in the toolbar.

3. Use the video player controls to navigate through the video:
   - Play/Pause: Toggle video playback
   - Previous/Next frame buttons: Move one frame at a time
   - Seek slider: Jump to a specific position
   - Timeline: Click to jump to a position

4. Two operating modes are available (toggle with keyboard shortcuts):
   - View Mode (C): For navigating and playing labeled segments
     - Clicking on a label plays that segment
     - Creating/editing labels is disabled
   - Edit Mode (X): For creating and adjusting labels
     - Create labels by clicking and dragging on the timeline
     - Adjust label boundaries by dragging the handles at the start or end
     - Move labels by dragging the middle section

5. Label format and management:
   - Labels are formatted as "1. action_name" for clarity
   - Labels are automatically assigned to separate tracks when they overlap
   - Edit label properties in the Labels panel
   - Delete labels by selecting them and pressing Delete

6. Save your project using the "Save Project" button to preserve your work.

7. Export labels to JSON format using the "Export Labels" button for use in other applications.

## Keyboard Shortcuts

- Space: Toggle play/pause
- Left/Right arrow keys: Move one frame backward/forward
- Delete key: Delete the selected label
- X: Switch to Edit Mode (for creating and adjusting labels)
- C: Switch to View Mode (for navigation and playing labeled segments)

## Playback Speed Control

The application includes playback speed control ranging from 0.25x to 8.0x normal speed. 
This allows for:
- Slow-motion analysis of critical moments (0.25x, 0.5x)
- Normal playback (1.0x)
- Fast review of lengthy content (1.5x, 2.0x, 4.0x, 8.0x)

Choose the appropriate speed from the dropdown in the video player controls.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
