# Video Label Tool

A video labeling tool built with PySide6 that allows you to mark and annotate periods of action in videos. This tool is designed for efficient video loading and low memory usage.

## Features

- Fast video loading with minimal memory footprint
- Frame-by-frame navigation and playback
- Timeline visualization with zooming support
- Create, edit, and delete labels for time periods in videos
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
   - Choose/Scroll Mode (C): For navigating without affecting labels
   - Edit Mode (X): For creating and adjusting labels

5. Creating and editing labels:
   - Switch to Edit Mode (press X)
   - Create labels by clicking and dragging on the timeline
   - Adjust label boundaries by dragging the handles at the start or end
   - Move labels by dragging the middle section
   - Labels are automatically numbered sequentially

6. Edit label properties in the Labels panel:
   - Name: Customize the label name
   - Color: Customize the label color
   - Description: Add detailed notes about the label

7. Save your project using the "Save Project" button to preserve your work.

8. Export labels to JSON format using the "Export Labels" button for use in other applications.

## Keyboard Shortcuts

- Left/Right arrow keys: Move one frame backward/forward
- Delete key: Delete the selected label
- X: Switch to Edit Mode (for creating and adjusting labels)
- C: Switch to Choose/Scroll Mode (for navigation without affecting labels)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
