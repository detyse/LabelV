# Video Label Tool

A video labeling tool built with PySide6 that allows you to mark and annotate periods of action in videos. This tool is designed for efficient video loading and low memory usage.

## Features

- Fast video loading with minimal memory footprint
- Frame-by-frame navigation and playback
- Timeline visualization with zooming support
- Create, edit, and delete labels for time periods in videos
- Organize labels by categories
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
   - Seek slider: Jump to a specific position in the video
   - Timeline: Click to jump to a position

4. Create labels by clicking and dragging on the timeline area below the video player.

5. Edit label properties in the Labels panel:
   - Name: Give your label a descriptive name
   - Category: Organize labels by category
   - Color: Customize the label color
   - Description: Add detailed notes about the label

6. Save your project using the "Save Project" button to preserve your work.

7. Export labels to JSON format using the "Export Labels" button for use in other applications.

## Keyboard Shortcuts

- Left/Right arrow keys: Move one frame backward/forward
- Delete key: Delete the selected label

## License

This project is licensed under the MIT License - see the LICENSE file for details.
