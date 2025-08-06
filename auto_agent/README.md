# Automated Video Editing Agent

## Overview

The Automated Video Editing Agent is an AI-powered tool that automates the video editing process. It takes a YouTube URL and natural language instructions, then creates a professionally edited video based on those specifications. The system uses advanced AI to analyze video content, select the most relevant scenes, and create a cohesive edit that matches the user's requirements.

## Features

- **YouTube Video Processing**: Automatically download and process videos from YouTube URLs
- **Natural Language Editing Instructions**: Describe your editing needs in plain English
- **AI-Powered Scene Selection**: Intelligent scene analysis and selection based on content relevance
- **Automated Timeline Creation**: Generate professional video edits with proper transitions
- **Audio Enhancement**: Improve audio quality and add background music
- **Web-Based Interface**: Easy-to-use Streamlit UI for interacting with the system
- **Downloadable Results**: Get your edited videos in MP4 format

## How It Works

The system follows a 6-step workflow:

1. **Video Ingestion & Metadata Extraction**: Downloads the video and extracts metadata including scene information
2. **Natural Language Prompt Parsing**: Interprets user instructions to determine editing requirements
3. **AI-Powered Scene Selection**: Analyzes and selects the most relevant scenes based on the editing specifications
4. **Transition Planning**: Creates a timeline with appropriate transitions between scenes
5. **Audio Enhancement**: Improves audio quality and adds background music if specified
6. **Video Rendering**: Renders the final video and provides a downloadable file

## Requirements

- Python 3.13+
- Required packages (installed via pip):
  - dotenv
  - ffmpeg-python
  - groq
  - librosa
  - opencv-python
  - pytube
  - scenedetect
  - streamlit
  - videodb
  - yt-dlp

## Environment Setup

The application requires the following environment variables to be set in a `.env` file:

```
VIDEO_DB_API_KEY=your_videodb_api_key
GROQ_API_KEY=your_groq_api_key
```

## Usage

### Web Interface

Run the Streamlit app:

```bash
streamlit run app.py
```

This will launch a web interface where you can:
1. Enter a YouTube URL
2. Provide natural language editing instructions
3. Process the video and view the results
4. Download the edited video

### Programmatic Usage

You can also use the system programmatically:

```python
from full.workflow import main_workflow

result = main_workflow(
    youtube_url="https://www.youtube.com/watch?v=example",
    user_prompt="Create a 30-second highlight reel focusing on action scenes",
    create_download=True
)

# Access the results
print(f"Status: {result['status']}")
print(f"Output URL: {result['render_result']['output_url']}")
```

## Project Structure

- `app.py`: Streamlit web interface
- `main.py`: Simple entry point for package usage
- `full/workflow.py`: End-to-end workflow implementation
- `src/`: Core functionality modules
  - `videoprocessing.py`: Video download and processing
  - `promptparsing.py`: Natural language instruction parsing
  - `scene_scorer.py`: Scene analysis and selection
  - `transition_planning.py`: Timeline and transition creation
  - `audio_enhancing.py`: Audio quality improvement
  - `video_rendering.py`: Final video rendering

## License

See the LICENSE file for details.