# Video Hook Creator

This script automatically creates an engaging hook (opening segment) for your videos using AI. It uses OpenAI's Whisper for transcription and GPT-4 to identify the most compelling segment to use as a hook.

## Prerequisites

1. Python 3.7+
2. FFmpeg installed on your system
3. OpenAI API key

## Installation

1. Install FFmpeg on your system:
   - macOS: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt-get install ffmpeg`
   - Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Edit the configuration variables at the top of `video_hook_creator.py`:
   - Set your OpenAI API key
   - Set the path to your input video file

2. Run the script:
   ```bash
   python video_hook_creator.py
   ```

The script will:
1. Extract audio from your video
2. Transcribe it using Whisper
3. Use GPT-4 to find the best hook segment
4. Create a new video with the hook at the beginning

## Output

The final video will be saved as `final_with_hook.mp4`. All temporary files will be automatically cleaned up.

## Notes

- The script looks for a 5-15 second segment that would work well as a hook
- Make sure your video has clear audio for accurate transcription
- The process may take several minutes depending on the length of your video 