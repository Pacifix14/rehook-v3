import openai
import subprocess
import os
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import glob
import json
import time

# Load environment variables from .env file
load_dotenv()

# --- Config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

VIDEO_FILE = "your_video.mp4"  # Replace with your input video file
AUDIO_FILE = "audio.mp3"
SRT_FILE = "transcript.srt"
HOOK_CLIP_1 = "hook_1.mp4"
HOOK_CLIP_2 = "hook_2.mp4"
MAIN_CLIP = "main.mp4"
FINAL_CLIP_1 = "final_with_hook_1_{timestamp}.mp4"
FINAL_CLIP_2 = "final_with_hook_2_{timestamp}.mp4"
TMP_HOOK_ENCODED_1 = "hook_encoded_1.mp4"
TMP_HOOK_ENCODED_2 = "hook_encoded_2.mp4"
TMP_MAIN_ENCODED = "main_encoded.mp4"
CONCAT_LIST_1 = "concat_list_1.txt"
CONCAT_LIST_2 = "concat_list_2.txt"

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Global variables for file paths
video_path = "your_video.mp4"
audio_path = "audio.mp3"
srt_path = "captions/transcript.srt"

def write_status(status, progress=None, available_videos=None, available_captions=None, timestamp_prefix=None):
    """Write the current status to a file"""
    timestamp = time.time()
    
    status_data = {
        "status": status,
        "timestamp": timestamp
    }
    
    if progress is not None:
        status_data["progress"] = progress
    
    # Add available videos and captions if provided
    if available_videos is not None:
        status_data["available_videos"] = available_videos
    
    if available_captions is not None:
        status_data["available_captions"] = available_captions
    
    with open('status.json', 'w') as f:
        json.dump(status_data, f)
    
    # Also print to console for debugging
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {status}" + 
          (f" ({progress}%)" if progress is not None else "") +
          (f" - {len(available_videos or [])} videos available" if available_videos is not None else ""))

def extract_audio():
    print("one, 1")
    """Extract audio from video file"""
    write_status("Extracting audio...")
    global audio_path
    
    # Log the paths being used
    print(f"Video path: {video_path}")
    print(f"Audio path: {audio_path}")
    
    # Ensure the uploads directory exists
    os.makedirs('uploads', exist_ok=True)
    
    # Use absolute paths
    input_path = os.path.abspath(video_path)
    output_path = os.path.abspath(audio_path)
    
    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video file not found: {input_path}")
    
    # Extract audio with -y flag to automatically overwrite
    write_status("Running FFmpeg to extract audio...")
    
    command = [
        'ffmpeg', '-y',  # Add -y flag to automatically overwrite
        '-i', input_path,
        '-q:a', '0', '-map', 'a',
        output_path
    ]
    
    try:
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"FFmpeg stdout: {result.stdout}")
        print(f"FFmpeg stderr: {result.stderr}")
        write_status(f"FFmpeg completed with exit code: {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed with error: {e.stderr}")
        write_status(f"FFmpeg error: {e.stderr}")
        raise
    
    # Verify the output file was created
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Failed to create audio file: {output_path}")
    
    audio_size = os.path.getsize(output_path)
    if audio_size == 0:
        raise ValueError(f"Audio file is empty (0 bytes): {output_path}")
        
    write_status(f"Audio extraction complete. Audio file size: {audio_size} bytes")

def transcribe_audio():
    print("two, 2")
    """Transcribe audio to text using OpenAI Whisper"""
    write_status("Transcribing audio...")
    global srt_path
    
    # Log the paths being used
    print(f"Audio path: {audio_path}")
    print(f"SRT path: {srt_path}")
    
    # Verify that audio file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Check audio file size
    audio_size = os.path.getsize(audio_path)
    if audio_size == 0:
        raise ValueError(f"Audio file is empty (0 bytes): {audio_path}")
    
    write_status(f"Sending audio to OpenAI for transcription (size: {audio_size} bytes)...")
    
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="srt"
            )
        
        # Verify transcript is not empty
        if not transcript:
            raise ValueError("Received empty transcript from OpenAI")
        
        # Write transcript to file
        with open(srt_path, "w") as f:
            f.write(transcript)
        
        # Print the transcript for debugging
        print("Generated transcript:")
        print(transcript)
        
        # Verify the output file was created
        if not os.path.exists(srt_path):
            raise FileNotFoundError(f"Failed to create transcript file: {srt_path}")
        
        transcript_size = os.path.getsize(srt_path)
        if transcript_size == 0:
            raise ValueError(f"Transcript file is empty (0 bytes): {srt_path}")
            
        write_status(f"Transcription complete. Transcript file size: {transcript_size} bytes")
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        write_status(f"Error during transcription: {str(e)}")
        raise

def parse_timestamps(srt_content):
    print("three, 3")
    """Parse SRT file and extract timestamps and text"""
    hooks = []
    current_hook = None
    
    for line in srt_content.split('\n'):
        # Check for timestamp line (e.g., "00:00:00,000 --> 00:00:05,500")
        timestamp_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', line)
        if timestamp_match:
            start_time = timestamp_match.group(1)
            end_time = timestamp_match.group(2)
            current_hook = {'start': start_time, 'end': end_time, 'text': ''}
        elif line.strip() and current_hook and not line.strip().isdigit():
            # Add text to current hook
            current_hook['text'] += line.strip() + ' '
        elif not line.strip() and current_hook:
            # End of hook
            current_hook['text'] = current_hook['text'].strip()
            hooks.append(current_hook)
            current_hook = None
    
    return hooks

def find_best_hooks():
    print("four, 4")
    """Find the best hooks in the transcript using GPT-4"""
    write_status("Finding best hooks...")
    
    with open(srt_path, 'r') as f:
        srt_content = f.read()
    
    # Print the SRT content for debugging
    print("SRT content:")
    print(srt_content)
    
    hooks = parse_timestamps(srt_content)
    
    # Print parsed hooks for debugging
    print("Parsed hooks:")
    print(hooks)
    
    # Prepare the transcript for GPT-4
    transcript = "\n".join([f"{hook['start']} - {hook['end']}: {hook['text']}" for hook in hooks])
    
    # Print the prepared transcript for debugging
    print("Prepared transcript for GPT-4:")
    print(transcript)
    
    # Create the prompt for GPT-4
    prompt = f"""Based on this transcript, identify the most engaging hooks that would work well for TikTok/Reels. 
    Each hook MUST NOT exceed 5.5 seconds in duration (this is a strict requirement).
    The hook should be attention-grabbing and make viewers want to watch the full video.
    
    Transcript:
    {transcript}
    
    Please provide the hooks in the following format:
    START_TIME --> END_TIME: Hook text
    
    For example:
    00:00:10,000 --> 00:00:15,500: This is an engaging hook that makes you want to watch more!
    
    IMPORTANT:
    - Only include hooks that are 5.5 seconds or shorter
    - Exclude any hooks that exceed 5.5 seconds
    - Focus on hooks that create curiosity or emotional engagement
    - The hook should make sense on its own without context
    """
    
    system_message = """You are an expert at creating viral content hooks for TikTok and Instagram Reels.
    Your task is to identify the most engaging moments from a transcript that would work well as hooks.
    CRITICAL: DO NOT suggest any hooks that exceed 5.5 seconds in duration.
    Focus on hooks that create curiosity, emotional engagement, or present an interesting perspective."""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    # Print GPT-4 response for debugging
    print("GPT-4 response:")
    print(response.choices[0].message.content)
    
    # Parse the response to extract hooks
    best_hooks = []
    for line in response.choices[0].message.content.split('\n'):
        match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3}): (.+)', line)
        if match:
            start_time = match.group(1)
            end_time = match.group(2)
            hook_text = match.group(3).strip()
            best_hooks.append({
                'start': start_time,
                'end': end_time,
                'text': hook_text
            })
    
    # Print final hooks for debugging
    print("Final hooks:")
    print(best_hooks)
    
    write_status("Best hooks identified", {"hooks_found": len(best_hooks)})
    return best_hooks

def cleanup_temp_files():
    print("five, 5")
    """Clean up all temporary files"""
    temp_files = [
        'hook_*.mp4',
        'transition.mp4',
        'file_list.txt',
        'audio.mp3',
        'transcript.srt'
    ]
    
    for pattern in temp_files:
        for file in glob.glob(pattern):
            try:
                if os.path.exists(file):
                    os.remove(file)
            except Exception as e:
                write_status(f"Warning: Could not remove {file}: {str(e)}")

def create_video_with_hook(hook, output_path):
    print("six, 6")
    """Create a video with the hook at the beginning."""
    try:
        # Ensure output path is absolute
        output_path = os.path.abspath(output_path)
        
        # Convert timestamps from comma to period format for FFmpeg
        start_time = hook['start'].replace(',', '.')
        end_time = hook['end'].replace(',', '.')
        
        # Create absolute paths for all files
        hook_video_path = os.path.abspath(f"hook_{start_time.replace(':', '_')}.mp4")
        transition_path = os.path.abspath("transition.mp4")
        main_video_path = os.path.abspath(video_path)
        
        # Extract the hook segment with audio
        hook_cmd = [
            'ffmpeg', '-y',
            '-i', main_video_path,
            '-ss', start_time,
            '-to', end_time,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-ar', '48000',
            '-ac', '2',
            hook_video_path
        ]
        
        write_status(f"Extracting hook segment: {hook['text']}...")
        result = subprocess.run(hook_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            write_status(f"Error extracting hook: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, hook_cmd, result.stderr)
        
        # Create a 0.25-second transition with silent audio
        transition_cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi',
            '-i', 'color=c=black:s=1280x720:d=0.25',
            '-f', 'lavfi',
            '-i', 'anullsrc=r=48000:cl=stereo',
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-ar', '48000',
            '-ac', '2',
            '-shortest',
            transition_path
        ]
        
        write_status("Creating transition...")
        result = subprocess.run(transition_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            write_status(f"Error creating transition: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, transition_cmd, result.stderr)
        
        # Get original video dimensions
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'json',
            main_video_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        video_info = json.loads(result.stdout)
        width = int(video_info['streams'][0]['width'])
        height = int(video_info['streams'][0]['height'])
        
        # Determine if video is portrait or landscape
        is_portrait = height > width
        if is_portrait:
            target_width = 720
            target_height = 1280
        else:
            target_width = 1280
            target_height = 720
        
        # Create final video with hook, blank transition, and original video
        write_status("Creating final video...")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", hook_video_path,
            "-i", transition_path,
            "-i", main_video_path,
            "-filter_complex", 
            f"[0:v]scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2[v0];"
            f"[1:v]scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2[v1];"
            f"[2:v]scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2[v2];"
            "[v0][0:a][v1][1:a][v2][2:a]concat=n=3:v=1:a=1[v][a]",
            "-map", "[v]",
            "-map", "[a]",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-ac", "2",
            "-preset", "medium",
            "-crf", "18",
            "-movflags", "+faststart",
            output_path
        ], check=True, capture_output=True, text=True)
        
        # Verify output file was created
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise FileNotFoundError(f"Output video is empty: {output_path}")
        
        # Clean up temporary files
        cleanup_temp_files()
        
        write_status(f"Video created successfully: {output_path}")
        
    except Exception as e:
        write_status(f"Error creating video: {str(e)}")
        raise

def ensure_clean_directories():
    print("seven, 7")
    """Ensure all directories are clean and exist"""
    directories = ['uploads', 'videos', 'captions']
    
    for directory in directories:
        # Remove directory if it exists
        if os.path.exists(directory):
            for file in os.listdir(directory):
                try:
                    os.remove(os.path.join(directory, file))
                except Exception as e:
                    write_status(f"Warning: Could not remove {file}: {str(e)}")
        else:
            # Create directory if it doesn't exist
            os.makedirs(directory)

def cleanup_existing_videos():
    print("eight, 8")
    """Clean up all existing videos and captions"""
    try:
        # Clean up videos directory
        for file in os.listdir('videos'):
            if file.endswith('.mp4'):
                os.remove(os.path.join('videos', file))
        
        # Clean up captions directory
        for file in os.listdir('captions'):
            if file.endswith('.txt'):
                os.remove(os.path.join('captions', file))
        
        # Clean up any temporary files
        cleanup_temp_files()
        
        print("Cleaned up existing videos and captions")
    except Exception as e:
        print(f"Error cleaning up existing files: {str(e)}")

def process_video(input_video_path, num_videos="max", timestamp=None):
    print("nine, 9")
    """Process a video file and return the results"""
    try:
        # Clear any previous status
        write_status("Starting video processing...", timestamp_prefix=timestamp)
        
        # Clean up existing videos before starting new processing
        cleanup_existing_videos()
        
        # Set global variables for this processing session
        global video_path, audio_path, srt_path
        video_path = os.path.abspath(input_video_path)
        audio_path = os.path.abspath(f"audio_{timestamp}.mp3")
        srt_path = os.path.abspath(f"transcript_{timestamp}.srt")
        
        print(f"Initializing video processing with:")
        print(f"  - video_path: {video_path}")
        print(f"  - audio_path: {audio_path}")
        print(f"  - srt_path: {srt_path}")
        print(f"  - num_videos: {num_videos}")
        print(f"  - timestamp: {timestamp}")
        
        # Ensure directories exist
        for directory in ['videos', 'captions']:
            os.makedirs(directory, exist_ok=True)
        
        # Process the video
        extract_audio()
        transcribe_audio()
        hooks = find_best_hooks()
        
        if not hooks or len(hooks) == 0:
            raise ValueError("No hooks found in the video")
            
        print(f"Found {len(hooks)} hooks in the video")
        
        # Determine how many videos to generate
        max_videos = len(hooks)
        
        if num_videos != "max":
            try:
                requested_videos = int(num_videos)
                if requested_videos <= 0:
                    write_status(f"Invalid number of videos requested: {num_videos}. Using maximum.", timestamp_prefix=timestamp)
                    num_videos_to_generate = max_videos
                else:
                    num_videos_to_generate = min(requested_videos, max_videos)
                    print(f"Will generate {num_videos_to_generate} videos (requested: {requested_videos}, available: {max_videos})")
            except ValueError:
                write_status(f"Invalid number format: {num_videos}. Using maximum.", timestamp_prefix=timestamp)
                num_videos_to_generate = max_videos
        else:
            num_videos_to_generate = max_videos
            
        # Limit hooks to the requested number
        hooks = hooks[:num_videos_to_generate]
        
        # Track generated files
        final_outputs = []
        captions = []
        
        # Process each hook and create videos
        for i, hook in enumerate(hooks):
            output_path = os.path.join('videos', f'output_{timestamp}_{i+1}_{hook["start"].replace(":", "_").replace(",", "_")}.mp4')
            caption_path = os.path.join('captions', f'caption_{timestamp}_{i+1}_{hook["start"].replace(":", "_").replace(",", "_")}.txt')
            
            # Make sure paths are absolute
            output_path = os.path.abspath(output_path)
            caption_path = os.path.abspath(caption_path)
            
            print(f"Creating video {i+1} of {num_videos_to_generate} with hook: {hook['text'][:30]}...")
            create_video_with_hook(hook, output_path)
            
            print(f"Creating caption {i+1}...")
            create_caption(hook, caption_path)
            
            final_outputs.append(os.path.basename(output_path))
            captions.append(os.path.basename(caption_path))
            
            # Update status with currently available videos and captions
            progress = int((i+1)/num_videos_to_generate*100)
            write_status(
                f"Created video {i+1} of {num_videos_to_generate}", 
                progress=progress,
                available_videos=final_outputs,
                available_captions=captions,
                timestamp_prefix=timestamp
            )
        
        # Clean up temporary files
        cleanup_temp_files()
        
        # Update final status
        write_status(
            "Video processing completed successfully", 
            progress=100,
            available_videos=final_outputs,
            available_captions=captions,
            timestamp_prefix=timestamp
        )
        
        print(f"Processing completed. Generated {len(final_outputs)} videos and {len(captions)} captions.")
        
        return {
            'success': True,
            'videos': final_outputs,
            'captions': captions
        }
    except Exception as e:
        import traceback
        error_msg = f"Error processing video: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        write_status(f"Error processing video: {str(e)}")
        raise

def create_caption(hook, caption_path):
    print("ten, 10")
    """Create a caption file for a hook"""
    write_status(f"Creating caption for hook: {hook['text'][:30]}...")

    try:
        # Ensure the captions directory exists
        os.makedirs(os.path.dirname(caption_path), exist_ok=True)

        # Generate caption using GPT-4
        prompt = f"""Create a viral TikTok/Reels caption in Brendan Kane's style based on this hook:

        Hook: {hook['text']}

        Requirements:
        - Start with the hook text
        - Create curiosity about the full video
        - Include a call to action
        - Use strategic emojis
        - Add relevant hashtags
        - Keep it concise and attention-grabbing
        - DO NOT use quotation marks anywhere in the caption

        Format:
        [Hook text]
        [Engaging line about the full video]
        [Call to action]
        [Relevant hashtags]"""

        system_message = """You are an expert at creating viral social media content in Brendan Kane's style.
        Create engaging captions that drive views and engagement.
        DO NOT use quotation marks in the caption."""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        caption = response.choices[0].message.content.strip()
        # Remove any remaining quotation marks
        caption = caption.replace('"', '')

        # Write caption to file using UTF-8 encoding
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(caption)

        write_status(f"Caption created successfully: {caption_path}")
        return caption_path
    except Exception as e:
        write_status(f"Error creating caption: {str(e)}")
        raise


def main():
    try:
        extract_audio()
        transcribe_audio()
        best_hooks = find_best_hooks()
        
        # Process each hook and create videos
        final_outputs = []
        captions = []
        
        for i, hook in enumerate(best_hooks):
            write_status(f"Processing hook {i+1} of {len(best_hooks)}...", 
                        {"current": i+1, "total": len(best_hooks)})
            
            # Create video with hook
            output_video = f"output_{i+1}_{hook['start'].replace(':', '_').replace(',', '_')}.mp4"
            create_video_with_hook(hook, output_video)
            final_outputs.append(output_video)
            
            # Create caption file
            caption_file = f"caption_{i+1}_{hook['start'].replace(':', '_').replace(',', '_')}.txt"
            create_caption(hook, caption_file)
            
            captions.append(caption_file)
        
        # Clean up temporary files
        cleanup_temp_files()
        
        print("\n✅ Done. Final videos with hooks saved as:")
        for i, output in enumerate(final_outputs, 1):
            print(f"{i}. {output}")
        
        print("\n✅ Done. Caption files saved as:")
        for i, caption in enumerate(captions, 1):
            print(f"{i}. {caption}")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        cleanup_temp_files()

if __name__ == "__main__":
    main()