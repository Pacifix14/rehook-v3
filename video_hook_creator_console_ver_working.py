import openai
import subprocess
import os
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import glob

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

def extract_audio():
    print("Extracting audio...")
    subprocess.run(["ffmpeg", "-y", "-i", VIDEO_FILE, "-vn", "-acodec", "libmp3lame", AUDIO_FILE])

def transcribe_audio():
    print("Transcribing with Whisper...")
    
    with open(AUDIO_FILE, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="srt"
        )

    with open(SRT_FILE, "w") as f:
        f.write(transcript)

def find_best_hooks():
    print("Asking GPT-4 for best hooks...")
    with open(SRT_FILE, "r") as f:
        srt_text = f.read()

    prompt = f"""
    I have a video transcript in SRT format. Please analyze the ENTIRE transcript and find the best possible hooks for the video.

    IMPORTANT:
    - Analyze the entire video transcript, not just the beginning
    - DO NOT suggest the original hook/intro of the video - find hooks from other parts of the content
    - Each hook MUST follow these rules:
      * Be between 5 and 12 words (MUST be at least 5 words)
      * Start with either:
        - A verb (action word)
        - A bold statement
        - A question
      * Create urgency, tension, or curiosity
      * Avoid generic intros like "Hi, I'm..." or "In this video..."
    - Each hook should be a complete, coherent thought that makes sense when read alone
    - Each hook MUST be between 2 and 5 seconds in duration - this is a strict requirement
    - Each hook should be different and unique
    - The hooks should make viewers stop scrolling and pay attention

    CRITICAL FORMAT REQUIREMENTS:
    - Your response MUST start with "Maximum potential hooks found: X" where X is the number of hooks
    - For each hook, you MUST follow this EXACT format:
      Hook 1:
      Start time: HH:MM:SS.mmm
      End time: HH:MM:SS.mmm
      Text: "exact hook text here"
    - DO NOT include any hooks that are shorter than 2 seconds or longer than 5 seconds
    - DO NOT include any hooks that are shorter than 5 words
    - If a potential hook is too short or too long, DO NOT include it in your response
    - DO NOT include the original hook/intro of the video

    Example format:
    Maximum potential hooks found: 3

    Hook 1:
    Start time: 00:00:05.000
    End time: 00:00:08.000
    Text: "What if I told you this simple trick could change everything?"

    Hook 2:
    Start time: 00:00:12.000
    End time: 00:00:15.000
    Text: "This mistake is costing you thousands every month"

    Hook 3:
    Start time: 00:00:20.000
    End time: 00:00:23.000
    Text: "Watch what happens when we try this impossible challenge"

    SRT:
    {srt_text}
    """

    chat_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a video hook expert that creates attention-grabbing openings. Analyze the entire video content and identify potential hooks that meet the criteria. You MUST follow the exact format specified in the prompt. CRITICAL: DO NOT suggest any hooks that are shorter than 2 seconds or longer than 5 seconds, DO NOT suggest hooks shorter than 5 words, and DO NOT suggest the original hook/intro of the video. If a potential hook would be too short, too long, or is the original hook, you must exclude it from your response."},
            {"role": "user", "content": prompt}
        ]
    )

    gpt_output = chat_response.choices[0].message.content
    print("\nGPT-4 suggestions:\n", gpt_output)
    return gpt_output

def parse_timestamps(gpt_output):
    try:
        # Get the maximum potential hooks
        max_hooks_match = re.search(r"Maximum potential hooks found:\s*(\d+)", gpt_output)
        if not max_hooks_match:
            raise ValueError("Could not find 'Maximum potential hooks found:' in GPT response")
        
        max_hooks = int(max_hooks_match.group(1))
        print(f"\nMaximum potential hooks identified in video: {max_hooks}")
        
        # Parse each hook
        hooks = []
        for hook_number in range(1, max_hooks + 1):
            print(f"Parsing timestamps for hook {hook_number}...")
            
            # Look for the hook section
            hook_section = re.search(
                rf"Hook {hook_number}:\s*Start time:\s*(\d{{2}}:\d{{2}}:\d{{2}}[.,]\d{{3}}).*?End time:\s*(\d{{2}}:\d{{2}}:\d{{2}}[.,]\d{{3}}).*?Text:\s*\"(.*?)\"",
                gpt_output,
                re.DOTALL
            )
            
            if not hook_section:
                raise ValueError(f"Could not find properly formatted hook {hook_number} in GPT response")
            
            start_time = hook_section.group(1).replace(',', '.')
            end_time = hook_section.group(2).replace(',', '.')
            hook_text = hook_section.group(3).strip()
            
            # Calculate duration for logging only
            duration = timestamp_to_seconds(end_time) - timestamp_to_seconds(start_time)
            print(f"Extracting hook {hook_number}: {start_time} to {end_time} (duration: {duration:.2f} seconds)")
            print(f"Hook text: {hook_text}")
            hooks.append((start_time, end_time))
        
        return hooks
    except Exception as e:
        print(f"\nError parsing GPT response: {str(e)}")
        print("\nGPT response was:")
        print(gpt_output)
        raise

def timestamp_to_seconds(timestamp):
    """Convert HH:MM:SS.mmm timestamp to seconds"""
    hours, minutes, seconds = timestamp.split(':')
    seconds, milliseconds = seconds.split('.')
    total_seconds = (int(hours) * 3600) + (int(minutes) * 60) + int(seconds) + (int(milliseconds) / 1000)
    return total_seconds

def extract_video_segments(start_time, end_time, output_file):
    print(f"Extracting hook to {output_file}...")
    duration = timestamp_to_seconds(end_time) - timestamp_to_seconds(start_time)
    
    subprocess.run([
        "ffmpeg", "-y", "-i", VIDEO_FILE,
        "-ss", start_time,
        "-t", str(duration),
        "-c:v", "libx264", "-c:a", "aac",
        "-preset", "fast",
        "-avoid_negative_ts", "1",
        output_file
    ])

def reencode_segments():
    print("Re-encoding for concat...")
    # Use the same encoding settings for both segments
    encoding_params = [
        "-c:v", "libx264", "-c:a", "aac",
        "-preset", "fast",
        "-crf", "23",  # Constant quality factor
        "-movflags", "+faststart",  # Enable fast start for web playback
        "-avoid_negative_ts", "1"  # Avoid negative timestamps
    ]
    
    subprocess.run(["ffmpeg", "-y", "-i", HOOK_CLIP_1] + encoding_params + [TMP_HOOK_ENCODED_1])
    subprocess.run(["ffmpeg", "-y", "-i", HOOK_CLIP_2] + encoding_params + [TMP_HOOK_ENCODED_2])
    subprocess.run(["ffmpeg", "-y", "-i", MAIN_CLIP] + encoding_params + [TMP_MAIN_ENCODED])

def create_video_with_hook(hook_number, start_time, end_time):
    # Extract the hook
    hook_file = f"hook_{hook_number}.mp4"
    print(f"Extracting hook {hook_number}...")
    extract_video_segments(start_time, end_time, hook_file)
    
    # Create a 0.25-second blank transition in portrait orientation
    transition_file = f"transition_{hook_number}.mp4"
    print(f"Creating blank transition for hook {hook_number}...")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "color=c=black:s=720x1280:d=0.25",  # Changed to portrait dimensions
        "-f", "lavfi",
        "-i", "anullsrc=r=48000:cl=stereo",
        "-t", "0.25",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-ac", "2",
        "-preset", "medium",
        "-crf", "18",
        "-movflags", "+faststart",
        transition_file
    ])
    
    # Create final video with hook, blank transition, and original video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output = f"final_with_hook_{hook_number}_{timestamp}.mp4"
    
    print(f"Creating final video {hook_number}...")
    # Use the filter_complex approach with scale filter to maintain portrait orientation
    subprocess.run([
        "ffmpeg", "-y",
        "-i", hook_file,
        "-i", transition_file,
        "-i", VIDEO_FILE,
        "-filter_complex", 
        "[0:v]scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2[v0];"
        "[1:v]scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2[v1];"
        "[2:v]scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2[v2];"
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
        final_output
    ])
    
    # Get the full transcript
    with open(SRT_FILE, "r") as f:
        full_transcript = f.read()
    
    # Get the hook text
    hook_text = None
    for line in full_transcript.split('\n'):
        if start_time in line and end_time in line:
            hook_text = line.strip()
            break
    
    # Generate caption using GPT-4 based on full transcript
    print(f"Generating caption for hook {hook_number}...")
    caption_prompt = f"""
    Create a viral TikTok/Reels caption in Brendan Kane's style based on this video transcript and hook:

    Hook text: {hook_text}

    Full transcript:
    {full_transcript}

    The caption should:
    1. Start with the hook text
    2. Create curiosity and engagement
    3. Include a call to action
    4. Use strategic emojis
    5. Include relevant hashtags
    6. Be concise and attention-grabbing
    7. Follow Brendan Kane's viral content principles
    8. DO NOT use quotation marks anywhere in the caption
    9. Reference the main content of the video
    10. Create urgency to watch the full video

    Format the caption with:
    - Strategic line breaks
    - Strategic emoji placement
    - A mix of engagement prompts and value statements
    - Relevant trending hashtags
    - References to the video's key points
    """
    
    caption_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a viral content expert specializing in creating high-engagement social media captions in Brendan Kane's style. Create captions that drive engagement, curiosity, and shares. DO NOT use quotation marks in your captions. Make sure to reference the video's content and create urgency to watch."},
            {"role": "user", "content": caption_prompt}
        ]
    )
    
    caption = caption_response.choices[0].message.content
    
    # Remove any remaining quotation marks
    caption = caption.replace('"', '').replace('"', '')
    
    # Create caption file
    caption_file = f"caption_{hook_number}_{timestamp}.txt"
    with open(caption_file, "w") as f:
        f.write(caption)
    
    # Clean up temporary files for this hook
    for temp_file in [hook_file, transition_file]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    return final_output, caption_file

def cleanup():
    # Remove any remaining temporary files
    temp_files = [
        AUDIO_FILE, SRT_FILE,
        "hook_*.mp4", "hook_encoded_*.mp4",
        "main_encoded_*.mp4", "concat_list_*.txt"
    ]
    for pattern in temp_files:
        for file in glob.glob(pattern):
            if os.path.exists(file):
                os.remove(file)

def process_video(video_path):
    """Process a video file and return the results"""
    try:
        # Set global variables for this processing session
        global VIDEO_FILE, AUDIO_FILE, SRT_FILE
        VIDEO_FILE = video_path
        AUDIO_FILE = f"audio_{os.path.basename(video_path)}.mp3"
        SRT_FILE = f"transcript_{os.path.basename(video_path)}.srt"
        
        # Process the video
        extract_audio()
        transcribe_audio()
        gpt_output = find_best_hooks()
        
        # Process each hook and create videos
        final_outputs = []
        captions = []
        hooks = parse_timestamps(gpt_output)
        
        for hook_number, (start_time, end_time) in enumerate(hooks, 1):
            final_output, caption_file = create_video_with_hook(hook_number, start_time, end_time)
            final_outputs.append(final_output)
            captions.append(caption_file)
        
        # Clean up temporary files
        cleanup()
        
        return {
            'success': True,
            'videos': final_outputs,
            'captions': captions
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def main():
    try:
        extract_audio()
        transcribe_audio()
        gpt_output = find_best_hooks()
        
        # Process each hook and create videos
        final_outputs = []
        captions = []
        hooks = parse_timestamps(gpt_output)
        
        for hook_number, (start_time, end_time) in enumerate(hooks, 1):
            final_output, caption_file = create_video_with_hook(hook_number, start_time, end_time)
            final_outputs.append(final_output)
            captions.append(caption_file)
        
        # Clean up temporary files
        cleanup()
        
        print("\n✅ Done. Final videos with hooks saved as:")
        for i, output in enumerate(final_outputs, 1):
            print(f"{i}. {output}")
        
        print("\n✅ Done. Caption files saved as:")
        for i, caption in enumerate(captions, 1):
            print(f"{i}. {caption}")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        cleanup()

if __name__ == "__main__":
    main() 