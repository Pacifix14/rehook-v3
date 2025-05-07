from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from flask_cors import CORS
import os
import traceback
from werkzeug.utils import secure_filename
from video_hook_creator import process_video
import json
from datetime import datetime
import time
import hashlib

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
VIDEO_FOLDER = 'videos'
CAPTION_FOLDER = 'captions'
STATIC_FOLDER = 'static'

# Ensure all directories exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, VIDEO_FOLDER, CAPTION_FOLDER, STATIC_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER
app.config['CAPTION_FOLDER'] = CAPTION_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Log form data
        print(f"Request form data: {dict(request.form)}")
        print(f"Request files: {list(request.files.keys())}")
        
        if 'file' not in request.files:
            print("Error: No file part in request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            print("Error: No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            print(f"Error: Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the uploaded file with timestamp
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        print(f"File saved successfully: {file_path}, size: {os.path.getsize(file_path)} bytes")
        
        # Get number of videos to generate
        num_videos = request.form.get('num_videos', 'max')
        
        # Create initial status with timestamp prefix for filtering
        with open('status.json', 'w') as f:
            json.dump({
                'status': 'Processing video...',
                'timestamp': time.time(),
                'timestamp_prefix': timestamp,
                'num_videos_requested': num_videos
            }, f)
        
        # Process the video
        try:
            print(f"Starting video processing with num_videos={num_videos}, timestamp={timestamp}")
            
            result = process_video(file_path, num_videos=num_videos, timestamp=timestamp)
            print(f"Processing completed successfully: {result}")
            return jsonify(result)
        except Exception as e:
            error_msg = f"Error processing video: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            with open('status.json', 'w') as f:
                json.dump({
                    'status': f"Error: {str(e)}",
                    'timestamp': time.time(),
                    'timestamp_prefix': timestamp,
                    'num_videos_requested': num_videos
                }, f)
            return jsonify({'error': str(e)}), 500
    except Exception as e:
        error_msg = f"Error in upload route: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({'error': str(e)}), 500

@app.route('/videos/<path:filename>')
def serve_video(filename):
    try:
        filepath = os.path.join(app.config['VIDEO_FOLDER'], filename)
        filesize = os.path.getsize(filepath) if os.path.exists(filepath) else 0
        print(f"Serving video: {filename}")
        print(f"Full path: {filepath}")
        print(f"Exists: {os.path.exists(filepath)}")
        print(f"Size: {filesize} bytes")
        print(f"Directory contents: {os.listdir(app.config['VIDEO_FOLDER'])}")
        
        if os.path.exists(filepath) and filesize > 0:
            response = send_from_directory(app.config['VIDEO_FOLDER'], filename)
            response.headers['Content-Type'] = 'video/mp4'
            response.headers['Accept-Ranges'] = 'bytes'
            response.headers['Content-Length'] = str(filesize)
            return response
        else:
            print(f"Video file not found or empty: {filename}")
            return f"Video file not found or empty: {filename}", 404
    except Exception as e:
        print(f"Error serving video {filename}: {str(e)}")
        return f"Error serving video: {str(e)}", 500

@app.route('/captions/<path:filename>')
def serve_caption(filename):
    return send_from_directory(app.config['CAPTION_FOLDER'], filename)

@app.route('/status')
def status():
    try:
        if os.path.exists('status.json'):
            # Get file modification time
            file_mod_time = os.path.getmtime('status.json')
            
            # Read current status
            with open('status.json', 'r') as f:
                status_data = json.load(f)
                
            # Add available videos and captions to the status response
            videos_dir = os.path.abspath(app.config['VIDEO_FOLDER'])
            captions_dir = os.path.abspath(app.config['CAPTION_FOLDER'])
            
            print(f"Checking for videos in: {videos_dir}")
            print(f"Directory exists: {os.path.exists(videos_dir)}")
            if os.path.exists(videos_dir):
                print(f"Directory contents: {os.listdir(videos_dir)}")
            
            # Get list of available videos and captions if they exist
            available_videos = []
            available_captions = []
            
            # Check if the status indicates we're in an active processing session
            current_status = status_data.get('status', '')
            is_processing = current_status != 'Ready to process video' and not current_status.startswith('Error')
            
            # Get the timestamp from the status or existing videos
            timestamp_pattern = status_data.get('timestamp_prefix', None)
            num_videos_requested = status_data.get('num_videos_requested', 'max')
            
            if os.path.exists(videos_dir):
                # If we have a timestamp pattern, use it to filter videos
                if timestamp_pattern:
                    available_videos = [f for f in os.listdir(videos_dir) 
                                      if f.endswith('.mp4') and timestamp_pattern in f]
                    # Sort by sequence number (the number after the timestamp)
                    available_videos.sort(key=lambda x: int(x.split('_')[2]))
                # If no timestamp pattern but we're processing, get the most recent batch of videos
                elif is_processing:
                    # Get all mp4 files and sort by modification time
                    all_videos = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
                    if all_videos:
                        # Sort by modification time, newest first
                        all_videos.sort(key=lambda x: os.path.getmtime(os.path.join(videos_dir, x)), reverse=True)
                        # Get the most recent video's timestamp prefix
                        most_recent = all_videos[0]
                        # Extract timestamp prefix (format: output_YYYYMMDD_HHMMSS)
                        timestamp_prefix = '_'.join(most_recent.split('_')[:2])
                        # Get all videos with this timestamp prefix
                        available_videos = [v for v in all_videos if timestamp_prefix in v]
                        # Sort by sequence number (the number after the timestamp)
                        available_videos.sort(key=lambda x: int(x.split('_')[2]))
            
            if os.path.exists(captions_dir):
                # If we have a timestamp pattern, use it to filter captions
                if timestamp_pattern:
                    available_captions = [f for f in os.listdir(captions_dir) 
                                        if f.endswith('.txt') and timestamp_pattern in f]
                    # Sort by sequence number (the number after the timestamp)
                    available_captions.sort(key=lambda x: int(x.split('_')[2]))
                # If no timestamp pattern but we're processing, get the most recent batch of captions
                elif is_processing:
                    # Get all txt files and sort by modification time
                    all_captions = [f for f in os.listdir(captions_dir) if f.endswith('.txt')]
                    if all_captions:
                        # Sort by modification time, newest first
                        all_captions.sort(key=lambda x: os.path.getmtime(os.path.join(captions_dir, x)), reverse=True)
                        # Get the most recent caption's timestamp prefix
                        most_recent = all_captions[0]
                        # Extract timestamp prefix (format: caption_YYYYMMDD_HHMMSS)
                        timestamp_prefix = '_'.join(most_recent.split('_')[:2])
                        # Get all captions with this timestamp prefix
                        available_captions = [c for c in all_captions if timestamp_prefix in c]
                        # Sort by sequence number (the number after the timestamp)
                        available_captions.sort(key=lambda x: int(x.split('_')[2]))
            
            # Add videos and captions to status data
            status_data['available_videos'] = available_videos
            status_data['available_captions'] = available_captions
            
            print(f"Status response: {json.dumps(status_data, indent=2)}")
            
            # Compute ETag based on content
            etag = hashlib.md5(json.dumps(status_data).encode()).hexdigest()
            
            # Check if client has latest version
            if_none_match = request.headers.get('If-None-Match')
            if if_none_match and if_none_match == etag:
                return "", 304  # Not Modified
                
            # Check if status is too old (more than 5 minutes)
            current_time = time.time()
            if current_time - file_mod_time > 300:  # 5 minutes in seconds
                status_data = {
                    "status": "Ready to process video", 
                    "timestamp": current_time,
                    "timestamp_prefix": None,
                    "available_videos": [],
                    "available_captions": []
                }
            
            # Set ETag and return data
            response = app.response_class(
                response=json.dumps(status_data),
                status=200,
                mimetype='application/json'
            )
            response.headers['ETag'] = etag
            response.headers['Cache-Control'] = 'no-cache'
            return response
        else:
            # Create initial status file if it doesn't exist
            initial_status = {
                "status": "Ready to process video", 
                "timestamp": time.time(),
                "timestamp_prefix": None,
                "available_videos": [],
                "available_captions": []
            }
            
            with open('status.json', 'w') as f:
                json.dump(initial_status, f)
            return jsonify(initial_status)
    except Exception as e:
        print(f"Error in status route: {str(e)}")
        # Return a proper initial state instead of an error
        return jsonify({
            "status": "Ready to process video",
            "timestamp": time.time(),
            "timestamp_prefix": None,
            "available_videos": [],
            "available_captions": []
        })

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(
        os.path.join(app.config['OUTPUT_FOLDER'], filename),
        as_attachment=True
    )

@app.route('/static/<path:filename>')
def serve_static(filename):
    print(f"Serving static file: {filename}")
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

# Clear status file on startup
try:
    with open('status.json', 'w') as f:
        json.dump({
            'status': 'Ready to process video', 
            'timestamp': time.time(),
            'timestamp_prefix': None
        }, f)
    print("Status file initialized")
except Exception as e:
    print(f"Error initializing status file: {str(e)}")

if __name__ == '__main__':
    print("Starting Flask app on port 5004...")
    app.run(host='0.0.0.0', port=5004, debug=False)