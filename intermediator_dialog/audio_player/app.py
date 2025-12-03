import os
import io
import re
import subprocess
import tempfile
from flask import Flask, render_template, jsonify, send_from_directory, abort, send_file, request
from mutagen import File as MutagenFile

app = Flask(__name__)

# Configuration - audio files are in output/audio relative to the main project (go up one level from audio_player/)
AUDIO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'audio'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/folders')
def get_folders():
    try:
        folders = []
        for d in os.listdir(AUDIO_ROOT):
            path = os.path.join(AUDIO_ROOT, d)
            if os.path.isdir(path):
                stat = os.stat(path)
                created = getattr(stat, 'st_birthtime', stat.st_ctime)
                folders.append({
                    'name': d,
                    'created': created
                })
        
        # Always sort by date, newest first
        folders.sort(key=lambda x: x['created'], reverse=True)
            
        return jsonify(folders)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def extract_turn_number(filename):
    """Extract turn number from filename like 'speaker_turn3.mp3' -> 3."""
    match = re.search(r'turn(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return float('inf')

def get_sorted_audio_files(folder_path):
    """Get all audio files sorted by turn number (for natural playback order)."""
    audio_files = []
    for f in os.listdir(folder_path):
        if f.lower().endswith(('.mp3', '.wav')):
            filepath = os.path.join(folder_path, f)
            stat = os.stat(filepath)
            created = getattr(stat, 'st_birthtime', stat.st_ctime)
            audio_files.append({
                'name': f,
                'created': created
            })

    # Sort by turn number extracted from filename
    audio_files.sort(key=lambda x: extract_turn_number(x['name']))
    return audio_files

@app.route('/api/files/<folder_id>')
def get_files(folder_id):
    try:
        folder_path = os.path.join(AUDIO_ROOT, folder_id)
        if not os.path.exists(folder_path):
            return jsonify({'error': 'Folder not found'}), 404
        
        files_data = []
        audio_files_list = get_sorted_audio_files(folder_path)

        for file_info in audio_files_list:
            f = file_info['name']
            file_path = os.path.join(folder_path, f)
            duration = 0
            try:
                audio = MutagenFile(file_path)
                if audio is not None:
                    duration = audio.info.length
            except Exception as e:
                print(f"Error reading metadata for {f}: {e}")
            
            files_data.append({
                'name': f,
                'duration': duration,
                'created': file_info['created']
            })
        
        return jsonify(files_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/audio/<folder_id>/<path:filename>')
def serve_audio(folder_id, filename):
    try:
        folder_path = os.path.join(AUDIO_ROOT, folder_id)
        return send_from_directory(folder_path, filename)
    except Exception as e:
        abort(404)

@app.route('/api/convert/<folder_id>')
def convert_folder(folder_id):
    try:
        folder_path = os.path.join(AUDIO_ROOT, folder_id)
        if not os.path.exists(folder_path):
            return jsonify({'error': 'Folder not found'}), 404

        audio_files_list = get_sorted_audio_files(folder_path)
        if not audio_files_list:
             return jsonify({'error': 'No audio files found'}), 404

        # Create a temporary file for the concat list
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as f_list:
            for file_info in audio_files_list:
                filename = file_info['name']
                file_path = os.path.join(folder_path, filename)
                # Escape single quotes for ffmpeg
                safe_path = file_path.replace("'", "'\''")
                f_list.write(f"file '{safe_path}'\n")
            list_path = f_list.name

        # Output file
        output_fd, output_path = tempfile.mkstemp(suffix='.mp3')
        os.close(output_fd)

        # Run ffmpeg
        # ffmpeg -f concat -safe 0 -i list.txt -c copy output.mp3
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', list_path,
            '-c', 'copy',
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)

        # Clean up list file
        os.unlink(list_path)

        # Read the output into memory to serve it
        with open(output_path, 'rb') as f:
            data = io.BytesIO(f.read())
        
        os.unlink(output_path)
        
        return send_file(
            data,
            as_attachment=True,
            download_name=f"{folder_id}_combined.mp3",
            mimetype="audio/mpeg"
        )

    except subprocess.CalledProcessError as e:
        return jsonify({'error': f"FFmpeg failed: {e.stderr.decode()}"}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"Serving audio from: {AUDIO_ROOT}")
    app.run(debug=True, port=5002)
