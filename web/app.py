"""
app.py
------
Flask web application for composer classification.
Provides API endpoints for file upload and prediction.
"""

import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import tempfile

# Add parent directory to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from web.predict import predict_composer, get_available_models

app = Flask(__name__, static_folder=None)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Allowed file extensions
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}
ALLOWED_MIDI_EXTENSIONS = {'mid', 'midi'}

# Create temp directory for uploads
UPLOAD_FOLDER = Path(tempfile.gettempdir()) / "composer_classification_uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)


def allowed_file(filename, extensions):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions


@app.route('/')
def index():
    """Serve the main page."""
    return send_from_directory(REPO_ROOT, 'index.html')


@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files from web folder."""
    return send_from_directory(REPO_ROOT / 'web', path)


@app.route('/api/models', methods=['GET'])
def list_models():
    """
    Get list of available trained models.
    
    Returns:
        JSON array of model information
    """
    try:
        models = get_available_models()
        return jsonify({
            "success": True,
            "models": models
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict composer from uploaded audio/MIDI file.
    
    Expected form data:
        - audio_file: Audio file (optional)
        - midi_file: MIDI file (optional)
        - model_path: Path to model file (required)
    
    Returns:
        JSON with prediction results
    """
    try:
        # Get model path from form
        model_path = request.form.get('model_path')
        if not model_path:
            return jsonify({
                "success": False,
                "error": "No model selected"
            }), 400
        
        # Check if model exists
        if not Path(model_path).exists():
            return jsonify({
                "success": False,
                "error": f"Model not found: {model_path}"
            }), 404
        
        # Get uploaded files
        audio_file = request.files.get('audio_file')
        midi_file = request.files.get('midi_file')
        
        if not audio_file and not midi_file:
            return jsonify({
                "success": False,
                "error": "No file uploaded. Please upload an audio or MIDI file."
            }), 400
        
        # Save uploaded files temporarily
        audio_path = None
        midi_path = None
        
        if audio_file and audio_file.filename:
            if not allowed_file(audio_file.filename, ALLOWED_AUDIO_EXTENSIONS):
                return jsonify({
                    "success": False,
                    "error": f"Invalid audio file type. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
                }), 400
            
            filename = secure_filename(audio_file.filename)
            audio_path = UPLOAD_FOLDER / filename
            audio_file.save(str(audio_path))
        
        if midi_file and midi_file.filename:
            if not allowed_file(midi_file.filename, ALLOWED_MIDI_EXTENSIONS):
                return jsonify({
                    "success": False,
                    "error": f"Invalid MIDI file type. Allowed: {', '.join(ALLOWED_MIDI_EXTENSIONS)}"
                }), 400
            
            filename = secure_filename(midi_file.filename)
            midi_path = UPLOAD_FOLDER / filename
            midi_file.save(str(midi_path))
        
        # Make prediction
        try:
            result = predict_composer(
                audio_path=str(audio_path) if audio_path else None,
                midi_path=str(midi_path) if midi_path else None,
                model_path=model_path
            )
            
            # Clean up uploaded files
            if audio_path and audio_path.exists():
                audio_path.unlink()
            if midi_path and midi_path.exists():
                midi_path.unlink()
            
            return jsonify({
                "success": True,
                "prediction": result
            })
            
        except ValueError as e:
            # Clean up on error
            if audio_path and audio_path.exists():
                audio_path.unlink()
            if midi_path and midi_path.exists():
                midi_path.unlink()
            
            return jsonify({
                "success": False,
                "error": str(e)
            }), 400
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Composer Classification API"
    })


if __name__ == '__main__':
    print("=" * 80)
    print("Composer Classification Web App")
    print("=" * 80)
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Repo root: {REPO_ROOT}")
    print("\nStarting server...")
    print("Open http://localhost:5000 in your browser")
    print("=" * 80)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
