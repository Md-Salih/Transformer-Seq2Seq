"""
Flask Web Application for Text Summarization

Modern, production-grade UI with:
- Token-by-token animated generation
- Real-time progress indicators
- Smooth transitions and loading states
- Responsive design
- Professional UX

This demonstrates the autoregressive generation visually!
"""

from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from inference import PretrainedInference
import torch
import json
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for React Native app

# Initialize model (loaded once at startup)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Initializing model on {device}...")

try:
    # Use sshleifer/distilbart-cnn-12-6 for fast, high-quality summarization
    print("Loading DistilBART model (optimized for speed and quality)...")
    model = PretrainedInference("sshleifer/distilbart-cnn-12-6", device=device)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load DistilBART model: {e}")
    print("Trying T5-small as fallback...")
    try:
        model = PretrainedInference("t5-small", device=device)
        print("✓ T5-Small model loaded successfully")
    except Exception as e2:
        print(f"Error: Could not load any model: {e2}")
        print("Install transformers: pip install transformers")
        model = None


@app.route('/')
def index():
    """Serve the main UI."""
    return render_template('index.html')


@app.route('/api/summarize', methods=['POST'])
def summarize():
    """
    Generate summary (standard non-streaming endpoint) with optimized parameters.
    """
    try:
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if len(text) < 10:
            return jsonify({'error': 'Text too short (minimum 10 characters)'}), 400
        
        if len(text) > 5000:
            return jsonify({'error': 'Text too long (maximum 5000 characters)'}), 400
        
        # Optimized generation parameters for accuracy
        max_length = data.get('max_length', 150)
        min_length = data.get('min_length', 40)
        num_beams = data.get('num_beams', 8)
        
        if model is None:
            return jsonify({'error': 'Model not loaded. Please install: pip install transformers torch'}), 500
        
        # Generate summary with optimized settings
        start_time = time.time()
        summary = model.summarize(
            text,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=2.0,
            no_repeat_ngram_size=3
        )
        elapsed = time.time() - start_time
        
        return jsonify({
            'summary': summary,
            'input_length': len(text),
            'summary_length': len(summary),
            'compression_ratio': round((1 - len(summary) / len(text)) * 100, 1),
            'generation_time': round(elapsed, 2),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in summarize: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/summarize/stream', methods=['POST'])
def summarize_stream():
    """
    Generate summary with token-by-token streaming.
    This powers the animated UI with optimized parameters!
    """
    try:
        data = request.json
        text = data.get('text', '').strip()
        max_length = data.get('max_length', 150)  # Increased for better quality
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        def generate():
            """Generator function for Server-Sent Events."""
            try:
                start_time = time.time()
                token_count = 0
                
                # Stream tokens in real-time as they are generated
                for token_info in model.summarize_streaming(text, max_length=max_length):  # type: ignore
                    token_count += 1
                    total = token_info.get('total', max_length)
                    progress = min(100, int((token_count / total) * 100))
                    
                    event_data = {
                        'token': token_info.get('token', ''),
                        'step': token_count,
                        'progress': progress,
                        'done': False
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"
                    time.sleep(0.03)  # Smooth animation delay
                
                # Send completion
                elapsed = time.time() - start_time
                completion_data = {
                    'done': True,
                    'progress': 100,
                    'time': round(elapsed, 2)
                }
                yield f"data: {json.dumps(completion_data)}\n\n"
                
            except Exception as e:
                print(f"Streaming error: {e}")
                error_data = {'error': str(e)}
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return Response(generate(), mimetype='text/event-stream',
                       headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': device,
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("TRANSFORMER SUMMARIZATION - WEB APPLICATION")
    print("=" * 70)
    print(f"\nDevice: {device}")
    print(f"Model loaded: {model is not None}")
    print("\nStarting server...")
    print("Open http://localhost:5000 in your browser")
    print("=" * 70 + "\n")
    
    # Disable auto-reloader to prevent threading issues
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
