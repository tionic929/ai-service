from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from rembg import remove
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io

app = Flask(__name__)

# Allow your specific website and local Electron app to call this API
CORS(app, resources={r"/*": {"origins": [
    "https://ncnian-id.svizcarra.online", 
    "http://localhost:5173", # Vite default
    "http://localhost:3000"  # Electron/React default
]}})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "service": "NCNIAN-AI"}), 200

@app.route('/enhance-photo', methods=['POST'])
def enhance():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    try:
        file = request.files['image'].read()
        
        # 1. REMOVE BACKGROUND
        no_bg_bytes = remove(file)
        img = Image.open(io.BytesIO(no_bg_bytes)).convert("RGBA")
        
        # 2. PHOTO RESTORATION (Auto-Leveling & Sharpness)
        # Increase Sharpness
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.8)
        # Increase Contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)
        
        # Save result to memory
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return send_file(img_byte_arr, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/clean-signature', methods=['POST'])
def signature():
    if 'image' not in request.files:
        return jsonify({"error": "No signature uploaded"}), 400
        
    try:
        file = request.files['image'].read()
        nparr = np.frombuffer(file, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # 3. SIGNATURE SCANNING (Binarization)
        # We use adaptive thresholding to handle uneven lighting on paper
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 8)
        
        # Convert to transparent PNG
        success, encoded_img = cv2.imencode('.png', thresh)
        return send_file(io.BytesIO(encoded_img.tobytes()), mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Render uses the PORT environment variable
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)