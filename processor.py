from flask import Flask, request, jsonify, render_template_string
import os, io, re
from PIL import Image
import numpy as np
import cv2
import pytesseract

# HuggingFace imports
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

# Point pytesseract to Windows install (update path if needed)
#TESS_PATH = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ------------------ HuggingFace Model Setup ------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_REPO = "umm-maybe/ai-image-detector"

processor = AutoImageProcessor.from_pretrained(MODEL_REPO)
hf_model = AutoModelForImageClassification.from_pretrained(MODEL_REPO).to(device)

# ------------------ Flask Setup ------------------
UPLOAD_DIR = './uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)
app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ID Verifier</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f7fa;
            color: #2c3e50;
            line-height: 1.6;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .container {
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            padding: 40px;
            max-width: 600px;
            width: 90%;
            margin: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #e9ecef;
        }
        
        .header h1 {
            color: #2c3e50;
            font-size: 32px;
            font-weight: 600;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #6c757d;
            font-size: 16px;
            margin-bottom: 0;
        }
        
        .upload-section {
            margin-bottom: 30px;
        }
        
        .file-input-wrapper {
            position: relative;
            display: inline-block;
            width: 100%;
            margin-bottom: 20px;
        }
        
        .file-input {
            width: 100%;
            padding: 15px;
            border: 2px dashed #3498db;
            border-radius: 8px;
            background-color: #f8f9fa;
            color: #495057;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .file-input:hover {
            border-color: #2980b9;
            background-color: #e3f2fd;
        }
        
        .file-input:focus {
            outline: none;
            border-color: #2980b9;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
        }
        
        .upload-btn {
            width: 100%;
            padding: 15px 30px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .upload-btn:hover {
            background-color: #2980b9;
            transform: translateY(-2px);
        }
        
        .upload-btn:active {
            transform: translateY(0);
        }
        
        .upload-btn:disabled {
            background-color: #95a5a6;
            cursor: not-allowed;
            transform: none;
        }
        
        .results {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #e9ecef;
        }
        
        .result-item {
            background-color: #f8f9fa;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #6c757d;
        }
        
        .result-item.valid {
            background-color: #d4edda;
            border-left-color: #28a745;
        }
        
        .result-item.invalid {
            background-color: #f8d7da;
            border-left-color: #dc3545;
        }
        
        .result-label {
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        
        .result-value {
            color: #495057;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            word-break: break-all;
        }
        
        .status-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .status-valid {
            background-color: #28a745;
            color: white;
        }
        
        .status-invalid {
            background-color: #dc3545;
            color: white;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            display: inline-block;
            width: 24px;
            height: 24px;
            border: 3px solid #e9ecef;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error-message {
            background-color: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #f5c6cb;
            margin-top: 15px;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 20px;
                margin: 10px;
            }
            
            .header h1 {
                font-size: 24px;
            }
            
            .upload-btn {
                padding: 12px 20px;
                font-size: 14px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ID Verifier</h1>
            <p>Upload an ID image for preprocessing, OCR analysis, and AI authenticity verification</p>
        </div>
        
        <div class="upload-section">
            <div class="file-input-wrapper">
                <input id="file" class="file-input" type="file" accept="image/*" />
            </div>
            <button class="upload-btn" onclick="upload()">Upload & Verify Document</button>
        </div>
        
        <div id="results" class="results" style="display: none;"></div>
    </div>

    <script>
        async function upload() {
            const fileInput = document.getElementById('file');
            const resultsDiv = document.getElementById('results');
            const uploadBtn = document.querySelector('.upload-btn');
            
            if (!fileInput.files[0]) {
                alert('Please select a file first');
                return;
            }
            
            // Show loading state
            uploadBtn.disabled = true;
            uploadBtn.innerHTML = '<span class="spinner"></span>Processing...';
            resultsDiv.style.display = 'block';
            resultsDiv.innerHTML = '<div class="loading"><span class="spinner"></span>Analyzing document...</div>';
            
            const formData = new FormData();
            formData.append('image', fileInput.files[0]);
            
            try {
                const response = await fetch('/api/verify', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    displayResults(data);
                } else {
                    displayError(data.error || 'An error occurred');
                }
            } catch (error) {
                displayError('Network error: ' + error.message);
            } finally {
                uploadBtn.disabled = false;
                uploadBtn.innerHTML = 'Upload & Verify Document';
            }
        }
        
        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            const validClass = data.valid ? 'valid' : 'invalid';
            const statusClass = data.valid ? 'status-valid' : 'status-invalid';
            const statusText = data.valid ? 'Valid' : 'Invalid';
            
            resultsDiv.innerHTML = `
                <div class="result-item ${validClass}">
                    <div class="result-label">Verification Status</div>
                    <div class="result-value">
                        <span class="status-badge ${statusClass}">${statusText}</span>
                    </div>
                </div>
                
                <div class="result-item">
                    <div class="result-label">Document Type</div>
                    <div class="result-value">${data.document_type || 'Unknown'}</div>
                </div>
                
                <div class="result-item">
                    <div class="result-label">OCR Text (First 500 chars)</div>
                    <div class="result-value">${data.ocr_text_excerpt || 'No text detected'}</div>
                </div>
                
                <div class="result-item">
                    <div class="result-label">Format Check Results</div>
                    <div class="result-value">${data.heuristics_reasons.join(', ') || 'No specific checks performed'}</div>
                </div>
                
                <div class="result-item">
                    <div class="result-label">AI Authenticity Score</div>
                    <div class="result-value">${(data.ai_fake_score * 100).toFixed(2)}% likelihood of being AI-generated</div>
                </div>
                
                <div class="result-item">
                    <div class="result-label">AI Analysis Method</div>
                    <div class="result-value">${data.ai_reasons.join(', ')}</div>
                </div>
            `;
        }
        
        function displayError(message) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `
                <div class="error-message">
                    <strong>Error:</strong> ${message}
                </div>
            `;
        }
        
        // Allow drag and drop
        const fileInput = document.getElementById('file');
        const container = document.querySelector('.container');
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            container.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            container.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            container.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight(e) {
            container.style.backgroundColor = '#e3f2fd';
        }
        
        function unhighlight(e) {
            container.style.backgroundColor = '#ffffff';
        }
        
        container.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length > 0) {
                fileInput.files = files;
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/api/verify', methods=['POST'])
def api_verify():
    file = request.files.get('image')
    if not file:
        return jsonify({'error':'no file uploaded'}), 400

    pil = Image.open(io.BytesIO(file.read())).convert('RGB')

    # Preprocess
    pre = preprocess_image_pil(pil)
    pre.save(os.path.join(UPLOAD_DIR, 'last_preprocessed.jpg'))

    # OCR
    ocr_text = run_ocr(pre)

    # Heuristic doc type check
    doc_type, heuristics_reasons = heuristics_doc_type_and_checks(ocr_text, pre)

    # AI detection with HuggingFace model
    ai_score, ai_reasons = run_hf_fake_detector(pre)

    heur_score = 1.0 if 'failed' not in ' '.join(heuristics_reasons).lower() else 0.0
    valid = (heur_score > 0.7) and (ai_score < 0.6)

    return jsonify({
        'valid': bool(valid),
        'document_type': doc_type,
        'ocr_text_excerpt': ocr_text[:500],
        'heuristics_reasons': heuristics_reasons,
        'ai_fake_score': float(ai_score),
        'ai_reasons': ai_reasons
    })

# ------------------ Preprocessing ------------------

def preprocess_image_pil(pil_img: Image.Image) -> Image.Image:
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img = deskew_image_cv(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10,10,7,21)
    img = unsharp_mask_cv(img)
    h,w = img.shape[:2]
    if max(h,w) > 1600:
        scale = 1600 / max(h,w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def deskew_image_cv(img_cv: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(thresh)
    if coords is None: return img_cv
    rect = cv2.minAreaRect(coords)
    angle = -(90 + rect[-1]) if rect[-1] < -45 else -rect[-1]
    (h,w) = img_cv.shape[:2]
    M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0)
    return cv2.warpAffine(img_cv, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def unsharp_mask_cv(img_cv: np.ndarray, amount=1.0):
    blurred = cv2.GaussianBlur(img_cv, (0,0), sigmaX=3)
    return cv2.addWeighted(img_cv, 1+amount, blurred, -amount, 0)

# ------------------ OCR & Heuristics ------------------

def run_ocr(pil_img: Image.Image) -> str:
    txt = pytesseract.image_to_string(pil_img, config='--oem 3 --psm 6')
    return re.sub(r'\s+', ' ', txt).strip()

def heuristics_doc_type_and_checks(ocr_text: str, pre_image: Image.Image):
    """
    Generic content check: text presence, name, DOB.
    """
    reasons = []
    text = ocr_text.strip()

    if not text:
        reasons.append("no-text-detected")
    else:
        reasons.append("text-detected")

    # Name pattern (capitalized words)
    name_match = re.search(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+){0,2}\b', ocr_text)
    if name_match:
        reasons.append(f"name-found:{name_match.group(0)}")
    else:
        reasons.append("name-not-found")

    # Date pattern (DOB)
    dob_match = re.search(r'\b(0?[1-9]|[12][0-9]|3[01])[-/](0?[1-9]|1[0-2])[-/](\d{2}|\d{4})\b', ocr_text)
    if dob_match:
        reasons.append(f"dob-found:{dob_match.group(0)}")
    else:
        reasons.append("dob-not-found")

    # We return a dummy doc type since frontend expects it
    doc_type = "generic"
    return doc_type, reasons

def detect_mrz(pil_img: Image.Image) -> bool:
    txt = pytesseract.image_to_string(
        pil_img.crop((0, pil_img.height - int(pil_img.height*0.28), pil_img.width, pil_img.height)),
        config='--oem 3 --psm 6'
    )
    return bool(re.search(r'[A-Z0-9<]{10,}\n[A-Z0-9<]{10,}', txt.replace(' ', '')))

# ------------------ HuggingFace Fake Detector ------------------

def run_hf_fake_detector(pil_img: Image.Image):
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = hf_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    # Assumes label 0 = real, 1 = fake (check model.config.id2label to confirm)
    fake_score = float(probs[1])
    return fake_score, [f'huggingface-model:{MODEL_REPO}']

if __name__ == '__main__':
    app.run(debug=True)
