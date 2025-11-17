from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import numpy as np
import cv2 as cv
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, Xception, ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from skimage import feature, filters

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class AdvancedDeepfakeDetector:
    """
    Advanced multi-modal deepfake detector with:
    - 3-model ensemble (EfficientNet, Xception, ResNet50V2)
    - Facial structure analysis
    - Texture artifact detection
    - Frequency domain analysis
    - Color consistency checks
    """
    
    def __init__(self):
        print("="*80)
        print("ADVANCED DEEPFAKE DETECTOR v2.0")
        print("Multi-Modal Analysis | Enhanced False Negative Detection")
        print("="*80)
        
        self.ensemble_models = []
        self._build_enhanced_ensemble()
        
        # Detection thresholds (lower = more sensitive to fakes)
        self.FAKE_THRESHOLD = 0.45
        self.HIGH_CONFIDENCE_THRESHOLD = 0.65
    
    def _build_enhanced_ensemble(self):
        print("\n🔧 Building Enhanced Detection Ensemble...")
        
        try:
            # MODEL 1: EfficientNetB0 (Best for texture)
            print("  [1/3] Building EfficientNetB0...")
            base1 = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            inputs1 = Input(shape=(224, 224, 3))
            x1 = base1(inputs1, training=False)
            x1 = GlobalAveragePooling2D()(x1)
            x1 = Dense(256, activation='relu')(x1)
            x1 = BatchNormalization()(x1)
            x1 = Dropout(0.5)(x1)
            outputs1 = Dense(1, activation='sigmoid')(x1)
            model1 = Model(inputs=inputs1, outputs=outputs1, name='EfficientNet')
            self.ensemble_models.append(model1)
            print("    ✓ EfficientNet ready")
            
            # MODEL 2: Xception (Best for artifacts)
            print("  [2/3] Building Xception...")
            base2 = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            inputs2 = Input(shape=(224, 224, 3))
            x2 = base2(inputs2, training=False)
            x2 = GlobalAveragePooling2D()(x2)
            x2 = Dense(256, activation='relu')(x2)
            x2 = BatchNormalization()(x2)
            x2 = Dropout(0.5)(x2)
            outputs2 = Dense(1, activation='sigmoid')(x2)
            model2 = Model(inputs=inputs2, outputs=outputs2, name='Xception')
            self.ensemble_models.append(model2)
            print("    ✓ Xception ready")
            
            # MODEL 3: ResNet50V2 (Best for facial features)
            print("  [3/3] Building ResNet50V2...")
            base3 = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            inputs3 = Input(shape=(224, 224, 3))
            x3 = base3(inputs3, training=False)
            x3 = GlobalAveragePooling2D()(x3)
            x3 = Dense(256, activation='relu')(x3)
            x3 = BatchNormalization()(x3)
            x3 = Dropout(0.5)(x3)
            outputs3 = Dense(1, activation='sigmoid')(x3)
            model3 = Model(inputs=inputs3, outputs=outputs3, name='ResNet50V2')
            self.ensemble_models.append(model3)
            print("    ✓ ResNet50V2 ready")
            
            print(f"\n✓ Ensemble of {len(self.ensemble_models)} models ready!")
            
        except Exception as e:
            print(f" Error building models: {e}")
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Preprocess image for model input"""
        if isinstance(image_path, str):
            img = cv.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load: {image_path}")
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        else:
            img = image_path
        
        img_resized = cv.resize(img, target_size)
        img_array = img_to_array(img_resized) / 127.5 - 1.0
        return np.expand_dims(img_array, axis=0)
    
    def analyze_image(self, image_path):
        """
        Complete multi-modal analysis pipeline
        Returns detailed JSON response for API
        """
        try:
            # Load image
            img = cv.imread(image_path)
            if img is None:
                return {'error': 'Could not load image'}
            
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            # STEP 1: ENSEMBLE MODEL PREDICTIONS 
            img_processed = self.preprocess_image(image_path)
            model_predictions = []
            
            for model in self.ensemble_models:
                pred = float(model.predict(img_processed, verbose=0)[0][0])
                model_predictions.append({
                    'name': model.name,
                    'score': pred,
                    'label': 'FAKE' if pred > 0.5 else 'REAL'
                })
            
            ensemble_score = np.mean([p['score'] for p in model_predictions])
            ensemble_std = np.std([p['score'] for p in model_predictions])
            
            # STEP 2: FORENSIC ANALYSIS 
            facial_score = self._analyze_facial_features(img_rgb)
            texture_score = self._analyze_texture_artifacts(img_rgb)
            freq_score = self._analyze_frequency_domain(img_rgb)
            color_score = self._analyze_color_consistency(img_rgb)
            
            #  STEP 3: WEIGHTED COMBINATION 
            weights = {
                'ensemble': 0.35,
                'facial': 0.25,
                'texture': 0.20,
                'frequency': 0.15,
                'color': 0.05
            }
            
            final_score = (
                ensemble_score * weights['ensemble'] +
                facial_score * weights['facial'] +
                texture_score * weights['texture'] +
                freq_score * weights['frequency'] +
                color_score * weights['color']
            )
            
            # ===== STEP 4: VERDICT & CONFIDENCE =====
            is_fake = final_score > self.FAKE_THRESHOLD
            
            if final_score > self.HIGH_CONFIDENCE_THRESHOLD:
                confidence_level = "VERY HIGH"
                confidence = (final_score - self.FAKE_THRESHOLD) / (1 - self.FAKE_THRESHOLD) * 100
            elif final_score > self.FAKE_THRESHOLD:
                confidence_level = "MODERATE"
                confidence = (final_score - self.FAKE_THRESHOLD) / (self.HIGH_CONFIDENCE_THRESHOLD - self.FAKE_THRESHOLD) * 50 + 50
            elif final_score > 0.3:
                confidence_level = "LOW"
                confidence = (self.FAKE_THRESHOLD - final_score) / (self.FAKE_THRESHOLD - 0.3) * 50 + 50
            else:
                confidence_level = "HIGH"
                confidence = (self.FAKE_THRESHOLD - final_score) / self.FAKE_THRESHOLD * 100
            
            # ===== STEP 5: DETERMINE MANIPULATION TYPE =====
            manipulation_type = None
            key_indicators = []
            
            if is_fake:
                if ensemble_score > 0.5:
                    key_indicators.append("AI models detected synthetic patterns")
                if facial_score > 0.5:
                    key_indicators.append("Facial structure shows anomalies")
                if texture_score > 0.6:
                    key_indicators.append("Unnatural texture smoothness detected")
                if freq_score > 0.5:
                    key_indicators.append("Suspicious frequency patterns found")
                if color_score > 0.5:
                    key_indicators.append("Color/lighting inconsistencies detected")
                
                # Classify manipulation type
                if texture_score > 0.7 and facial_score > 0.6:
                    manipulation_type = "AI-Generated / GAN-Created Face"
                elif facial_score > 0.7:
                    manipulation_type = "Face Morphing / Blended Faces"
                elif ensemble_score > 0.7:
                    manipulation_type = "Deepfake / Face Swap"
                else:
                    manipulation_type = "Manipulated / Synthetic Content"
            
            # ===== BUILD RESPONSE =====
            verdict = 'FAKE' if is_fake else 'REAL'
            explanation = (
                "⚠️ WARNING: This image is likely MANIPULATED, AI-GENERATED, or a DEEPFAKE!"
                if is_fake else
                "✓ VERIFIED: This image appears to be AUTHENTIC and REAL"
            )
            
            return {
                'classification': verdict,
                'confidence': round(confidence, 2),
                'confidence_level': confidence_level,
                'explanation': explanation,
                'ensemble_score': round(final_score, 4),
                'threshold': self.FAKE_THRESHOLD,
                'authenticity_score': round((1 - final_score) * 100, 1),
                'manipulation_type': manipulation_type,
                'key_indicators': key_indicators,
                'model_predictions': model_predictions,
                'forensic_analysis': {
                    'ensemble': round(ensemble_score, 4),
                    'facial': round(facial_score, 4),
                    'texture': round(texture_score, 4),
                    'frequency': round(freq_score, 4),
                    'color': round(color_score, 4),
                    'ensemble_std': round(ensemble_std, 4),
                    'agreement': 'High' if ensemble_std < 0.1 else 'Medium' if ensemble_std < 0.2 else 'Low'
                },
                'weights': weights,
                'quality_metrics': self._get_basic_quality_metrics(img),
                'facial_analysis': self._get_face_detection_info(img),
                'image_properties': {
                    'resolution': f"{w}x{h}",
                    'aspect_ratio': round(w/h, 2)
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            import traceback
            return {'error': str(e), 'traceback': traceback.format_exc()}
    
    def _analyze_facial_features(self, img):
        """Analyze facial structure for deepfake indicators"""
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return 0.3  # Neutral score
        
        anomaly_score = 0.0
        
        for (x, y, w, h) in faces:
            face_region = gray[y:y+h, x:x+w]
            face_color = img[y:y+h, x:x+w]
            
            # Sharpness check (deepfakes often overly smooth)
            sharpness = cv.Laplacian(face_region, cv.CV_64F).var()
            if sharpness < 80:
                anomaly_score += 0.3
            elif sharpness > 300:
                anomaly_score += 0.2
            
            # Eye detection
            eyes = eye_cascade.detectMultiScale(face_region, 1.1, 5)
            if len(eyes) < 2:
                anomaly_score += 0.2
            
            # Skin texture uniformity
            skin_variance = np.var(face_color)
            if skin_variance < 100:
                anomaly_score += 0.3
            
            # Edge consistency
            edges = cv.Canny(face_region, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size
            if edge_density < 0.03:
                anomaly_score += 0.2
        
        return min(anomaly_score, 1.0)
    
    def _analyze_texture_artifacts(self, img):
        """Detect GAN/deepfake texture artifacts"""
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        artifact_score = 0.0
        
        #  Local Binary Patterns
        radius = 3
        n_points = 8 * radius
        lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
        lbp_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))
        
        if lbp_entropy < 3.5:
            artifact_score += 0.3
        
        # Gradient magnitude variance
        sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
        sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_var = np.var(gradient_magnitude)
        
        if gradient_var < 500:
            artifact_score += 0.25
        
        # Noise consistency
        noise_estimate = filters.gaussian(gray, sigma=1) - gray
        noise_std = np.std(noise_estimate)
        
        if noise_std < 2.0:
            artifact_score += 0.25
        
        #  Checkerboard artifact detection
        h, w = gray.shape
        patch_size = 16
        checkerboard_score = 0
        
        for y in range(0, h - patch_size, patch_size):
            for x in range(0, w - patch_size, patch_size):
                patch = gray[y:y+patch_size, x:x+patch_size]
                if patch.shape == (patch_size, patch_size):
                    if np.std(patch) < 5:
                        checkerboard_score += 1
        
        total_patches = ((h // patch_size) * (w // patch_size))
        uniform_ratio = checkerboard_score / total_patches if total_patches > 0 else 0
        
        if uniform_ratio > 0.3:
            artifact_score += 0.2
        
        return min(artifact_score, 1.0)
    
    def _analyze_frequency_domain(self, img):
        """Analyze frequency domain for manipulation signatures"""
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        
        # FFT analysis
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Low frequency energy
        low_freq = magnitude[center_h-20:center_h+20, center_w-20:center_w+20]
        low_freq_energy = np.mean(low_freq)
        
        # High frequency energy
        high_freq_mask = np.ones_like(magnitude)
        high_freq_mask[center_h-50:center_h+50, center_w-50:center_w+50] = 0
        high_freq_energy = np.mean(magnitude * high_freq_mask)
        
        freq_ratio = high_freq_energy / (low_freq_energy + 1e-10)
        
        if freq_ratio < 0.01:
            return 0.7
        elif freq_ratio > 0.1:
            return 0.5
        return 0.2
    
    def _analyze_color_consistency(self, img):
        """Analyze color distribution and lighting consistency"""
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        r_std, g_std, b_std = np.std(r), np.std(g), np.std(b)
        channel_similarity = np.std([r_std, g_std, b_std])
        
        if channel_similarity < 3:
            return 0.6
        return 0.2
    
    def _get_basic_quality_metrics(self, img):
        """Get basic image quality metrics"""
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 100, 200)
        
        return {
            'sharpness': float(cv.Laplacian(gray, cv.CV_64F).var()),
            'noise': float(np.std(gray)),
            'edge_density': float(np.sum(edges > 0) / edges.size)
        }
    
    def _get_face_detection_info(self, img):
        """Get face detection information"""
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        face_details = []
        for (x, y, w, h) in faces:
            face_region = gray[y:y+h, x:x+w]
            face_details.append({
                'size': f"{w}x{h}",
                'sharpness': float(cv.Laplacian(face_region, cv.CV_64F).var()),
                'position': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
            })
        
        return {
            'faces_detected': len(faces),
            'face_details': face_details
        }


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



print("\n Initializing Advanced Deepfake Detector...")
detector = AdvancedDeepfakeDetector()


@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint for image analysis"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP, WEBP'}), 400
    
    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Analyze image
        result = detector.analyze_image(filepath)
        
        # file info
        result['filename'] = filename
        result['file_size'] = os.path.getsize(filepath)
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(detector.ensemble_models),
        'model_names': [m.name for m in detector.ensemble_models],
        'threshold': detector.FAKE_THRESHOLD,
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    print("\n" + "="*80)
    print("DEEPFAKE DETECTOR WEB APP v2.0")
    print("Advanced Multi-Modal Analysis System")
    print("="*80)
    print(f"\nAccess at: http://localhost:5000")
    print(f"Models loaded: {len(detector.ensemble_models)}")
    print(f" Detection threshold: {detector.FAKE_THRESHOLD}")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
