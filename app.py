import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt
from PIL import Image
import io
import os

# Define your custom metrics
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision_metric = tf.keras.metrics.Precision()
        self.recall_metric = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision_metric.update_state(y_true, y_pred, sample_weight)
        self.recall_metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision_metric.result()
        r = self.recall_metric.result()
        if p + r == 0:
            return 0.0
        return 2 * ((p * r) / (p + r))

    def reset_state(self):
        self.precision_metric.reset_state()
        self.recall_metric.reset_state()

# Patching InputLayer for compatibility
from tensorflow.keras.layers import InputLayer as OriginalKerasInputLayer

def patch_input_layer():
    class PatchedInputLayer(OriginalKerasInputLayer):
        @classmethod
        def from_config(cls, config):
            if 'batch_shape' in config and 'input_shape' not in config:
                config['input_shape'] = config['batch_shape'][1:]
                del config['batch_shape']
            return super().from_config(config)
            
    tf.keras.utils.get_custom_objects()['InputLayer'] = PatchedInputLayer

class DummyDTypePolicy:
    def __init__(self, name=None, **kwargs):
        self.name = name or 'float32'
        self._compute_dtype = tf.float32
        self._variable_dtype = tf.float32

    @property
    def compute_dtype(self):
        return self._compute_dtype

    @property
    def variable_dtype(self):
        return self._variable_dtype

    def get_config(self):
        return {'name': self.name}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

patch_input_layer()

# Constants
IMG_SIZE = 224

# Page configuration
st.set_page_config(
    page_title="AI Retina Scanner - Deteksi Retinopati Diabetik",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Modern Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
    }
    
    .header-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="0.5"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
        animation: float 20s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(1deg); }
    }
    
    .header-content {
        position: relative;
        z-index: 2;
        text-align: center;
    }
    
    .header-title {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 0 4px 8px rgba(0,0,0,0.2);
        letter-spacing: -0.02em;
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 2rem;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
    }
    
    .header-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,0.2));
    }
    
    /* Modern Cards */
    .modern-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }
    
    .modern-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    /* Upload Area */
    .upload-area {
        border: 2px dashed #e0e0e0;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        background: linear-gradient(45deg, #f8f9ff 0%, #f0f8ff 100%);
        transition: all 0.3s ease;
        margin: 2rem 0;
    }
    
    .upload-area:hover {
        border-color: #667eea;
        background: linear-gradient(45deg, #f0f8ff 0%, #e8f4ff 100%);
    }
    
    /* Result Cards */
    .result-card {
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
        color: white;
        text-align: center;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
    }
    
    .result-content {
        position: relative;
        z-index: 2;
    }
    
    .result-title {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .result-description {
        font-size: 1rem;
        opacity: 0.9;
        margin-bottom: 1rem;
        line-height: 1.5;
    }
    
    .confidence-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        backdrop-filter: blur(10px);
    }
    
    /* Recommendation Box */
    .recommendation-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 5px solid #667eea;
        border-radius: 0 12px 12px 0;
        padding: 1.5rem;
        margin: 1rem 0;
        position: relative;
    }
    
    .recommendation-box::before {
        content: '💡';
        position: absolute;
        top: -10px;
        left: -15px;
        background: #667eea;
        color: white;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
    }
    
    .recommendation-title {
        color: #495057;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        margin-left: 1rem;
    }
    
    .recommendation-text {
        color: #6c757d;
        line-height: 1.6;
        margin-left: 1rem;
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9ff 0%, #ffffff 100%);
    }
    
    /* Modern Button */
    .analyze-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
        margin: 2rem 0;
    }
    
    .analyze-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6c757d;
        padding: 2rem;
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f8ff 100%);
        border-radius: 16px;
        margin-top: 3rem;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    /* Animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit branding */
    .stDeployButton {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Preprocessing functions (unchanged)
def crop_all_sides(img, tol=7):
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    mask = gray > tol

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    if len(rows) > 0 and len(cols) > 0:
        top, bottom = rows[0], rows[-1]
        left, right = cols[0], cols[-1]
        img = img[top:bottom+1, left:right+1]

    return img

def pad_to_square(img, pad=25, pad_color=(0, 0, 0)):
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=pad_color)
    return padded

def resize_img(img, size=224):
    return cv2.resize(img, (size, size))

def create_retina_mask(img, threshold=15):
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def apply_black_background(img, mask):
    if mask.ndim == 2 and img.ndim == 3:
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    else:
        mask_3ch = mask
    black_bg = np.zeros_like(img)
    result = np.where(mask_3ch == 255, img, black_bg)
    return result

def preprocess_image_for_prediction(img_array, sigmaX=10):
    img = img_array.copy()
    img = resize_img(img, size=IMG_SIZE)
    img = crop_all_sides(img)
    retina_mask = create_retina_mask(img)
    img = apply_black_background(img, retina_mask)
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX)
    img = cv2.addWeighted(img, 4.0, blurred, -4.0, 128)
    img = resize_img(img, size=IMG_SIZE)
    img = pad_to_square(img)
    img = resize_img(img, size=IMG_SIZE)
    return img

@st.cache_resource
def load_trained_model():
    model_path = 'BestModel.h5'
    
    if not os.path.exists(model_path):
        st.error(f"❌ File model '{model_path}' tidak ditemukan!")
        st.info("📝 Pastikan file 'BestModel.h5' ada di direktori yang sama dengan aplikasi ini.")
        return None
    
    try:
        custom_objects = {
            'accuracy': tf.keras.metrics.Accuracy(),
            'auc_1': tf.keras.metrics.AUC(name='auc_1'), 
            'precision_2': tf.keras.metrics.Precision(name='precision_2'),
            'recall_2': tf.keras.metrics.Recall(name='recall_2'), 
            'F1Score': F1Score(), 
            'DTypePolicy': DummyDTypePolicy 
        }

        with st.spinner("🔄 Memuat model AI..."):
            model = load_model(model_path, custom_objects=custom_objects, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def predict_retinopathy(model, processed_img):
    img_normalized = processed_img.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    prediction = model.predict(img_batch, verbose=0)
    return prediction[0]

def get_severity_info(class_idx):
    severity_info = {
        0: {
            "name": "Normal - Tidak Ada Retinopati",
            "short_name": "Normal",
            "description": "Mata dalam kondisi sehat, tidak terdeteksi tanda-tanda retinopati diabetik.",
            "color": "linear-gradient(135deg, #00c851 0%, #007e33 100%)",
            "icon": "✅",
            "recommendation": "Pertahankan gaya hidup sehat dan lakukan pemeriksaan mata rutin setiap tahun."
        },
        1: {
            "name": "Retinopati Diabetik Ringan",
            "short_name": "Ringan",
            "description": "Terdeteksi mikroaneurisma ringan pada pembuluh darah retina.",
            "color": "linear-gradient(135deg, #ffbb33 0%, #ff8800 100%)",
            "icon": "⚠️",
            "recommendation": "Kontrol gula darah lebih ketat dan konsultasi rutin dengan dokter mata setiap 6-12 bulan."
        },
        2: {
            "name": "Retinopati Diabetik Sedang",
            "short_name": "Sedang",
            "description": "Terdapat perdarahan ringan, eksudasi, dan mikroaneurisma yang lebih banyak.",
            "color": "linear-gradient(135deg, #ff6348 0%, #e55039 100%)",
            "icon": "🟡",
            "recommendation": "Segera konsultasi dengan dokter mata spesialis retina untuk evaluasi lebih lanjut."
        },
        3: {
            "name": "Retinopati Diabetik Berat",
            "short_name": "Berat",
            "description": "Perdarahan retina ekstensif, cotton wool spots, dan gangguan vaskular signifikan.",
            "color": "linear-gradient(135deg, #e74c3c 0%, #c0392b 100%)",
            "icon": "🔴",
            "recommendation": "SEGERA konsultasi dengan dokter mata spesialis retina untuk penanganan intensif dalam 2-4 minggu."
        },
        4: {
            "name": "Retinopati Diabetik Proliferatif",
            "short_name": "Proliferatif",
            "description": "Neovaskularisasi aktif dengan risiko tinggi komplikasi serius dan kehilangan penglihatan.",
            "color": "linear-gradient(135deg, #8e44ad 0%, #663399 100%)",
            "icon": "🚨",
            "recommendation": "DARURAT MATA - Segera konsultasi dengan dokter mata spesialis retina dalam 1-2 minggu. Mungkin diperlukan terapi laser atau pembedahan."
        }
    }
    return severity_info.get(class_idx, severity_info[0])

def main():
    # Modern Header Banner
    st.markdown("""
    <div class="header-banner">
        <div class="header-content">
            <span class="header-icon">👁️🔬</span>
            <h1 class="header-title">AI Retina Scanner</h1>
            <p class="header-subtitle">
                Sistem Deteksi Cerdas Retinopati Diabetik menggunakan Deep Learning
                <br>Akurasi Tinggi • Hasil Cepat • Teknologi Terdepan
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with modern styling
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; color: white; text-align: center;">
            <h2 style="margin: 0; color: white;">🤖 Model AI Info</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **🏗️ Arsitektur Model:**
        - **Base Model:** EfficientNetB0
        - **Global Average Pooling**
        - **Dropout Layer** (Regularisasi)
        - **Dense Layer** (128 units)
        - **Batch Normalization**
        - **Output Layer** (5 kelas)
        
        **🔬 Pipeline Preprocessing:**
        - **Image Cropping** - Hapus border hitam
        - **Retina Masking** - Isolasi area retina
        - **Background Removal** - Latar belakang hitam
        - **Ben Graham Enhancement** - Peningkatan kontras
        - **Resizing** - Normalisasi ke 224×224 px
        - **Padding** - Mempertahankan aspek rasio
        
        **📊 Performa Model:**
        - **Akurasi:** >92%
        - **Presisi:** >90%
        - **Recall:** >88%
        - **F1-Score:** >89%
        """)
        
        st.markdown("""
        <div style="background: #f0f8ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #667eea; margin-top: 2rem;">
            <small><strong>💡 Tips:</strong><br>
            Gunakan gambar fundus retina berkualitas tinggi untuk hasil optimal.</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Load model
    model = load_trained_model()
    
    if model is None:
        st.markdown("""
        <div class="modern-card">
            <h3 style="color: #e74c3c; text-align: center;">⚠️ Model Tidak Ditemukan</h3>
            <p style="text-align: center;">Silakan upload file model atau periksa keberadaan file BestModel.h5</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_model = st.file_uploader(
            "📤 Upload Model File (BestModel.h5)", 
            type=['h5'], 
            help="Upload file model yang sudah dilatih"
        )
        
        if uploaded_model is not None:
            try:
                temp_model_path = "BestModel.h5"
                with open(temp_model_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
                
                model = load_trained_model()
                if model is not None:
                    if os.path.exists(temp_model_path):
                        os.remove(temp_model_path)
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
        return
    
    # Success message for loaded model
    st.success("✅ Model AI berhasil dimuat dan siap untuk analisis!")
    
    # File uploader with modern styling
    st.markdown("""
    <div class="modern-card">
        <h3 style="text-align: center; color: #495057; margin-bottom: 1rem;">
            📁 Upload Gambar Fundus Retina
        </h3>
        <p style="text-align: center; color: #6c757d;">
            Pilih gambar fundus retina (PNG, JPG, JPEG) untuk dianalisis oleh AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Pilih gambar...",
        type=['png', 'jpg', 'jpeg'],
        help="Format yang didukung: PNG, JPG, JPEG (Maks. 200MB)"
    )
    
    if uploaded_file is not None:
        try:
            # Load and convert image
            pil_image = Image.open(uploaded_file)
            img_array = np.array(pil_image)
            
            # Convert to RGB if needed
            if len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 3:  # Already RGB
                pass
            else:  # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            
            # Display original image in modern card
            st.markdown("""
            <div class="modern-card">
                <h3 style="text-align: center; color: #495057; margin-bottom: 1rem;">
                    🖼️ Gambar Fundus Retina
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(img_array, caption="Gambar yang diupload", use_column_width=True)
            
            # Modern analyze button
            analyze_clicked = st.button(
                "🔍 Mulai Analisis AI", 
                type="primary",
                help="Klik untuk memulai proses analisis retinopati diabetik"
            )
            
            if analyze_clicked:
                # Progress bar
                progress_bar = st.progress(0)
                
                with st.spinner("🧠 AI sedang menganalisis gambar retina..."):
                    try:
                        # Preprocessing
                        progress_bar.progress(25)
                        processed_img = preprocess_image_for_prediction(img_array, sigmaX=10)
                        
                        # Prediction
                        progress_bar.progress(75)
                        predictions = predict_retinopathy(model, processed_img)
                        predicted_class = np.argmax(predictions)
                        confidence = predictions[predicted_class]
                        
                        progress_bar.progress(100)
                        
                        # Get severity info
                        severity_info = get_severity_info(predicted_class)
                        
                        # Results section
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("---")
                        
                        # Main result card
                        st.markdown(f"""
                        <div class="result-card" style="background: {severity_info['color']};">
                            <div class="result-content">
                                <div style="font-size: 3rem; margin-bottom: 1rem;">{severity_info['icon']}</div>
                                <h2 class="result-title">{severity_info['name']}</h2>
                                <p class="result-description">{severity_info['description']}</p>
                                <div class="confidence-badge">
                                    Confidence: {confidence:.1%}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Recommendation
                        st.markdown(f"""
                        <div class="recommendation-box">
                            <h4 class="recommendation-title">Rekomendasi Medis</h4>
                            <p class="recommendation-text">{severity_info['recommendation']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Probability distribution
                        st.markdown("""
                        <div class="modern-card">
                            <h3 style="text-align: center; color: #495057; margin-bottom: 2rem;">
                                📊 Distribusi Probabilitas Diagnosis
                            </h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        class_names = [
                            "Normal",
                            "Ringan", 
                            "Sedang",
                            "Berat",
                            "Proliferatif"
                        ]
                        
                        # Enhanced bar chart
                        fig, ax = plt.subplots(figsize=(12, 7))
                        
                        # Create gradient colors
                        colors = ['#00c851', '#ffbb33', '#ff6348', '#e74c3c', '#8e44ad']
                        bars = ax.bar(class_names, predictions, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
                        
                        # Highlight predicted class
                        bars[predicted_class].set_alpha(1.0)
                        bars[predicted_class].set_edgecolor('#333')
                        bars[predicted_class].set_linewidth(3)
                        
                        # Styling
                        ax.set_ylabel('Probabilitas (%)', fontsize=12, fontweight='bold')
                        ax.set_title('Distribusi Probabilitas Tingkat Keparahan Retinopati Diabetik', 
                                   fontsize=14, fontweight='bold', pad=20)
                        ax.set_ylim(0, 1)
                        ax.grid(True, alpha=0.3, axis='y')
                        ax.set_facecolor('#f8f9fa')
                        
                        # Add percentage labels on bars
                        for i, (bar, prob) in enumerate(zip(bars, predictions)):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                    f'{prob:.1%}', ha='center', va='bottom',
                                    fontweight='bold', fontsize=11)
                        
                        plt.xticks(fontsize=11, fontweight='500')
                        plt.yticks(fontsize=10)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Detailed probability table
                        st.markdown("""
                        <div class="modern-card">
                            <h3 style="text-align: center; color: #495057; margin-bottom: 1rem;">
                                📈 Detail Analisis Probabilitas
                            </h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Create detailed dataframe
                        prob_data = []
                        for i, (name, prob) in enumerate(zip(class_names, predictions)):
                            severity_detail = get_severity_info(i)
                            prob_data.append({
                                "🏷️ Diagnosis": severity_detail['name'],
                                "📊 Probabilitas": f"{prob:.4f}",
                                "📈 Persentase": f"{prob:.1%}",
                                "🎯 Status": "✅ TERDETEKSI" if i == predicted_class else "❌"
                            })
                        
                        import pandas as pd
                        df = pd.DataFrame(prob_data)
                        
                        # Style the dataframe
                        def highlight_prediction(row):
                            if "✅ TERDETEKSI" in row['🎯 Status']:
                                return ['background-color: #d4edda; color: #155724; font-weight: bold'] * len(row)
                            return [''] * len(row)
                        
                        styled_df = df.style.apply(highlight_prediction, axis=1)
                        st.dataframe(styled_df, use_container_width=True, hide_index=True)
                        
                        # Additional insights
                        st.markdown("""
                        <div class="modern-card">
                            <h3 style="color: #495057; margin-bottom: 1rem;">🔍 Insight Analisis</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                label="🎯 Confidence Score", 
                                value=f"{confidence:.1%}",
                                delta=f"{'Tinggi' if confidence > 0.8 else 'Sedang' if confidence > 0.6 else 'Rendah'}"
                            )
                        
                        with col2:
                            second_highest = np.partition(predictions, -2)[-2]
                            certainty = confidence - second_highest
                            st.metric(
                                label="🔒 Certainty Gap", 
                                value=f"{certainty:.1%}",
                                delta=f"{'Yakin' if certainty > 0.3 else 'Cukup Yakin' if certainty > 0.1 else 'Perlu Konfirmasi'}"
                            )
                        
                        with col3:
                            risk_level = "Tinggi" if predicted_class >= 3 else "Sedang" if predicted_class >= 1 else "Rendah"
                            st.metric(
                                label="⚡ Risk Level", 
                                value=risk_level,
                                delta=severity_info['short_name']
                            )
                        
                    except Exception as e:
                        st.error(f"❌ Error saat melakukan prediksi: {str(e)}")
                        st.exception(e)
            
        except Exception as e:
            st.error(f"❌ Error saat memproses gambar: {str(e)}")
    
    else:
        # Landing information when no file uploaded
        st.markdown("""
        <div class="modern-card" style="text-align: center; padding: 3rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🏥</div>
            <h3 style="color: #495057; margin-bottom: 1rem;">Selamat Datang di AI Retina Scanner</h3>
            <p style="color: #6c757d; font-size: 1.1rem; line-height: 1.6; max-width: 600px; margin: 0 auto;">
                Teknologi AI terdepan untuk deteksi dini retinopati diabetik. 
                Upload gambar fundus retina untuk mendapatkan analisis komprehensif dalam hitungan detik.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="modern-card" style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🎯</div>
                <h4 style="color: #495057;">Akurasi Tinggi</h4>
                <p style="color: #6c757d;">Akurasi >92% dengan teknologi EfficientNetB0</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="modern-card" style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">⚡</div>
                <h4 style="color: #495057;">Hasil Cepat</h4>
                <p style="color: #6c757d;">Analisis selesai dalam hitungan detik</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="modern-card" style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🔬</div>
                <h4 style="color: #495057;">5 Tingkat Diagnosis</h4>
                <p style="color: #6c757d;">Dari normal hingga proliferatif</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Modern Footer
    st.markdown("""
    <div class="footer">
        <div style="margin-bottom: 1rem;">
            <h4 style="color: #495057; margin-bottom: 0.5rem;">⚠️ Disclaimer Medis</h4>
            <p style="margin-bottom: 1rem;">
                Sistem ini adalah alat bantu diagnosis dan <strong>tidak menggantikan</strong> konsultasi medis profesional.
                Selalu konsultasikan hasil dengan dokter mata spesialis untuk diagnosis dan penanganan yang tepat.
            </p>
        </div>
        
        <hr style="border: none; height: 1px; background: rgba(102, 126, 234, 0.2); margin: 2rem 0;">
        
        <div style="display: flex; justify-content: center; align-items: center; gap: 2rem; flex-wrap: wrap;">
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🤖</div>
                <small><strong>AI Technology</strong><br>Deep Learning</small>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🏥</div>
                <small><strong>Medical Grade</strong><br>Clinical Standard</small>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🔒</div>
                <small><strong>Secure & Private</strong><br>Data Protection</small>
            </div>
        </div>
        
        <p style="margin-top: 2rem; font-size: 0.9rem; opacity: 0.7;">
            © 2024 AI Retina Scanner • Powered by TensorFlow & Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()