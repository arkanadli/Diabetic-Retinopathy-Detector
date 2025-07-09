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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Define your custom metrics if they are not built-in Keras metrics
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

# --- Patching InputLayer for compatibility ---
from tensorflow.keras.layers import InputLayer as OriginalKerasInputLayer

def patch_input_layer():
    """
    Patches the InputLayer to handle 'batch_shape' arguments from older models.
    This modifies the global Keras custom objects registry.
    """
    class PatchedInputLayer(OriginalKerasInputLayer):
        @classmethod
        def from_config(cls, config):
            if 'batch_shape' in config and 'input_shape' not in config:
                config['input_shape'] = config['batch_shape'][1:]
                del config['batch_shape']
            return super().from_config(config)
            
    tf.keras.utils.get_custom_objects()['InputLayer'] = PatchedInputLayer

# --- Dummy DTypePolicy for compatibility ---
class DummyDTypePolicy:
    """A dummy class to act as a placeholder for DTypePolicy during deserialization."""
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

# Call the patch functions at the beginning of your script, before load_model
patch_input_layer()

# Constants
IMG_SIZE = 224

# Page configuration
st.set_page_config(
    page_title="AI Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for modern, responsive design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for theming */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --accent-color: #f093fb;
        --success-color: #00d4aa;
        --warning-color: #ffc107;
        --danger-color: #ff6b6b;
        --info-color: #74b9ff;
        --dark-bg: #0e1117;
        --card-bg: #1a1d29;
        --text-primary: #ffffff;
        --text-secondary: #b8bcc8;
        --border-color: #262730;
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --border-radius: 12px;
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Global styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main > div {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .hero-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 50%, var(--accent-color) 100%);
        padding: 3rem 2rem;
        border-radius: var(--border-radius);
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="2" fill="white" opacity="0.1"/><circle cx="20" cy="20" r="1" fill="white" opacity="0.1"/><circle cx="80" cy="30" r="1.5" fill="white" opacity="0.1"/></svg>');
        animation: float 20s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.9);
        margin-bottom: 0;
        font-weight: 400;
    }
    
    /* Card components */
    .info-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow);
        transition: var(--transition);
    }
    
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    
    .result-card {
        background: linear-gradient(135deg, var(--card-bg) 0%, rgba(102, 126, 234, 0.1) 100%);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
    }
    
    .severity-badge {
        display: inline-block;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .confidence-score {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, var(--info-color) 0%, rgba(116, 185, 255, 0.1) 100%);
        border: 1px solid var(--info-color);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid var(--info-color);
    }
    
    /* Statistics cards */
    .stat-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stat-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        text-align: center;
        transition: var(--transition);
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        color: var(--text-secondary);
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Upload area styling */
    .upload-area {
        background: var(--card-bg);
        border: 2px dashed var(--border-color);
        border-radius: var(--border-radius);
        padding: 2rem;
        text-align: center;
        transition: var(--transition);
        margin: 1rem 0;
    }
    
    .upload-area:hover {
        border-color: var(--primary-color);
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        border: none;
        border-radius: var(--border-radius);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        transition: var(--transition);
        box-shadow: var(--shadow);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: var(--card-bg);
        border-radius: var(--border-radius);
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Progress bar */
    .progress-container {
        background: var(--border-color);
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 8px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        border-radius: 10px;
        transition: var(--transition);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2rem;
        }
        
        .hero-subtitle {
            font-size: 1rem;
        }
        
        .hero-header {
            padding: 2rem 1rem;
        }
        
        .info-card, .result-card {
            padding: 1rem;
        }
        
        .stat-container {
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        }
    }
    
    /* File uploader styling */
    .stFileUploader > label {
        color: var(--text-primary) !important;
        font-weight: 600;
    }
    
    .stFileUploader div[data-testid="stFileUploaderDropzone"] {
        background: var(--card-bg);
        border: 2px dashed var(--border-color);
        border-radius: var(--border-radius);
        transition: var(--transition);
    }
    
    .stFileUploader div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--primary-color);
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Spinner customization */
    .stSpinner > div > div {
        border-top-color: var(--primary-color);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background: var(--card-bg);
        border-radius: var(--border-radius);
        overflow: hidden;
        box-shadow: var(--shadow);
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: var(--border-radius);
        border: none;
        box-shadow: var(--shadow);
    }
    
    /* Custom animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .pulse-animation {
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* Footer */
    .footer {
        background: var(--card-bg);
        border-top: 1px solid var(--border-color);
        padding: 2rem;
        margin-top: 3rem;
        text-align: center;
        border-radius: var(--border-radius);
    }
    
    /* Icon styling */
    .icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)

# --- Preprocessing functions ---
def crop_all_sides(img, tol=7):
    """Crop all sides (top, bottom, left, right) of the image to remove black borders"""
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
    """Pad image to make it square with a specified border"""
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=pad_color)
    return padded

def resize_img(img, size=224):
    """Resize image to specified size"""
    return cv2.resize(img, (size, size))

def create_retina_mask(img, threshold=15):
    """Create a mask for the retina excluding black pixels"""
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
    """Apply mask to image and set background to black"""
    if mask.ndim == 2 and img.ndim == 3:
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    else:
        mask_3ch = mask
    black_bg = np.zeros_like(img)
    result = np.where(mask_3ch == 255, img, black_bg)
    return result

def preprocess_image_for_prediction(img_array, sigmaX=10):
    """Apply all preprocessing steps to an image array."""
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
    """Load the trained model directly using load_model, with compatibility patches."""
    model_path = 'BestModel.h5'
    
    if not os.path.exists(model_path):
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

        with st.spinner("🤖 Loading AI model..."):
            model = load_model(model_path, custom_objects=custom_objects, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def predict_retinopathy(model, processed_img):
    """Make prediction using the loaded model"""
    img_normalized = processed_img.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    prediction = model.predict(img_batch, verbose=0)
    return prediction[0]

def get_severity_info(class_idx):
    """Get information about each severity level"""
    severity_info = {
        0: {
            "name": "No DR",
            "full_name": "No Diabetic Retinopathy",
            "description": "No signs of diabetic retinopathy detected. The retina appears healthy.",
            "color": "#00d4aa",
            "recommendation": "Maintain good blood sugar control and continue regular eye checkups.",
            "urgency": "Low",
            "icon": "✅"
        },
        1: {
            "name": "Mild DR",
            "full_name": "Mild Diabetic Retinopathy",
            "description": "Microaneurysms are present in the retina. Early stage of diabetic retinopathy.",
            "color": "#ffc107",
            "recommendation": "Consult with an eye doctor and maintain stricter blood sugar control.",
            "urgency": "Medium",
            "icon": "⚠️"
        },
        2: {
            "name": "Moderate DR",
            "full_name": "Moderate Diabetic Retinopathy",
            "description": "Microaneurysms, hemorrhages, and hard exudates are present.",
            "color": "#fd7e14",
            "recommendation": "Consult with a retinal specialist promptly for further evaluation.",
            "urgency": "Medium-High",
            "icon": "🔶"
        },
        3: {
            "name": "Severe DR",
            "full_name": "Severe Diabetic Retinopathy",
            "description": "Extensive retinal hemorrhages and cotton wool spots are present.",
            "color": "#dc3545",
            "recommendation": "URGENT: Consult with a retinal specialist immediately for intensive treatment.",
            "urgency": "High",
            "icon": "🚨"
        },
        4: {
            "name": "Proliferative DR",
            "full_name": "Proliferative Diabetic Retinopathy",
            "description": "Neovascularization is present with high risk of vision loss.",
            "color": "#6f42c1",
            "recommendation": "EMERGENCY: Immediate consultation with retinal specialist. Laser therapy or surgery may be required.",
            "urgency": "Critical",
            "icon": "🆘"
        }
    }
    return severity_info.get(class_idx, severity_info[0])

def create_probability_chart(predictions, class_names, predicted_class):
    """Create an interactive probability chart using Plotly"""
    colors = ['#444444'] * len(predictions)
    severity_info = get_severity_info(predicted_class)
    colors[predicted_class] = severity_info['color']
    
    fig = go.Figure(data=[
        go.Bar(
            x=class_names,
            y=predictions,
            marker_color=colors,
            text=[f'{p:.1%}' for p in predictions],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Prediction Probability Distribution",
        xaxis_title="Severity Level",
        yaxis_title="Probability",
        template="plotly_dark",
        showlegend=False,
        height=400,
        font=dict(family="Inter", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def display_info_section():
    """Display information about diabetic retinopathy"""
    st.markdown("""
    <div class="info-card">
        <h3>🔬 About Diabetic Retinopathy</h3>
        <p>Diabetic retinopathy is a diabetes complication that affects the eyes. It's caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina).</p>
        
        <div class="stat-container">
            <div class="stat-card">
                <div class="stat-value">285M</div>
                <div class="stat-label">People with diabetes worldwide</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">93M</div>
                <div class="stat-label">People with diabetic retinopathy</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">AI</div>
                <div class="stat-label">Powered detection system</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display sidebar with information and tips"""
    with st.sidebar:
        st.markdown("## 📊 Model Information")
        st.markdown("""
        <div class="sidebar-content">
            <h4>🤖 AI Model Details</h4>
            <ul>
                <li><strong>Architecture:</strong> EfficientNetB0</li>
                <li><strong>Input Size:</strong> 224x224 pixels</li>
                <li><strong>Classes:</strong> 5 severity levels</li>
                <li><strong>Training:</strong> Medical grade dataset</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("## 💡 Tips for Better Results")
        st.markdown("""
        <div class="sidebar-content">
            <h4>📸 Image Guidelines</h4>
            <ul>
                <li>Use high-quality fundus images</li>
                <li>Ensure proper lighting and focus</li>
                <li>Avoid blurry or low-resolution images</li>
                <li>Center the optic disc and macula</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("## ⚠️ Important Notice")
        st.markdown("""
        <div class="sidebar-content">
            <p><strong>Medical Disclaimer:</strong> This AI system is designed as a screening tool and should not replace professional medical diagnosis. Always consult with qualified healthcare professionals for medical advice.</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Display sidebar
    display_sidebar()
    
    # Hero header
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">🔬 AI-Powered Diabetic Retinopathy Detection</h1>
        <p class="hero-subtitle">Advanced machine learning system for early detection and classification of diabetic retinopathy severity</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Information section
    display_info_section()
    
    # Load model
    model = load_trained_model()
    
    if model is None:
        st.markdown("""
        <div class="info-card">
            <h3>⚠️ Model Not Found</h3>
            <p>The AI model file 'BestModel.h5' was not found. Please upload the model file to continue.</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_model = st.file_uploader(
            "📤 Upload Model File (BestModel.h5)", 
            type=['h5'], 
            help="Upload the trained model file"
        )
        
        if uploaded_model is not None:
            try:
                temp_model_path = "BestModel.h5"
                with open(temp_model_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
                
                model = load_trained_model()
                if model is not None:
                    st.success("✅ Model loaded successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
        return
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📁 Upload Fundus Image")
        
        uploaded_file = st.file_uploader(
            "Choose a fundus retina image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a high-quality fundus image for analysis"
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
                
                # Display original image
                st.markdown("### 🖼️ Original Image")
                st.image(img_array, caption="Uploaded fundus image", use_column_width=True)
                
                # Preprocessing
                processed_img = preprocess_image_for_prediction(img_array, sigmaX=10)
                
                # Analysis button
                if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                    with st.spinner("🤖 AI is analyzing the image..."):
                        try:
                            predictions = predict_retinopathy(model, processed_img)
                            predicted_class = np.argmax(predictions)
                            confidence = predictions[predicted_class]
                            
                            # Display results
                            severity_info = get_severity_info(predicted_class)
                            
                            st.markdown("---")
                            st.markdown("## 📋 Analysis Results")
                            
                            # Main result card
                            st.markdown(f"""
                            <div class="result-card">
                                <div class="severity-badge" style="background-color: {severity_info['color']}; color: white;">
                                    {severity_info['icon']} {severity_info['name']}
                                </div>
                                <h3 style="color: {severity_info['color']};">{severity_info['full_name']}</h3>
                                <p>{severity_info['description']}</p>
                                <div class="confidence-score" style="color: {severity_info['color']};">
                                    {confidence:.1%} Confidence
                                </div>
                                <p><strong>Urgency Level:</strong> {severity_info['urgency']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Recommendation card
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <h4>💡 Medical Recommendation</h4>
                                <p>{severity_info['recommendation']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Probability chart
                            st.markdown("### 📊 Detailed Probability Analysis")
                            
                            class_names = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
                            fig = create_probability_chart(predictions, class_names, predicted_class)
                            st.plotly_chart(fig, use_container_width=True)