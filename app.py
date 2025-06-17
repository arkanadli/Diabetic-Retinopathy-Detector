import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import io
import os

# Import necessary custom metrics if they are part of your model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import InputLayer as OriginalInputLayer

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
def patch_input_layer():
    """
    Patches the InputLayer to handle 'batch_shape' arguments from older models.
    This modifies the global Keras custom objects registry.
    """
    class PatchedInputLayer(OriginalInputLayer):
        @classmethod
        def from_config(cls, config):
            if 'batch_shape' in config and 'input_shape' not in config:
                config['input_shape'] = config['batch_shape'][1:]
                # Remove batch_shape from config to avoid the TypeError
                del config['batch_shape']
            return super().from_config(config)
            
    # Register the patched layer in Keras's custom objects registry
    tf.keras.utils.get_custom_objects()['InputLayer'] = PatchedInputLayer
    print("InputLayer has been patched for batch_shape compatibility.")

# Call the patch function at the beginning of your script, before load_model
patch_input_layer()
# --- End of patching ---


# Konstanta (unchanged)
IMG_SIZE = 224

# Konfigurasi halaman (unchanged)
st.set_page_config(
    page_title="Deteksi Retinopati Diabetik",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fungsi preprocessing (unchanged)
def resize_img(img, size=224):
    """Resize image to specified size"""
    return cv2.resize(img, (size, size))

def crop_all_sides(img, crop_ratio=0.1):
    """Crop all sides of the image"""
    h, w = img.shape[:2]
    crop_h = int(h * crop_ratio)
    crop_w = int(w * crop_ratio)
    return img[crop_h:h-crop_h, crop_w:w-crop_w]

def create_retina_mask(img, threshold=10):
    """Create mask for retina area (non-black pixels)"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    return mask.astype(np.uint8) * 255

def apply_black_background(img, mask):
    """Apply mask with black background"""
    result = img.copy()
    result[mask == 0] = [0, 0, 0]
    return result

def pad_to_square(img):
    """Pad image to make it square"""
    h, w = img.shape[:2]
    max_dim = max(h, w)
    
    # Calculate padding
    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2
    
    # Pad the image
    padded = cv2.copyMakeBorder(
        img, pad_h, max_dim - h - pad_h, pad_w, max_dim - w - pad_w,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    
    return padded

def preprocess_without_steps(img_array, sigmaX=10):
    """Preprocess image for model prediction"""
    img = img_array.copy()
    
    # Resize to target size
    img = resize_img(img, size=IMG_SIZE)
    
    # Crop all sides
    img = crop_all_sides(img)
    
    # Create retina mask excluding black pixels
    retina_mask = create_retina_mask(img)
    
    # Apply Ben Graham's contrast enhancement
    # Blur menggunakan Gaussian blur
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX)
    
    # Hitung difference dan tambahkan offset brightness
    img = cv2.addWeighted(img, 4.0, blurred, -4.0, 128)
    
    # Apply masking with black background
    img = apply_black_background(img, retina_mask)
    
    # Final resize
    img = resize_img(img, size=IMG_SIZE)
    
    # Pad to square
    img = pad_to_square(img)
    img = resize_img(img, size=IMG_SIZE)
    
    return img

def preprocess_with_steps(img_array, sigmaX=10):
    """Preprocess image and return steps for visualization"""
    steps = {}
    img = img_array.copy()
    
    steps['1. Original'] = img.copy()
    
    # Resize to target size
    img = resize_img(img, size=IMG_SIZE)
    steps['2. Resized 224x224'] = img.copy()
    
    # Crop all sides
    img = crop_all_sides(img)
    steps['3. Cropped Image'] = img.copy()
    
    # Create retina mask excluding black pixels
    retina_mask = create_retina_mask(img)
    steps['4. Retina Mask'] = cv2.cvtColor(retina_mask, cv2.COLOR_GRAY2RGB)
    
    # Apply Ben Graham's contrast enhancement
    # Blur menggunakan Gaussian blur
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX)
    
    # Hitung difference dan tambahkan offset brightness
    img = cv2.addWeighted(img, 4.0, blurred, -4.0, 128)
    
    # Apply masking with black background
    img = apply_black_background(img, retina_mask)
    steps['5. Enhanced (Ben Graham)'] = img.copy()
    
    # Final resize
    img = resize_img(img, size=IMG_SIZE)
    
    # Pad to square
    img = pad_to_square(img)
    img = resize_img(img, size=IMG_SIZE)
    steps['6. Final Preprocessed'] = img.copy()
    
    return img, steps

@st.cache_resource
def load_trained_model():
    """Load the trained model"""
    model_path = 'BestModel.h5'
    
    if not os.path.exists(model_path):
        st.error(f"❌ File model '{model_path}' tidak ditemukan!")
        st.info("📝 Pastikan file 'BestModel.h5' ada di direktori yang sama dengan aplikasi ini.")
        return None
    
    try:
        # Define custom objects for loading the model
        custom_objects = {
            'accuracy': tf.keras.metrics.Accuracy(),
            # Ensure these names match exactly what was used during saving
            'auc_1': tf.keras.metrics.AUC(name='auc_1'),
            'precision_2': tf.keras.metrics.Precision(name='precision_2'),
            'recall_2': tf.keras.metrics.Recall(name='recall_2'),
            'F1Score': F1Score(), # Instantiate your custom F1Score class
            # Use tf.keras.Policy instead of tf.keras.mixed_precision.DTypePolicy
            # This is the correct class in newer Keras versions for policy handling
            'DTypePolicy': tf.keras.Policy
        }

        with st.spinner("Loading model..."):
            # Load the model with custom_objects
            # The global patch for InputLayer will handle the batch_shape issue
            model = load_model(model_path, custom_objects=custom_objects, compile=False)
        st.success(f"✅ Model berhasil dimuat dari '{model_path}'")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.exception(e) # Display full exception for debugging
        return None

def predict_retinopathy(model, processed_img):
    """Make prediction using the loaded model"""
    # Normalize pixel values to [0, 1]
    img_normalized = processed_img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Make prediction
    prediction = model.predict(img_batch, verbose=0)
    
    return prediction[0]

def get_severity_info(class_idx):
    """Get information about each severity level"""
    severity_info = {
        0: {
            "name": "No DR (Tidak Ada Retinopati Diabetik)",
            "description": "Tidak ada tanda-tanda retinopati diabetik yang terdeteksi.",
            "color": "#28a745",
            "recommendation": "Tetap jaga kontrol gula darah dan lakukan pemeriksaan rutin."
        },
        1: {
            "name": "Mild DR (Retinopati Diabetik Ringan)",
            "description": "Terdapat mikroaneurisma pada retina.",
            "color": "#ffc107",
            "recommendation": "Konsultasi dengan dokter mata dan jaga kontrol gula darah lebih ketat."
        },
        2: {
            "name": "Moderate DR (Retinopati Diabetik Sedang)",
            "description": "Terdapat mikroaneurisma, perdarahan, dan eksudasi.",
            "color": "#fd7e14",
            "recommendation": "Segera konsultasi dengan dokter mata spesialis retina."
        },
        3: {
            "name": "Severe DR (Retinopati Diabetik Berat)",
            "description": "Terdapat banyak perdarahan retina dan cotton wool spots.",
            "color": "#dc3545",
            "recommendation": "Segera konsultasi dengan dokter mata spesialis retina untuk penanganan intensif."
        },
        4: {
            "name": "Proliferative DR (Retinopati Diabetik Proliferatif)",
            "description": "Terdapat neovaskularisasi dan risiko tinggi kehilangan penglihatan.",
            "color": "#6f42c1",
            "recommendation": "SEGERA konsultasi dengan dokter mata spesialis retina. Mungkin diperlukan terapi laser atau pembedahan."
        }
    }
    return severity_info.get(class_idx, severity_info[0])

# Interface utama (unchanged)
def main():
    # Header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin-bottom: 0.5rem;">👁️ Sistem Deteksi Retinopati Diabetik</h1>
        <p style="color: white; text-align: center; opacity: 0.9;">Menggunakan Deep Learning dengan EfficientNetB0</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Informasi Model")
        st.markdown("""
        **Arsitektur Model:**
        - Base Model: EfficientNetB0
        - Global Average Pooling
        - Dense Layer (128 units)
        - Batch Normalization
        - Dropout
        - Output Layer (5 classes)
        
        **Preprocessing:**
        - Resize & Crop
        - Ben Graham Enhancement
        - Gaussian Blur
        - Retina Masking
        """)
        
        st.header("⚙️ Parameter")
        sigma_x = st.slider("Sigma X (Gaussian Blur)", 5, 20, 10, 1)
        show_steps = st.checkbox("Tampilkan Tahapan Preprocessing", value=True)
    
    # Load model
    model = load_trained_model()
    
    if model is None:
        st.warning("⚠️ Model tidak dapat dimuat. Upload file model atau periksa path file.")
        
        # Option to upload model file
        st.subheader("📤 Upload Model File")
        uploaded_model = st.file_uploader(
            "Upload file BestModel.h5", 
            type=['h5'], 
            help="Upload file model yang sudah dilatih"
        )
        
        if uploaded_model is not None:
            try:
                # Save uploaded model temporarily
                temp_model_path = "BestModel.h5"
                with open(temp_model_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
                
                # Reload model
                model = load_trained_model()
                if model is not None:
                    # Clean up the temporary file after successful load
                    if os.path.exists(temp_model_path):
                        os.remove(temp_model_path)
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Error saving or reloading model: {str(e)}")
                st.exception(e) # Show full traceback for upload errors too
        
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Upload Gambar Fundus Retina",
        type=['png', 'jpg', 'jpeg'],
        help="Upload gambar fundus retina untuk dianalisis"
    )
    
    if uploaded_file is not None:
        try:
            # Load dan konversi gambar
            pil_image = Image.open(uploaded_file)
            img_array = np.array(pil_image)
            
            # Konversi ke RGB jika perlu
            if len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 3:  # Already RGB
                pass
            else:  # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🖼️ Gambar Original")
                st.image(img_array, caption="Gambar yang diupload", use_column_width=True)
            
            # Preprocessing
            if show_steps:
                processed_img, steps = preprocess_with_steps(img_array, sigma_x)
                
                with col2:
                    st.subheader("🔄 Tahapan Preprocessing")
                    
                    # Tampilkan tahapan preprocessing
                    step_names = list(steps.keys())
                    selected_step = st.selectbox("Pilih tahapan:", step_names, index=len(step_names)-1)
                    st.image(steps[selected_step], caption=selected_step, use_column_width=True)
                
                # Tampilkan semua tahapan dalam grid
                st.subheader("📋 Semua Tahapan Preprocessing")
                cols = st.columns(3)
                for i, (step_name, step_img) in enumerate(steps.items()):
                    with cols[i % 3]:
                        st.image(step_img, caption=step_name, use_column_width=True)
                        
            else:
                processed_img = preprocess_without_steps(img_array, sigma_x)
                with col2:
                    st.subheader("🔄 Gambar Setelah Preprocessing")
                    st.image(processed_img, caption="Siap untuk prediksi", use_column_width=True)
            
            # Prediksi
            if st.button("🔍 Analisis Retinopati Diabetik", type="primary"):
                with st.spinner("Sedang menganalisis gambar..."):
                    try:
                        # Prediksi
                        predictions = predict_retinopathy(model, processed_img)
                        predicted_class = np.argmax(predictions)
                        confidence = predictions[predicted_class]
                        
                        # Informasi hasil prediksi
                        severity_info = get_severity_info(predicted_class)
                        
                        # Tampilkan hasil
                        st.markdown("---")
                        st.subheader("📋 Hasil Analisis")
                        
                        # Card hasil utama
                        st.markdown(f"""
                        <div style="background-color: {severity_info['color']}; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
                            <h3 style="color: white; margin-bottom: 0.5rem;">{severity_info['name']}</h3>
                            <p style="color: white; margin-bottom: 0.5rem; opacity: 0.9;">{severity_info['description']}</p>
                            <p style="color: white; font-weight: bold;">Confidence: {confidence:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Rekomendasi
                        st.markdown(f"""
                        <div style="background-color: #f8f9fa; padding: 1rem; border-left: 4px solid {severity_info['color']}; margin: 1rem 0;">
                            <h4>💡 Rekomendasi:</h4>
                            <p>{severity_info['recommendation']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Grafik probabilitas
                        st.subheader("📊 Distribusi Probabilitas")
                        
                        class_names = [
                            "No DR",
                            "Mild DR", 
                            "Moderate DR",
                            "Severe DR",
                            "Proliferative DR"
                        ]
                        
                        # Bar chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = ax.bar(class_names, predictions)
                        
                        # Highlight predicted class
                        bars[predicted_class].set_color(severity_info['color'])
                        
                        ax.set_ylabel('Probabilitas')
                        ax.set_title('Distribusi Probabilitas Tingkat Keparahan')
                        ax.set_ylim(0, 1)
                        
                        # Add percentage labels on bars
                        for i, (bar, prob) in enumerate(zip(bars, predictions)):
                            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                                    f'{prob:.1%}', ha='center', va='bottom')
                        
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Detail probabilitas
                        st.subheader("📈 Detail Probabilitas")
                        prob_df = {
                            "Tingkat Keparahan": class_names,
                            "Probabilitas": [f"{p:.4f}" for p in predictions],
                            "Persentase": [f"{p:.1%}" for p in predictions]
                        }
                        st.dataframe(prob_df, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error saat melakukan prediksi: {str(e)}")
            
        except Exception as e:
            st.error(f"❌ Error saat memproses gambar: {str(e)}")
    
    # Footer (unchanged)
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>⚠️ <strong>Disclaimer:</strong> Sistem ini adalah alat bantu diagnosis dan tidak menggantikan konsultasi medis profesional.</p>
        <p>Selalu konsultasikan hasil dengan dokter mata spesialis.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()