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
    print("InputLayer has been patched for batch_shape compatibility.")

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

# Konstanta
IMG_SIZE = 224

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Retinopati Diabetik",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Combined preprocessing function, no longer returning steps
def preprocess_image_for_prediction(img_array, sigmaX=10):
    """Apply all preprocessing steps to an image array."""
    img = img_array.copy()
    img = resize_img(img, size=IMG_SIZE) # Initial resize is actually done here before other steps

    # Crop to remove black borders
    img = crop_all_sides(img)
    
    # Create retina mask and apply black background
    retina_mask = create_retina_mask(img)
    img = apply_black_background(img, retina_mask) # Apply black background based on mask

    # Ben Graham Enhancement (Gaussian Blur and addWeighted)
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX)
    img = cv2.addWeighted(img, 4.0, blurred, -4.0, 128)

    # Resize and pad to square
    img = resize_img(img, size=IMG_SIZE) # Resize again after enhancement
    img = pad_to_square(img)
    img = resize_img(img, size=IMG_SIZE) # Final resize to IMG_SIZE

    return img


@st.cache_resource
def load_trained_model():
    """Load the trained model directly using load_model, with compatibility patches."""
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

        with st.spinner("Loading model..."):
            model = load_model(model_path, custom_objects=custom_objects, compile=False)
        # st.success(f"✅ Model berhasil dimuat dari '{model_path}'")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.exception(e)
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

def main():
    # Header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h1 style="color: white; text-align: center; margin-bottom: 0.5rem;">Sistem Deteksi Retinopati Diabetik</h1>
        <p style="color: white; text-align: center; opacity: 0.9;"></p>
    </div>
    """, unsafe_allow_html=True)
    
    # # Sidebar
    # with st.sidebar:
    #     st.header("📊 Informasi Model")
    #     st.markdown("""
    #     **Arsitektur Model:**
    #     - Base Model: EfficientNetB0
    #     - Global Average Pooling
    #     - Dropout
    #     - Dense Layer
    #     - Batch Normalization
    #     - Dropout
    #     - Output Layer (5 kelas)
        
    #     **Tahapan Preprocessing Citra:**
    #     - **Cropping** 
    #     - **Masking Retina** 
    #     - **Apply Black Background** 
    #     - **Image Sharpening**
    #     - **Resizing ke 224x224 piksel**
    #     - **Penambahan Padding** 
    #     """)
        
    # Load model
    model = load_trained_model()
    
    if model is None:
        st.warning("⚠️ Model tidak dapat dimuat. Upload file model atau periksa path file.")
        
        st.subheader("📤 Upload Model File")
        uploaded_model = st.file_uploader(
            "Upload file BestModel.h5", 
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
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Error saving or reloading model: {str(e)}")
                st.exception(e)
            
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
            
            # Display only the original image
            st.subheader("🖼️ Gambar Original")
            st.image(img_array, caption="Gambar yang diupload", use_column_width=True)
            
            # Preprocessing (happens internally without display)
            # The sigmaX value is now fixed at 10 within the preprocess_image_for_prediction function
            processed_img = preprocess_image_for_prediction(img_array, sigmaX=10) 
            
            # Prediction button
            if st.button("🔍 Analisis Retinopati Diabetik", type="primary"):
                with st.spinner("Sedang menganalisis gambar..."):
                    try:
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
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>⚠️ <strong>Disclaimer:</strong> Sistem ini adalah alat bantu diagnosis dan tidak menggantikan konsultasi medis profesional.</p>
        <p>Selalu konsultasikan hasil dengan dokter mata spesialis.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()