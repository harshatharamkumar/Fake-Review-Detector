import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
from datetime import datetime
import os
import warnings
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

warnings.filterwarnings('ignore')

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        return True
    except:
        return False

# Try to import TensorFlow with graceful fallback
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="AI Text Classifier",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Simple, clean CSS
st.markdown("""
<style>
    /* Clean, modern styling */
    .main-title {
        text-align: center;
        color: #2c3e50;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    .positive {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: white;
        box-shadow: 0 8px 25px rgba(46, 204, 113, 0.3);
    }
    
    .negative {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        box-shadow: 0 8px 25px rgba(231, 76, 60, 0.3);
    }
    
    .fake {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);
        border-left: 5px solid #c0392b;
    }
    
    .normal {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: white;
        box-shadow: 0 8px 25px rgba(46, 204, 113, 0.3);
        border-left: 5px solid #27ae60;
    }
    
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #fff5f5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #e74c3c;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #f0fff4;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2ecc71;
        margin: 1rem 0;
    }
    
    .preprocessing-box {
        background: #f8f9ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #e1e5e9;
        padding: 1rem;
        font-size: 1rem;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Warning box for fake reviews */
    .warning-box {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #f39c12;
        margin: 1rem 0;
        text-align: center;
    }
    
    /* Info card styling */
    .info-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Feature highlight */
    .feature-highlight {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

class TextPreprocessor:
    """Handles all text preprocessing steps used during training"""
    
    def __init__(self):
        self.stopwords_list = None
        self.lemmatizer = None
        self.exclude = string.punctuation
        self.setup_nltk_components()
    
    def setup_nltk_components(self):
        """Initialize NLTK components"""
        try:
            self.stopwords_list = stopwords.words('english')
            self.lemmatizer = WordNetLemmatizer()
        except LookupError:
            st.warning("⚠️ NLTK data not found. Attempting to download...")
            if download_nltk_data():
                self.stopwords_list = stopwords.words('english')
                self.lemmatizer = WordNetLemmatizer()
                st.success("✅ NLTK data loaded successfully!")
            else:
                st.error("❌ Failed to download NLTK data. Preprocessing may not work correctly.")
    
    def remove_url(self, text):
        """Remove URLs from text"""
        return re.sub(r"http\S+", "", text)
    
    def remove_punctuation(self, text):
        """Remove punctuation from text"""
        return text.translate(str.maketrans("", "", self.exclude))
    
    def remove_stopwords(self, text):
        """Remove stopwords from text"""
        if self.stopwords_list is None:
            return text
        
        new_text = []
        for word in text.split():
            if word not in self.stopwords_list:
                new_text.append(word)
        return " ".join(new_text)
    
    def lemmatize_text(self, text):
        """Lemmatize words in text"""
        if self.lemmatizer is None:
            return text
        
        try:
            words = word_tokenize(text)
            lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
            return " ".join(lemmatized_words)
        except:
            # Fallback if tokenization fails
            words = text.split()
            lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
            return " ".join(lemmatized_words)
    
    def preprocess_text(self, text):
        """Apply all preprocessing steps in the same order as training"""
        if not text or not isinstance(text, str):
            return ""
        
        # Step 1: Convert to lowercase
        processed_text = text.lower()
        
        # Step 2: Remove URLs
        processed_text = self.remove_url(processed_text)
        
        # Step 3: Remove punctuation
        processed_text = self.remove_punctuation(processed_text)
        
        # Step 4: Remove stopwords
        processed_text = self.remove_stopwords(processed_text)
        
        # Step 5: Lemmatize
        processed_text = self.lemmatize_text(processed_text)
        
        # Clean up extra spaces
        processed_text = ' '.join(processed_text.split())
        
        return processed_text

class SimpleTextClassifier:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.max_length = 100  # Default value
        self.model_loaded = False
        self.preprocessor = TextPreprocessor()
        
    def load_model(self, model_path="best_lstm_model"):
        """Load model components"""
        try:
            # Check if files exist
            model_file = f"{model_path}.h5"
            tokenizer_file = f"{model_path}_tokenizer.pkl"
            label_file = f"{model_path}_label_encoder.pkl"
            params_file = f"{model_path}_params.pkl"
            
            missing_files = []
            for file in [model_file, tokenizer_file, label_file, params_file]:
                if not os.path.exists(file):
                    missing_files.append(file)
            
            if missing_files:
                return False, f"Missing files: {', '.join(missing_files)}"
            
            if not TF_AVAILABLE:
                return False, "TensorFlow not available"
            
            # Load components
            self.model = tf.keras.models.load_model(model_file)
            
            with open(tokenizer_file, 'rb') as f:
                self.tokenizer = pickle.load(f)
            
            with open(label_file, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            with open(params_file, 'rb') as f:
                params = pickle.load(f)
                self.max_length = params.get('max_length', 100)
            
            self.model_loaded = True
            return True, "Model loaded successfully!"
            
        except Exception as e:
            return False, f"Error loading model: {str(e)}"
    
    def predict(self, text):
        """Make prediction on text with proper preprocessing"""
        if not self.model_loaded:
            return None, None, None
        
        try:
            # Store original text for display
            original_text = text
            
            # Apply the same preprocessing used during training
            preprocessed_text = self.preprocessor.preprocess_text(text)
            
            if not preprocessed_text.strip():
                return None, None, "Text became empty after preprocessing"
            
            # Tokenize and pad
            sequences = self.tokenizer.texts_to_sequences([preprocessed_text])
            padded = pad_sequences(sequences, maxlen=self.max_length, padding='post')
            
            # Get prediction
            prediction = self.model.predict(padded, verbose=0)
            
            # Handle binary vs multi-class
            if len(prediction[0]) == 1:
                # Binary classification
                confidence = float(prediction[0][0])
                predicted_class = int(confidence > 0.5)
                display_confidence = confidence if predicted_class == 1 else (1 - confidence)
            else:
                # Multi-class classification
                predicted_class = np.argmax(prediction[0])
                display_confidence = float(np.max(prediction[0]))
            
            # Get label
            predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
            
            return str(predicted_label), display_confidence, preprocessed_text
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None, None, str(e)

def create_confidence_gauge(confidence, label):
    """Create a simple confidence gauge"""
    # Determine color based on fake vs normal
    if str(label) == '1':  # Fake review
        color = '#e74c3c'
        title_text = "Fake Review Probability"
    else:  # Normal review
        color = '#2ecc71' 
        title_text = "Authentic Review Probability"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        title = {'text': title_text, 'font': {'size': 18, 'color': '#2c3e50'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickcolor': '#34495e'},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#bdc3c7",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(46, 204, 113, 0.2)'},
                {'range': [30, 70], 'color': 'rgba(241, 196, 15, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(231, 76, 60, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#e67e22", 'width': 3},
                'thickness': 0.8,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        height=280, 
        margin=dict(l=20, r=20, t=60, b=20),
        font={'color': "#2c3e50", 'family': "Arial"},
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def main():
    # Download NLTK data on startup
    download_nltk_data()
    
    # Initialize classifier
    if 'classifier' not in st.session_state:
        st.session_state.classifier = SimpleTextClassifier()
        st.session_state.model_status = "not_loaded"
    
    # Header
    st.markdown('<h1 class="main-title">🕵️ Fake Review Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered authenticity verification for product reviews</p>', unsafe_allow_html=True)
    
    # Model loading section
    if st.session_state.model_status == "not_loaded":
        st.markdown("""
        <div class="info-box">
            <h4>🤖 AI Model Loading</h4>
            <p>Click the button below to load your trained fake review detection model. Make sure these files are in your directory:</p>
            <ul>
                <li><code>best_lstm_model.h5</code></li>
                <li><code>best_lstm_model_tokenizer.pkl</code></li>
                <li><code>best_lstm_model_label_encoder.pkl</code></li>
                <li><code>best_lstm_model_params.pkl</code></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
                if st.button("🚀 Load Detection Model"):
                    with st.spinner("Loading model and NLTK components..."):
                        success, message = st.session_state.classifier.load_model()
                        if success:
                            st.session_state.model_status = "loaded"
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                            
                            # Show TensorFlow installation help if needed
                            if "TensorFlow not available" in message:
                                st.markdown("""
                                <div class="error-box">
                                    <h4>⚠️ TensorFlow Installation Required</h4>
                                    <p>Install TensorFlow with:</p>
                                    <code>pip install tensorflow</code>
                                    <p>Or for CPU-only version:</p>
                                    <code>pip install tensorflow-cpu</code>
                                </div>
                                """, unsafe_allow_html=True)
    
    elif st.session_state.model_status == "loaded":
        # Model loaded successfully
        st.markdown("""
        <div class="feature-highlight">
            <h3>✅ Fake Review Detection Ready!</h3>
            <p>AI model loaded successfully! Paste any product review below to check its authenticity.</p>
            <p><strong>🔧 Preprocessing Applied:</strong> Text will be automatically cleaned and processed exactly as during training.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature information
        st.markdown("""
        <div class="info-card">
            <h4>🎯 How It Works</h4>
            <p><strong>🔍 Advanced AI Analysis:</strong> Our LSTM neural network analyzes text patterns, language structure, and authenticity markers to detect fake reviews.</p>
            <p><strong>🔧 Smart Preprocessing:</strong> Text is automatically cleaned using the same steps as training: lowercase conversion, URL removal, punctuation removal, stopword removal, and lemmatization.</p>
            <p><strong>📊 Confidence Scoring:</strong> Get detailed probability scores showing how likely a review is to be fake or authentic.</p>
            <p><strong>⚡ Instant Results:</strong> Real-time detection with professional-grade accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Text input section
        st.markdown("### 📝 Review Analysis")
        
        # Text input
        user_input = st.text_area(
            "🔍 Paste Product Review Here:",
            value="",
            height=150,
            placeholder="Paste the product review you want to analyze here...\n\nExample:\n'This product is absolutely amazing! Best purchase ever made. 5 stars definitely recommend to everyone!!!'",
            help="Enter any product review text to check if it's authentic or potentially fake"
        )
        
        # Analysis section
        if user_input.strip():
            col1, col2 = st.columns([2, 1])
            
            with col1:
                analyze_button = st.button("🔍 Analyze Review")
            
            with col2:
                st.markdown(f"**📊 {len(user_input)} characters, {len(user_input.split())} words**")
            
            if analyze_button:
                with st.spinner("🔍 Analyzing review authenticity..."):
                    prediction, confidence, preprocessed_text = st.session_state.classifier.predict(user_input.strip())
                    
                    if prediction is not None:
                        # Show preprocessing results
                        st.markdown("### 🔧 Text Preprocessing")
                        st.markdown(f"""
                        <div class="preprocessing-box">
                            <strong>Original:</strong> {user_input.strip()}<br><br>
                            <strong>Preprocessed:</strong> {preprocessed_text}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show results
                        st.markdown("---")
                        st.markdown("### 📊 Detection Results")
                        
                        # Determine result style based on 0/1 labels
                        if str(prediction) == '1':  # Fake review
                            result_class = "fake"
                            emoji = "⚠️"
                            result_text = "FAKE REVIEW DETECTED"
                            description = "This review shows characteristics of being artificially generated or manipulated."
                        else:  # Normal/authentic review
                            result_class = "normal"
                            emoji = "✅"
                            result_text = "AUTHENTIC REVIEW"
                            description = "This review appears to be genuine and written by a real customer."
                        
                        # Result display
                        st.markdown(f"""
                        <div class="result-card {result_class}">
                            <h2>{emoji} {result_text}</h2>
                            <p>{description}</p>
                            <h3>Confidence: {confidence*100:.1f}%</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Warning for fake reviews
                        if str(prediction) == '1':
                            st.markdown(f"""
                            <div class="warning-box">
                                <h4>🚨 Warning: Potentially Fake Review</h4>
                                <p>Our AI model detected patterns commonly found in fake reviews. Consider this when making purchasing decisions.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Confidence gauge
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            fig = create_confidence_gauge(confidence, prediction)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Additional analysis info
                        st.markdown("""
                        <div class="info-card">
                            <h4>📈 Analysis Details</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Stats columns
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("📝 Original Chars", len(user_input.strip()))
                        
                        with col2:
                            st.metric("🔧 Processed Words", len(preprocessed_text.split()) if preprocessed_text else 0)
                        
                        with col3:
                            confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
                            st.metric("🎯 Confidence", confidence_level)
                        
                        with col4:
                            risk_level = "High Risk" if (str(prediction) == '1' and confidence > 0.7) else "Low Risk"
                            st.metric("⚠️ Risk Level", risk_level)
                        
                        # Timestamp
                        st.markdown(f"""
                        <div style="text-align: center; margin-top: 1.5rem; color: #7f8c8d; font-size: 0.9rem;">
                            Analysis completed at {datetime.now().strftime("%H:%M:%S on %B %d, %Y")}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    else:
                        st.error(f"❌ Analysis failed: {preprocessed_text}")
        
        # Reset button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Load Different Model"):
                st.session_state.model_status = "not_loaded"
                st.session_state.classifier = SimpleTextClassifier()
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 1.5rem;">
        <p>🕵️ <strong>Fake Review Detector</strong> | AI-Powered Authenticity Verification</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem;">Protecting consumers from deceptive reviews with advanced machine learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()