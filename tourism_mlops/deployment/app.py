
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from huggingface_hub import hf_hub_download
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Tourism Package Prediction", 
    page_icon="✈️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 10px;
        text-align: center;
    }
    .positive-prediction {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .negative-prediction {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Cache the model loading function
@st.cache_resource
def load_model():
    """Load the best model from HuggingFace Hub"""
    try:
        # Download model from HuggingFace Hub
        model_path = hf_hub_download(
            repo_id="shashidj/tourism-package-prediction-model",
            filename="model.pkl"
        )
        
        # Load the model
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Feature encoding mappings (based on training data)
FEATURE_MAPPINGS = {
    'TypeofContact': {'Company Invited': 0, 'Self Inquiry': 1},
    'Occupation': {
        'Salaried': 0, 'Small Business': 1, 'Large Business': 2, 
        'Free Lancer': 3, 'Student': 4
    },
    'Gender': {'Female': 0, 'Male': 1},
    'ProductPitched': {
        'Basic': 0, 'Standard': 1, 'Deluxe': 2, 'Super Deluxe': 3, 'King': 4
    },
    'MaritalStatus': {'Single': 0, 'Married': 1, 'Divorced': 2, 'Unmarried': 3},
    'Designation': {
        'Executive': 0, 'Manager': 1, 'Senior Manager': 2, 'AVP': 3, 'VP': 4
    }
}

def encode_features(input_data):
    """Encode categorical features using predefined mappings"""
    encoded_data = input_data.copy()
    
    for feature, mapping in FEATURE_MAPPINGS.items():
        if feature in encoded_data.columns:
            encoded_data[feature] = encoded_data[feature].map(mapping)
    
    return encoded_data

def predict_tourism_package(model, input_data):
    """Make prediction using the loaded model"""
    try:
        # Encode categorical features
        encoded_data = encode_features(input_data)
        
        # Make prediction
        prediction = model.predict(encoded_data)[0]
        prediction_proba = model.predict_proba(encoded_data)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
        
        return prediction, prediction_proba
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def main():
    # Header
    st.markdown('<div class="main-header">✈️ Tourism Package Purchase Prediction</div>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check your configuration.")
        return
    
    # Sidebar for model information
    st.sidebar.header("📊 Model Information")
    st.sidebar.info("""
    **Algorithm**: Best performing model from 6 algorithms  
    **Features**: Customer demographics and behavior  
    **Purpose**: Predict tourism package purchase likelihood  
    **Accuracy**: Optimized through hyperparameter tuning
    """)
    
    # Main interface
    st.header("Customer Information Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔢 Demographic Information")
        
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        
        gender = st.selectbox("Gender", options=list(FEATURE_MAPPINGS['Gender'].keys()))
        
        marital_status = st.selectbox(
            "Marital Status", 
            options=list(FEATURE_MAPPINGS['MaritalStatus'].keys())
        )
        
        occupation = st.selectbox(
            "Occupation", 
            options=list(FEATURE_MAPPINGS['Occupation'].keys())
        )
        
        designation = st.selectbox(
            "Designation", 
            options=list(FEATURE_MAPPINGS['Designation'].keys())
        )
    
    with col2:
        st.subheader("📞 Interaction & Financial Details")
        
        type_of_contact = st.selectbox(
            "Type of Contact", 
            options=list(FEATURE_MAPPINGS['TypeofContact'].keys())
        )
        
        product_pitched = st.selectbox(
            "Product Pitched", 
            options=list(FEATURE_MAPPINGS['ProductPitched'].keys())
        )
        
        duration_of_pitch = st.number_input(
            "Duration of Pitch (minutes)", 
            min_value=1, max_value=300, value=30
        )
        
        monthly_income = st.number_input(
            "Monthly Income", 
            min_value=0, max_value=1000000, value=50000
        )
        
        number_of_followups = st.number_input(
            "Number of Follow-ups", 
            min_value=0, max_value=20, value=3
        )
    
    # Additional features
    st.subheader("📈 Additional Information")
    
    col3, col4 = st.columns(2)
    
    with col3:
        number_of_trips = st.number_input(
            "Number of Trips", 
            min_value=0, max_value=50, value=2
        )
        
        number_of_children_visiting = st.number_input(
            "Number of Children Visiting", 
            min_value=0, max_value=10, value=1
        )
    
    with col4:
        passport = st.selectbox("Passport", options=[0, 1], format_func=lambda x: "Yes" if x else "No")
        
        own_car = st.selectbox("Own Car", options=[0, 1], format_func=lambda x: "Yes" if x else "No")
    
    # Prediction button
    if st.button("🔮 Predict Package Purchase", type="primary"):
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'Age': [age],
            'TypeofContact': [type_of_contact],
            'DurationOfPitch': [duration_of_pitch],
            'Occupation': [occupation],
            'Gender': [gender],
            'NumberOfFollowups': [number_of_followups],
            'ProductPitched': [product_pitched],
            'MonthlyIncome': [monthly_income],
            'NumberOfTrips': [number_of_trips],
            'Passport': [passport],
            'OwnCar': [own_car],
            'NumberOfChildrenVisiting': [number_of_children_visiting],
            'MaritalStatus': [marital_status],
            'Designation': [designation]
        })
        
        # Make prediction
        prediction, prediction_proba = predict_tourism_package(model, input_data)
        
        if prediction is not None:
            # Display results
            st.header("🎯 Prediction Results")
            
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box positive-prediction">
                    <h3>✅ HIGH LIKELIHOOD TO PURCHASE</h3>
                    <p><strong>Confidence:</strong> {prediction_proba[1]:.2%}</p>
                    <p>This customer shows strong indicators for tourism package purchase!</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.success("💡 Recommendation: Prioritize this lead for follow-up and personalized offers!")
                
            else:
                st.markdown(f"""
                <div class="prediction-box negative-prediction">
                    <h3>❌ LOW LIKELIHOOD TO PURCHASE</h3>
                    <p><strong>Confidence:</strong> {prediction_proba[0]:.2%}</p>
                    <p>This customer may need additional nurturing before conversion.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.warning("💡 Recommendation: Focus on building relationship and addressing concerns before pitching.")
            
            # Probability visualization
            fig = go.Figure(data=[
                go.Bar(
                    x=['Will NOT Purchase', 'Will Purchase'],
                    y=[prediction_proba[0], prediction_proba[1]],
                    marker_color=['#ff7f7f', '#90ee90']
                )
            ])
            
            fig.update_layout(
                title="Purchase Probability Distribution",
                xaxis_title="Prediction",
                yaxis_title="Probability",
                yaxis=dict(range=[0, 1], tickformat='.0%')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance insight
            st.subheader("📋 Customer Summary")
            summary_data = {
                'Feature': ['Age', 'Monthly Income', 'Product Type', 'Follow-ups', 'Contact Type'],
                'Value': [age, f"${monthly_income:,}", product_pitched, number_of_followups, type_of_contact]
            }
            summary_df = pd.DataFrame(summary_data)
            st.table(summary_df)

    # Footer information
    st.markdown("---")
    st.markdown("""
    **About this Application:**  
    This Tourism Package Prediction system uses machine learning to analyze customer behavior and predict purchase likelihood.  
    Built with MLOps best practices including experiment tracking, model versioning, and automated deployment.
    
    **🔧 Technical Stack:** Python • Scikit-learn • MLflow • HuggingFace • Streamlit • Docker
    """)

if __name__ == "__main__":
    main()
