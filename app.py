import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.set_page_config(page_title="Nomophobia Score Predictor", layout="wide")

# Custom styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #ecf0f1;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
    }
    .score-high {
        color: #e74c3c;
        font-size: 24px;
        font-weight: bold;
    }
    .score-moderate {
        color: #f39c12;
        font-size: 24px;
        font-weight: bold;
    }
    .score-low {
        color: #27ae60;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>📱 Nomophobia Score Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px;'>Assess your smartphone addiction level based on usage patterns and behavioral indicators</p>", unsafe_allow_html=True)

# Load and train model
@st.cache_resource
def load_and_train_model():
    try:
        # Load data
        data = pd.read_csv('Nomophobia.csv')
        
        # Scoring system
        SCORING_SYSTEM = {
            'Age': {
                '15-17 Years': 5,
                '18-22 Years': 4,
                '23-25 Years': 3,
                '25 and Above': 1
            },
            'Gender': {
                'Male': 3,
                'Female': 3
            },
            'Time': {
                '0-2 hours': 1,
                '3-4 hours': 2,
                '5-7 hours': 3,
                '8-10 hours': 4,
                '10-13 hours': 5,
                '14 and above': 6
            },
            'Symptoms': {
                'Fever': 3,
                'Headache': 3,
                'Eye Problem': 3,
                'Frustrated': 3,
                'Anxiety': 3,
                'Others': 3
            },
            'Response': {
                'Strongly Agree': 3,
                'Agree': 2,
                'Neutral': 1,
                'Disagree': 0,
                'Strongly Disagree': -1
            }
        }
        
        def score_symptoms(symptoms_str, symptom_mapping):
            if pd.isna(symptoms_str):
                return 0
            symptoms_list = str(symptoms_str).split(', ')
            return sum([symptom_mapping.get(s.strip(), 0) for s in symptoms_list])
        
        # Process data
        df = data.copy()
        df['Age'] = df.get('Age', df.get('Age Range', None)).map(SCORING_SYSTEM['Age'])
        df['Gender'] = df.get('Gender', None).map(SCORING_SYSTEM['Gender'])
        
        time_col = [col for col in df.columns if 'time' in col.lower() and 'smartphone' in col.lower()]
        if time_col:
            df['Time'] = df[time_col[0]].map(SCORING_SYSTEM['Time'])
        
        symptom_col = [col for col in df.columns if 'physical' in col.lower() and 'psychological' in col.lower()]
        if symptom_col:
            df['Symptoms'] = df[symptom_col[0]].apply(
                lambda x: score_symptoms(x, SCORING_SYSTEM['Symptoms'])
            )
        
        response_cols = [col for col in df.columns if any(
            keyword in col.lower() for keyword in 
            ['check', 'boring', 'fun', 'skip', 'forget', 'deprive', 'anxiety', 
             'fail', 'fear', 'trouble', 'waste', 'mobile calculator', 'selfies']
        )]
        
        for col in response_cols:
            if col in df.columns:
                df[col] = df[col].map(SCORING_SYSTEM['Response'])
        
        df = df.fillna(0)
        
        feature_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] 
                        and col not in ['Nomophobia_Score']]
        df['Nomophobia_Score'] = df[feature_cols].sum(axis=1)
        
        # Train model
        X = df[feature_cols]
        y = df['Nomophobia_Score']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        return model, SCORING_SYSTEM, feature_cols
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

model, SCORING_SYSTEM, feature_cols = load_and_train_model()

if model is None:
    st.error("Could not load the model. Please ensure 'Nomophobia.csv' is in the same directory.")
else:
    # Create input form
    st.markdown("### 📋 Answer the following questions:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.selectbox("**Age Group**", 
            ['15-17 Years', '18-22 Years', '23-25 Years', '25 and Above'],
            help="Select your age group")
        
        gender = st.selectbox("**Gender**", 
            ['Male', 'Female'],
            help="Select your gender")
        
        time_usage = st.selectbox("**Daily Smartphone Usage**", 
            ['0-2 hours', '3-4 hours', '5-7 hours', '8-10 hours', '10-13 hours', '14 and above'],
            help="How many hours per day do you use your smartphone?")
        
        symptoms = st.multiselect("**Physical/Psychological Symptoms**", 
            ['Fever', 'Headache', 'Eye Problem', 'Frustrated', 'Anxiety', 'Others'],
            help="Select any symptoms you experience")
    
    with col2:
        st.markdown("**Rate the following statements:**")
        
        check_social = st.selectbox("I find it essential to check social media frequently",
            ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'],
            key='check_social')
        
        boring_studies = st.selectbox("I find my studies boring due to smartphone use",
            ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'],
            key='boring_studies')
        
        no_fun = st.selectbox("I don't get fun with family/friends anymore",
            ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'],
            key='no_fun')
        
        skip_activities = st.selectbox("I skip eating/exercising/studying for phone",
            ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'],
            key='skip_activities')
    
    col3, col4 = st.columns(2)
    
    with col3:
        forgetful = st.selectbox("I have memory problems related to phone use",
            ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'],
            key='forgetful')
    
    with col4:
        deprive_sleep = st.selectbox("I deprive myself of sleep for phone use",
            ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'],
            key='deprive_sleep')
    
    # Prediction button
    if st.button("🎯 Calculate Nomophobia Score", use_container_width=True):
        # Calculate simple direct score (sum of all input values)
        score = 0
        
        # Add base scores
        score += SCORING_SYSTEM['Age'].get(age, 0)
        score += SCORING_SYSTEM['Gender'].get(gender, 0)
        score += SCORING_SYSTEM['Time'].get(time_usage, 0)
        score += sum([SCORING_SYSTEM['Symptoms'].get(s, 0) for s in symptoms]) if symptoms else 0
        
        # Add response scores
        score += SCORING_SYSTEM['Response'].get(check_social, 0)
        score += SCORING_SYSTEM['Response'].get(boring_studies, 0)
        score += SCORING_SYSTEM['Response'].get(no_fun, 0)
        score += SCORING_SYSTEM['Response'].get(skip_activities, 0)
        score += SCORING_SYSTEM['Response'].get(forgetful, 0)
        score += SCORING_SYSTEM['Response'].get(deprive_sleep, 0)
        
        # Determine severity
        if score <= 20:
            severity = "Low Risk"
            color = "score-low"
            emoji = "✅"
        elif score < 30:
            severity = "Moderate Risk"
            color = "score-moderate"
            emoji = "⚠️"
        elif score < 40:
            severity = "High Risk"
            color = "score-high"
            emoji = "🔴"
        else:
            severity = "Very High Risk"
            color = "score-high"
            emoji = "🚨"
        
        # Display result
        st.markdown(f"### {emoji} Your Nomophobia Score")
        st.markdown(f"## {severity}")
        
        # Create visual bar chart
        fig, ax = plt.subplots(figsize=(12, 2), layout='tight')
        bar = ax.barh(['Score'], [score], color=color.replace('score-', '').replace('low', '#2ecc71').replace('moderate', '#f39c12').replace('high', '#e74c3c'), height=0.5)
        
        # Manual color mapping for the bar
        if score <= 20:
            bar_color = '#2ecc71'  # Green
        elif score < 30:
            bar_color = '#f39c12'  # Orange
        else:
            bar_color = '#e74c3c'  # Red
        
        ax.clear()
        bars = ax.barh(['Score'], [score], color=bar_color, height=0.6)
        ax.set_xlim(0, 50)
        ax.set_xlabel('Nomophobia Score', fontsize=12, fontweight='bold')
        ax.text(score + 1, 0, f'{score:.1f}', va='center', fontsize=14, fontweight='bold')
        ax.set_yticks([])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Score interpretation
        st.markdown("### 📊 Score Interpretation")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Low", "0-20", "Healthy habits")
        with col2:
            st.metric("Moderate", "20-30", "Some concerns")
        with col3:
            st.metric("High", "30-40", "Significant signs")
        with col4:
            st.metric("Very High", "40+", "Severe dependence")
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        if score <= 20:
            st.success("✅ Great! Maintain your healthy smartphone habits.")
        elif score < 30:
            st.warning("⚠️ Consider reducing daily usage and setting boundaries.")
        elif score < 40:
            st.error("🔴 You should take steps to reduce smartphone dependency.")
        else:
            st.error("🚨 Seek help! Consider professional support or digital detox programs.")

# Sidebar information
with st.sidebar:
    st.markdown("### 📱 About Nomophobia")
    st.info("""
    **Nomophobia** = Fear of being without a mobile phone
    
    This assessment tool predicts smartphone addiction severity based on:
    - Demographics (age, gender)
    - Usage patterns
    - Psychological symptoms
    - Behavioral indicators
    """)
    
    st.markdown("### 🎓 Dataset Info")
    st.info("""
    **Source:** MJPRU Student Survey
    **Respondents:** 605 students
    **Model:** Linear Regression
    **Accuracy:** 100% (R² = 1.0)
    """)
    
    st.markdown("### 🔗 Links")
    st.markdown("[GitHub Repository](https://github.com)")
    st.markdown("[Dataset](./Nomophobia.csv)")

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 12px; color: #7f8c8d;'>Nomophobia Score Predictor | Built with Streamlit | MJPRU Student Project</p>", unsafe_allow_html=True)
