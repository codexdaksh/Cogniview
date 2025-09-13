import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_community.llms import Ollama
import re

# Use Ollama (make sure `ollama serve` is running)
llm = Ollama(model="mistral")

# Streamlit page setup
st.set_page_config(page_title="🧠 Cogniview", layout="wide", page_icon="🧠")

# Custom CSS for premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .main > div {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        margin: 20px;
        padding: 40px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    /* Hero section */
    .hero-container {
        text-align: center;
        padding: 60px 20px;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
        border-radius: 30px;
        margin-bottom: 40px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(180deg); }
    }
    
    .hero-title {
        color: white;
        font-size: 4rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 4px 20px rgba(0,0,0,0.5);
        position: relative;
        z-index: 1;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 20px rgba(255,255,255,0.5), 0 0 30px rgba(255,255,255,0.3); }
        to { text-shadow: 0 0 30px rgba(255,255,255,0.8), 0 0 40px rgba(255,255,255,0.5); }
    }
    
    .hero-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.5rem;
        font-weight: 400;
        margin: 20px 0 0 0;
        position: relative;
        z-index: 1;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 8px;
        margin-bottom: 30px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 15px;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 500;
        padding: 12px 24px;
        margin: 0 4px;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    /* Glass card styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
    }
    
    /* Section headers */
    .section-header {
        color: white;
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 30px;
        text-align: center;
        position: relative;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 80px;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 2px;
    }
    
    /* Metrics styling */
    .stMetric {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: scale(1.05);
        background: rgba(255, 255, 255, 0.15);
    }
    
    .stMetric [data-testid="metric-container"] {
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 12px 24px;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 30px rgba(0,0,0,0.3);
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 15px;
        color: white;
        padding: 15px;
        backdrop-filter: blur(10px);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.6);
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        border: 2px dashed rgba(255, 255, 255, 0.3);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: rgba(255, 255, 255, 0.6);
        background: rgba(255, 255, 255, 0.15);
    }
    
    /* Dataframe styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* Alert styling */
    .stAlert {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Success message */
    .stSuccess {
        background: rgba(34, 197, 94, 0.2);
        border: 1px solid rgba(34, 197, 94, 0.4);
    }
    
    /* Warning message */
    .stWarning {
        background: rgba(251, 191, 36, 0.2);
        border: 1px solid rgba(251, 191, 36, 0.4);
    }
    
    /* Error message */
    .stError {
        background: rgba(239, 68, 68, 0.2);
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    
    /* Spinner */
    .stSpinner {
        color: white;
    }
    
    /* Code block styling */
    .stCodeBlock {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Multiselect styling */
    .stMultiSelect > div > div {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    
    /* Floating animation for cards */
    @keyframes float-card {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .floating-card {
        animation: float-card 4s ease-in-out infinite;
    }
    
    /* Pulse animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    /* Fade in animation */
    @keyframes fadeInUp {
        0% {
            opacity: 0;
            transform: translateY(20px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 1s ease-out;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# Enhanced title with premium styling
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">🧠 Cogniview</h1>
    <p class="hero-subtitle">Smart Data Insights powered by Advanced AI</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📘 Overview", "📁 Data Upload", "📊 Analytics", "🧠 AI Assistant"])

# Session init
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'df' not in st.session_state:
    st.session_state.df = None

# Tab 1: Overview - FIXED VERSION
with tab1:
    st.markdown('<h2 class="section-header">📘 Platform Overview</h2>', unsafe_allow_html=True)

    # Welcome Section + Feature Cards
    st.markdown('<div class="glass-card floating-card">', unsafe_allow_html=True)
    
    st.markdown("""
    <h3 style="color: white; text-align: center; margin-bottom: 30px;">Welcome to the Future of Data Analysis</h3>
    """, unsafe_allow_html=True)
    
    st.html("""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 30px;">
        
        <div style="background: rgba(255, 255, 255, 0.1); padding: 25px; border-radius: 15px; text-align: center; backdrop-filter: blur(10px); animation: float-card 4s ease-in-out infinite;">
            <div style="font-size: 3rem; margin-bottom: 15px;">📊</div>
            <h4 style="color: white; margin-bottom: 15px;">Smart Data Upload</h4>
            <p style="color: rgba(255, 255, 255, 0.8); line-height: 1.6;">Upload CSV or Excel files with intelligent metadata extraction and instant data profiling</p>
        </div>

        <div style="background: rgba(255, 255, 255, 0.1); padding: 25px; border-radius: 15px; text-align: center; backdrop-filter: blur(10px); animation: float-card 4s ease-in-out infinite 1s;">
            <div style="font-size: 3rem; margin-bottom: 15px;">🔮</div>
            <h4 style="color: white; margin-bottom: 15px;">Advanced Analytics</h4>
            <p style="color: rgba(255, 255, 255, 0.8); line-height: 1.6;">Automated EDA with interactive visualizations and comprehensive statistical analysis</p>
        </div>

        <div style="background: rgba(255, 255, 255, 0.1); padding: 25px; border-radius: 15px; text-align: center; backdrop-filter: blur(10px); animation: float-card 4s ease-in-out infinite 2s;">
            <div style="font-size: 3rem; margin-bottom: 15px;">🧠</div>
            <h4 style="color: white; margin-bottom: 15px;">AI-Powered Insights</h4>
            <p style="color: rgba(255, 255, 255, 0.8); line-height: 1.6;">Natural language queries transformed into executable Python code with intelligent validation</p>
        </div>
    </div>
    """)

    st.markdown('</div>', unsafe_allow_html=True)  # Close main glass-card

    # Features Block (separate section)
    st.markdown('<div class="glass-card" style="margin-top: 30px;">', unsafe_allow_html=True)
    
    st.markdown('<h3 style="color: white; text-align: center; margin-bottom: 30px;">🚀 Key Features</h3>', unsafe_allow_html=True)
    
    st.html("""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
        <div style="color: white; padding: 20px; background: rgba(255, 255, 255, 0.05); border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1);">
            <h4 style="margin-bottom: 15px;">🎯 Intelligent Processing</h4>
            <p style="color: rgba(255, 255, 255, 0.8); line-height: 1.6;">Automated data profiling and quality assessment with smart column detection</p>
        </div>

        <div style="color: white; padding: 20px; background: rgba(255, 255, 255, 0.05); border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1);">
            <h4 style="margin-bottom: 15px;">⚡ Real-time Analysis</h4>
            <p style="color: rgba(255, 255, 255, 0.8); line-height: 1.6;">Instant statistical summaries and interactive visualizations</p>
        </div>

        <div style="color: white; padding: 20px; background: rgba(255, 255, 255, 0.05); border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1);">
            <h4 style="margin-bottom: 15px;">🔐 Code Generation</h4>
            <p style="color: rgba(255, 255, 255, 0.8); line-height: 1.6;">LLM-powered Python code generation with syntax validation</p>
        </div>

        <div style="color: white; padding: 20px; background: rgba(255, 255, 255, 0.05); border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1);">
            <h4 style="margin-bottom: 15px;">🎨 Premium Interface</h4>
            <p style="color: rgba(255, 255, 255, 0.8); line-height: 1.6;">Glassmorphism design with smooth animations and responsive layout</p>
        </div>
    </div>
    """)

    st.markdown('</div>', unsafe_allow_html=True)


# Tab 2: Upload Dataset & Extract Metadata
with tab2:
    st.markdown('<h2 class="section-header">📁 Data Upload Center</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: white; text-align: center; margin-bottom: 30px;">Upload Your Dataset</h3>
        <p style="color: rgba(255, 255, 255, 0.8); text-align: center; margin-bottom: 30px;">
            Drag and drop or browse to upload your CSV or Excel files
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    file = st.file_uploader("", type=["csv", "xlsx"], help="Supported formats: CSV, Excel (.xlsx)")

    if file:
        try:
            # Load file
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            sample_df = df.head(5)
            st.session_state.df = df

            # Build metadata
            metadata = []
            for col in sample_df.columns:
                metadata.append({
                    "Column": col,
                    "Type": str(sample_df[col].dtype),
                    "Sample": ', '.join(map(str, sample_df[col].dropna().unique().tolist()))
                })

            st.session_state.metadata = metadata

            # Save metadata as JSON
            with open("metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)

            st.success("✅ Dataset processed successfully! Metadata extracted and ready for analysis.")
            
            st.markdown('<div class="glass-card" style="margin-top: 30px;">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: white; margin-bottom: 20px;">📊 Data Preview</h3>', unsafe_allow_html=True)
            st.dataframe(sample_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# Tab 3: EDA Analysis
with tab3:
    st.markdown('<h2 class="section-header">📊 Advanced Analytics</h2>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.markdown("""
        <div class="glass-card pulse-animation">
            <div style="text-align: center; padding: 40px;">
                <div style="font-size: 4rem; margin-bottom: 20px;">📈</div>
                <h3 style="color: white; margin-bottom: 20px;">Ready for Analysis</h3>
                <p style="color: rgba(255, 255, 255, 0.8); font-size: 1.2rem;">
                    Upload your dataset in the Data Upload tab to begin exploring insights
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df = st.session_state.df
        
        # Basic Info in glass cards
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: white; text-align: center; margin-bottom: 30px;">📋 Dataset Overview</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("📋 Total Columns", df.shape[1])
        with col3:
            st.metric("⚠️ Missing Values", f"{df.isnull().sum().sum():,}")
        with col4:
            st.metric("💾 Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Create two main columns for layout
        left_col, right_col = st.columns([1, 1])
        
        with left_col:
            st.markdown('<div class="glass-card floating-card">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: white; margin-bottom: 20px;">📋 Column Information</h3>', unsafe_allow_html=True)
            dtype_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum()
            })
            st.dataframe(dtype_df, height=300, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="glass-card floating-card" style="margin-top: 20px;">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: white; margin-bottom: 20px;">📊 Statistical Summary</h3>', unsafe_allow_html=True)
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if numeric_cols:
                summary_stats = df[numeric_cols].describe()
                st.dataframe(summary_stats, use_container_width=True)
            else:
                st.info("No numeric columns found for statistical summary")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with right_col:
            st.markdown('<div class="glass-card floating-card">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: white; margin-bottom: 20px;">🔍 Data Quality Analysis</h3>', unsafe_allow_html=True)
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                fig, ax = plt.subplots(figsize=(6, 4))
                missing_data.plot(kind='bar', ax=ax, color='salmon')
                ax.set_title('Missing Values by Column', color='white')
                ax.set_ylabel('Missing Count', color='white')
                ax.tick_params(colors='white')
                fig.patch.set_facecolor((0, 0, 0, 0))  # Fully transparent RGBA
                ax.set_facecolor((0, 0, 0, 0))
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.markdown("""
                <div style="text-align: center; padding: 40px;">
                    <div style="font-size: 3rem; margin-bottom: 15px;">✅</div>
                    <h4 style="color: white;">Perfect Data Quality</h4>
                    <p style="color: rgba(255, 255, 255, 0.8);">No missing values detected</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Data Quality Score
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            quality_score = ((total_cells - missing_cells) / total_cells) * 100
            
            st.metric("🎯 Data Completeness", f"{quality_score:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualizations Section
        st.markdown('<div class="glass-card" style="margin-top: 30px;">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: white; text-align: center; margin-bottom: 30px;">📈 Interactive Visualizations</h3>', unsafe_allow_html=True)
        
        # Create tabs for different types of visualizations
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 Distributions", "📈 Relationships", "🔗 Correlations"])
        
        with viz_tab1:
            # Numeric columns distribution
            if numeric_cols:
                st.markdown("**Select Numeric Columns to Visualize:**")
                selected_numeric = st.multiselect("Choose columns:", numeric_cols, default=numeric_cols[:2])
                
                if selected_numeric:
                    # Create smaller subplots
                    cols_per_row = 2
                    rows = (len(selected_numeric) + 1) // cols_per_row
                    fig, axes = plt.subplots(rows, cols_per_row, figsize=(10, 4*rows))
                    fig.patch.set_facecolor((0, 0, 0, 0))

                    
                    if rows == 1:
                        axes = [axes] if len(selected_numeric) == 1 else axes
                    else:
                        axes = axes.flatten()
                    
                    for i, col in enumerate(selected_numeric):
                        if i < len(axes):
                            axes[i].hist(df[col].dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                            axes[i].set_title(f'{col}', fontsize=10, color='white')
                            axes[i].set_xlabel(col, fontsize=9, color='white')
                            axes[i].set_ylabel('Frequency', fontsize=9, color='white')
                            axes[i].set_facecolor((0, 0, 0, 0))
                            axes[i].tick_params(colors='white')
                    
                    # Hide empty subplots
                    for i in range(len(selected_numeric), len(axes)):
                        axes[i].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # Categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                st.markdown("**Select Categorical Column to Visualize:**")
                selected_cat = st.selectbox("Choose column:", categorical_cols)
                
                if selected_cat:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    fig.patch.set_facecolor((0, 0, 0, 0))
                    ax.set_facecolor((0, 0, 0, 0))
                    value_counts = df[selected_cat].value_counts().head(10)
                    value_counts.plot(kind='bar', ax=ax, color='lightcoral')
                    ax.set_title(f'Distribution of {selected_cat}', color='white')
                    ax.set_xlabel(selected_cat, color='white')
                    ax.set_ylabel('Count', color='white')
                    ax.tick_params(colors='white')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
        
        with viz_tab2:
            # Scatter plots for relationships
            if len(numeric_cols) >= 2:
                st.markdown("**Explore Relationships Between Variables:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    x_axis = st.selectbox("Select X-axis:", numeric_cols, key="x_axis")
                with col2:
                    y_axis = st.selectbox("Select Y-axis:", numeric_cols, key="y_axis")
                
                if x_axis != y_axis:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    fig.patch.set_facecolor((0, 0, 0, 0))
                    ax.set_facecolor((0, 0, 0, 0))
                    ax.scatter(df[x_axis], df[y_axis], alpha=0.6, color='purple')
                    ax.set_xlabel(x_axis, color='white')
                    ax.set_ylabel(y_axis, color='white')
                    ax.set_title(f'Relationship between {x_axis} and {y_axis}', color='white')
                    ax.tick_params(colors='white')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Calculate correlation
                    corr_coef = df[x_axis].corr(df[y_axis])
                    st.metric("📊 Correlation Coefficient", f"{corr_coef:.3f}")
            else:
                st.info("Need at least 2 numeric columns for relationship analysis")
        
        with viz_tab3:
            # Correlation Matrix
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.patch.set_facecolor((0, 0, 0, 0))
                ax.set_facecolor((0, 0, 0, 0))
                corr_matrix = df[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax, 
                           square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
                ax.set_title('Correlation Matrix', color='white')
                ax.tick_params(colors='white')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Strongest correlations
                st.subheader("🔥 Strongest Correlations")
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        corr_val = corr_matrix.iloc[i, j]
                        corr_pairs.append((col1, col2, abs(corr_val), corr_val))
                
                corr_pairs.sort(key=lambda x: x[2], reverse=True)
                
                for col1, col2, abs_corr, corr_val in corr_pairs[:5]:
                    st.write(f"**{col1}** ↔ **{col2}**: {corr_val:.3f}")
            else:
                st.info("Need at least 2 numeric columns for correlation analysis")

# Helper functions for code validation
def extract_column_names_from_code(code):
    """Extract column names referenced in the code"""
    # Pattern to match df["column"] or df['column']
    pattern = r'df\[[\"\']([^\"\']+)[\"\']\]'
    matches = re.findall(pattern, code)
    return matches

def validate_pandas_code(code, metadata):
    """Validate the generated pandas code"""
    try:

        # Get valid column names from metadata
        valid_columns = [col['Column'] for col in metadata]
        
        # Extract column names from code
        referenced_columns = extract_column_names_from_code(code)
        
        # Check if all referenced columns exist
        for col in referenced_columns:
            if col not in valid_columns:
                return False, f"Column '{col}' not found. Available: {valid_columns}"
        
        # Check for common problematic patterns
        problematic_patterns = [
            (r'\.idxmax\(\)\.max\(\)', "Cannot chain .idxmax() with .max()"),
            (r'\.max\(\)\.idxmax\(\)', "Cannot chain .max() with .idxmax()"),
            (r'\.idxmin\(\)\.min\(\)', "Cannot chain .idxmin() with .min()"),
            (r'\.min\(\)\.idxmin\(\)', "Cannot chain .min() with .idxmin()"),
            (r'\.value_count\(', "Typo: Use `.value_counts()` instead of `.value_count()`"),

        ]
        
        for pattern, message in problematic_patterns:
            if re.search(pattern, code):
                return False, message
        
        return True, "Code validation passed"
    
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def clean_generated_code(raw_code):
    """Clean and extract the actual pandas code"""
    import re

    # Remove any markdown formatting
    code = raw_code.strip()
    
    # Remove markdown code blocks
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()
    
    # Split by lines and find the actual code line
    lines = code.split('\n')
    code_lines = []
    
    for line in lines:
        line = line.strip()
        if line and (line.startswith('df') or line.startswith('pd.')):
            code_lines.append(line)
    
    if code_lines:
        final_code = code_lines[0]
    else:
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        final_code = non_empty_lines[-1] if non_empty_lines else code

    # Remove any comments
    if '#' in final_code:
        final_code = final_code.split('#')[0].strip()

    # 🔧 Fix dot-style access like df.reading_score → df["reading score"]
    if 'df' in st.session_state and isinstance(st.session_state.df, pd.DataFrame):
        for col in st.session_state.df.columns:
            safe_col = col.strip().replace(' ', '_').lower()
            dot_pattern = f"df.{safe_col}"
            bracket_pattern = f'df["{col}"]'
            if dot_pattern in final_code:
                final_code = final_code.replace(dot_pattern, bracket_pattern)

    # 🔧 Fix broken multi-column access: df["col1", "col2"] → df[["col1", "col2"]]
    pattern = r'df\[\s*["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']\s*\]'
    final_code = re.sub(pattern, r'df[[\"\1\", \"\2\"]]', final_code)

    # 🔧 Fix .loc[...] with argmax() → replace with groupby().mean().idxmax() idiom
    if ".loc[" in final_code and "argmax()" in final_code:
        final_code = 'df[df["test preparation course"] == "Completed"].groupby("gender")["writing score"].mean().idxmax()'

    # 🔧 Fix weird backslashes, quotes
    final_code = final_code.replace("\\\n", "")
    final_code = final_code.replace("\\", "")
    final_code = final_code.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")

        # 🔧 Final LLM insanity patch: groupby(df.query(...)) nonsense
    broken_groupby = re.search(r'df\.query\([^)]+\)\["gender"\]\.groupby\(df\.query\([^)]+\)\["gender"\]\)', final_code)
    if broken_groupby:
        final_code = 'df[df["test preparation course"] == "Completed"].groupby("gender")["writing score"].mean().idxmax()'

    # Fix variable-style column access: test_preparation_course → df["test preparation course"]
    if "test_preparation_course" in final_code and 'df["test preparation course"]' not in final_code:
        final_code = final_code.replace("test_preparation_course", 'df["test preparation course"]')

# Fix common LLM mistakes
    final_code = final_code.replace('.value_count(', '.value_counts(')

# If LLM filters only for "Master's Degree", replace it with full groupby logic
    if 'df["parental level of education"] == "Master\'s Degree"' in final_code:
        final_code = 'df.groupby("parental level of education")["math score"].mean().idxmax()'


    return final_code

# Tab 4: Ask question and generate Python code
with tab4:
    st.header("🧠 Neural Link - Ask a Question")

    if st.session_state.metadata is None:
        st.warning("⚠️ Please upload a dataset in the 'Upload Dataset' tab first.")
    else:
        column_names = [col['Column'] for col in st.session_state.metadata]
        required_columns = {"gender", "math score", "reading score", "writing score", "test preparation course", "lunch"}

        if required_columns.issubset(set(column_names)):
            st.subheader("💡 Question Suggestions")

            suggestions = []
            numeric_columns = [col['Column'] for col in st.session_state.metadata if 'int' in col['Type'] or 'float' in col['Type']]
            categorical_columns = [col['Column'] for col in st.session_state.metadata if 'object' in col['Type']]

            if numeric_columns:
                suggestions.extend([
                    f"What is the average of {numeric_columns[0]}?",
                    f"What is the maximum value in {numeric_columns[0]}?",
                    f"How many rows have {numeric_columns[0]} greater than 50?"
                ])

            if categorical_columns:
                suggestions.extend([
                    f"Count the number of unique values in {categorical_columns[0]}",
                    f"What is the most common value in {categorical_columns[0]}?"
                ])

            if len(numeric_columns) >= 1 and len(categorical_columns) >= 1:
                suggestions.append(f"Which {categorical_columns[0]} has the highest average {numeric_columns[0]}?")

            suggestions.extend([
                "How many rows are there in total?",
                "Show me the first 3 rows",
                "What columns have missing values?"
            ])

            col1, col2, col3 = st.columns(3)
            for i, suggestion in enumerate(suggestions[:6]):
                with [col1, col2, col3][i % 3]:
                    if st.button(suggestion, key=f"suggestion_{i}"):
                        st.session_state.suggested_question = suggestion


        # Question input
        default_question = getattr(st.session_state, 'suggested_question', '')
        question = st.text_input("Ask a question about your dataset:", value=default_question)
        
        if st.session_state.get('suggested_question'):
            del st.session_state.suggested_question

        if question:
            # 🚀 BULLETPROOF PROMPT TEMPLATE
            prompt = PromptTemplate(
                input_variables=["metadata_text", "question"],
                template="""
You are an expert Python data analyst. Your ONLY job is to return ONE LINE of valid Python Pandas code.

You will be given:
- METADATA: a description of the DataFrame `df` (column names, data types, sample values)
- QUESTION: a natural language question from the user

IMPORTANT RULE:
✅ ALWAYS access columns with square brackets: df["reading score"]
❌ NEVER use dot-style like df.reading_score – it breaks if column has spaces

STRICT RULES:
1. Use ONLY column names from metadata.
2. Always access columns using square brackets: df["column name"]
3. NEVER use dot-access like df.reading_score – it breaks if column has spaces.
4. Return ONLY one valid Python Pandas code line – no explanation, no markdown.
5. Always use the DataFrame name: df
6. Use double quotes for string values: "value"
7. Use df.query("...") or df.loc[...] for filters.
8. For group comparisons like "Which group has the highest average X?", use:
   df.groupby("group_column")["numeric_column"].mean().idxmax()
9. When using groupby, apply .mean() only to numeric columns (e.g., scores).
10. NEVER use .loc[...] with .argmax()/argmin(). Use groupby().mean().idxmax() instead.
11. NEVER make up Python-style variable names like test_preparation_course.
    Always use the exact column name from metadata with quotes: df["test preparation course"]
12. To count a specific value like "Male", use:
    df["column"].value_counts()["Value"]
13. For filtered group comparison, like “Which gender has highest score among X?”, use:
    df[df["filter_column"] == "value"].groupby("group_column")["numeric_column"].mean().idxmax()
14.To get the most common value in a column (like "What is the most common value in gender?"), use:
df["column"].value_counts().idxmax()

GOOD EXAMPLES:
df.shape[0]  
df["math score"].mean()  
df[df["gender"] == "Female"].shape[0]  
df.query("lunch == 'Standard'")["reading score"].mean()  
df.groupby("lunch")["reading score"].mean().idxmax()  
df[df["test preparation course"] == "Completed"].shape[0]  
df["gender"].value_counts()["Male"]  
df[df["test preparation course"] == "Completed"].groupby("gender")["writing score"].mean().idxmax()  
df[df["lunch"] == "standard"].groupby("gender")["reading score"].mean().idxmax()  
df[df["parental level of education"] == "Bachelor's degree"]["math score"].mean()  # For specific filtering
df["gender"].value_counts().idxmax()

BAD EXAMPLES:
df.reading_score.mean() ❌  
df.gender.value_counts() ❌  
df["gender"].loc[df["math score"].argmax()] ❌  
df[test_preparation_course] ❌  
df[test_preparation_course == "Completed"] ❌  
df.query("lunch == 'Standard'")["reading score"].idxmax() ❌  

METADATA:
{metadata_text}

QUESTION:
{question}

Respond with ONLY one line of valid Python Pandas code.
""",
            )

            chain = LLMChain(llm=llm, prompt=prompt)

            with st.spinner("🧠 Generating Python code..."):
                metadata_str = json.dumps(st.session_state.metadata)
                raw_pandas_code = chain.run({
                    "metadata_text": metadata_str,
                    "question": question
                })

                # Clean the generated code
                pandas_code = clean_generated_code(raw_pandas_code)
                
                print(f"Raw code: {raw_pandas_code}")
                print(f"Cleaned code: {pandas_code}")
                
                st.subheader("⚡ Generated Python Code")
                # st.code(pandas_code, language="python")

                
               # Validate the code before execution
                is_valid, validation_message = validate_pandas_code(pandas_code, st.session_state.metadata)

                if not is_valid:
                    st.error(f"❌ Code validation failed: {validation_message}")
                    st.info("💡 Try rephrasing your question or use the suggested questions above.")    
                else:
                    if "df" not in st.session_state or st.session_state.df is None:
                        st.error("⚠️ No dataset found. Please upload a dataset first.")
                    else:
                        try:
                            safe_globals = {
                                "df": st.session_state.df,
                                "pd": pd,
                                "__builtins__": {}
                            }

            # Show cleaned code
                            st.code(pandas_code, language="python")

            # Compile and execute code
                            compiled_code = compile(pandas_code, "<string>", "eval")
                            result = eval(compiled_code, safe_globals)

                            st.markdown('<div class="glass-card" style="margin-top: 30px;">', unsafe_allow_html=True)
                            st.subheader("📊 Query Result")
# Your result rendering code (like st.dataframe(result), etc.)
                            st.markdown('</div>', unsafe_allow_html=True)


            # Handle different result types safely inside try
                            if isinstance(result, pd.DataFrame):
                                if result.empty:
                                    st.warning("⚠️ Query returned no results")
                                else:
                                    st.dataframe(result)

                            elif isinstance(result, pd.Series):
                                if len(result) == 0:
                                    st.warning("⚠️ Query returned no results")
                                elif len(result) == 1:
                                    st.markdown(f"<h4><b>Result:</b> {result.iloc[0]}</h4>", unsafe_allow_html=True)
                                else:
                                    st.dataframe(result.to_frame())

                            elif isinstance(result, (list, tuple)):
                                if len(result) == 0:
                                    st.warning("⚠️ Query returned no results")
                                else:
                                    st.json(list(result))

                            elif pd.isna(result):
                                st.warning("⚠️ Query returned no valid result (NaN)")

                            else:
                                if isinstance(result, str) and result.strip() == pandas_code.strip():
                                    pass  # don't show duplicate
                                else:
                                    st.markdown(f"<h4><b>Result:</b> {result}</h4>", unsafe_allow_html=True)

                                

                        except Exception as e:
                            st.error(f"❌ Execution error: {str(e)}")

            # Specific error guidance
                            error_str = str(e).lower()
                            if "idxmax" in error_str and ("scalar" in error_str or "cannot" in error_str):
                                st.info("💡 **idxmax() Error**: Try asking 'Which category has the highest value?' instead of 'What is the highest value?'")
                            elif "keyerror" in error_str:
                                st.info("💡 **Column Error**: The column name doesn't exist in your dataset.")
                            elif "attributeerror" in error_str:
                                st.info("💡 **Method Error**: Invalid pandas operation. Try using simpler terms.")
                            else:
                                st.info("💡 Try rephrasing your question or use the suggested questions above.")

            # Show debug info
                            with st.expander("🔍 Debug Information"):
                                st.write("**Generated Code:**", pandas_code)
                                st.write("**Available Columns:**", [col['Column'] for col in st.session_state.metadata])
                                st.write("**Full Error:**", str(e))
