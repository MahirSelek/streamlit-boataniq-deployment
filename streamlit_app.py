"""
BoataniQ Streamlit App
AI-powered boat recognition and analysis application
"""

import streamlit as st
import os
import json
from PIL import Image
import io
from boat_vertex_ai_analyzer import BoatVertexAIAnalyzer
from image_preprocessor import ImagePreprocessor

# Page configuration
st.set_page_config(
    page_title="BoataniQ - AI Boat Analyzer",
    page_icon="⛵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = ImagePreprocessor()
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

def initialize_analyzer():
    """Initialize the Vertex AI analyzer with credentials from Streamlit secrets"""
    try:
        # Try to get credentials from Streamlit secrets
        if 'gcp_credentials' in st.secrets:
            credentials_json = json.dumps(st.secrets['gcp_service_account'])
            analyzer = BoatVertexAIAnalyzer(credentials_json=credentials_json)
            return analyzer, None
        else:
            return None, "GCP credentials not found in Streamlit secrets. Please configure secrets.toml"
    except Exception as e:
        return None, f"Error initializing analyzer: {str(e)}"

# Initialize analyzer
if st.session_state.analyzer is None:
    with st.spinner("Initializing AI analyzer..."):
        analyzer, error = initialize_analyzer()
        if analyzer:
            st.session_state.analyzer = analyzer
        else:
            st.error(error)
            st.stop()

# Main title
st.title("⛵ BoataniQ - AI Boat Analyzer")
st.markdown("Upload a boat image to get detailed AI-powered analysis and identification")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    **BoataniQ** uses Google Cloud Vertex AI (Gemini Flash 2.0) to analyze boat images and provide:
    - Boat type identification
    - Brand and model recognition
    - Technical specifications
    - Market analysis
    - Detailed descriptions
    """)
    
    st.header("Instructions")
    st.markdown("""
    1. Upload a clear boat image
    2. Wait for AI analysis
    3. View detailed results
    """)
    
    if st.session_state.analyzer:
        model_info = st.session_state.analyzer.get_model_info()
        st.header("AI Model")
        st.info(f"""
        **Model:** {model_info['model_name']}
        **Provider:** {model_info['provider']}
        """)

# Main content area
uploaded_file = st.file_uploader(
    "Upload a boat image",
    type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
    help="Upload a clear image of a boat for analysis"
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Analyze button
    if st.button("🔍 Analyze Boat", type="primary", use_container_width=True):
        with st.spinner("Analyzing image..."):
            try:
                # Read image bytes
                image_bytes = uploaded_file.read()
                
                # Validate image
                with st.expander("🔍 Image Validation", expanded=False):
                    validation_result = st.session_state.preprocessor.validate_boat_image(image_bytes)
                    
                    if not validation_result['can_proceed']:
                        st.error(f"❌ Image validation failed: {validation_result.get('rejection_reason', 'Unknown error')}")
                        st.warning("Please upload a clear, well-lit boat image from a good angle.")
                        st.stop()
                    else:
                        st.success("✅ Image validation passed")
                        quality_score = validation_result.get('quality_validation', {}).get('quality_score', 0)
                        boat_confidence = validation_result.get('boat_detection', {}).get('confidence', 0)
                        st.metric("Quality Score", f"{quality_score:.2f}")
                        st.metric("Boat Detection Confidence", f"{boat_confidence:.2f}")
                
                # Preprocess image
                processed_bytes, preprocessing_info = st.session_state.preprocessor.preprocess_image(
                    image_bytes, enhance_quality=True
                )
                
                # Analyze with AI
                analysis_result = st.session_state.analyzer.analyze_boat_image_from_bytes(processed_bytes)
                
                # Check for errors
                if 'error' in analysis_result:
                    st.error(f"❌ Analysis Error: {analysis_result['error']}")
                    st.stop()
                
                # Check if image was rejected by AI
                if not analysis_result.get('is_valid_image', True) or analysis_result.get('rejection_reason'):
                    st.error(f"❌ Image Rejected: {analysis_result.get('rejection_reason', 'Unknown reason')}")
                    st.warning("Please upload a clear boat image from a good angle where the boat is clearly visible.")
                    st.stop()
                
                # Check confidence
                confidence = analysis_result.get('confidence', 0)
                if isinstance(confidence, str):
                    try:
                        confidence = float(confidence)
                    except:
                        confidence = 0
                
                if confidence < 30:
                    st.warning(f"⚠️ Low confidence ({confidence}%). Results may be less accurate.")
                
                # Display results
                st.success("✅ Analysis Complete!")
                
                # Main results in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Boat Type", analysis_result.get('boat_type', 'Unknown'))
                    st.metric("Brand", analysis_result.get('brand', 'Unknown'))
                
                with col2:
                    st.metric("Model", analysis_result.get('model', 'Unknown'))
                    st.metric("Estimated Year", analysis_result.get('estimated_year', 'Unknown'))
                
                with col3:
                    st.metric("Confidence", f"{confidence}%")
                    st.metric("Length", f"{analysis_result.get('length_estimate', 'Unknown')}m")
                
                # Detailed information in tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "🔧 Technical Specs", "🎨 Design Analysis", "📊 Market Info"])
                
                with tab1:
                    st.subheader("Overview")
                    st.markdown(f"**Boat Type:** {analysis_result.get('boat_type', 'Unknown')}")
                    st.markdown(f"**Brand:** {analysis_result.get('brand', 'Unknown')}")
                    st.markdown(f"**Model:** {analysis_result.get('model', 'Unknown')}")
                    st.markdown(f"**Model Line:** {analysis_result.get('model_line', 'Unknown')}")
                    st.markdown(f"**Estimated Year:** {analysis_result.get('estimated_year', 'Unknown')}")
                    st.markdown(f"**Length:** {analysis_result.get('length_estimate', 'Unknown')}m")
                    st.markdown(f"**Width:** {analysis_result.get('width_estimate', 'Unknown')}m")
                    st.markdown(f"**Hull Material:** {analysis_result.get('hull_material', 'Unknown')}")
                    st.markdown(f"**Engine Type:** {analysis_result.get('engine_type', 'Unknown')}")
                    st.markdown(f"**Hull Type:** {analysis_result.get('hull_type', 'Unknown')}")
                    st.markdown(f"**Condition:** {analysis_result.get('condition', 'Unknown')}")
                    st.markdown(f"**Price Estimate:** {analysis_result.get('price_estimate', 'Unknown')}")
                    
                    if analysis_result.get('key_features'):
                        st.subheader("Key Features")
                        features = analysis_result['key_features']
                        if isinstance(features, list):
                            for feature in features:
                                st.markdown(f"- {feature}")
                    
                    if analysis_result.get('detailed_description'):
                        st.subheader("Detailed Description")
                        st.markdown(analysis_result['detailed_description'])
                
                with tab2:
                    st.subheader("Technical Specifications")
                    tech_specs = analysis_result.get('technical_specs', {})
                    if tech_specs:
                        for spec, value in tech_specs.items():
                            if value and value != 'Unknown':
                                st.markdown(f"**{spec.replace('_', ' ').title()}:** {value}")
                    else:
                        st.info("Technical specifications not available for this analysis.")
                
                with tab3:
                    st.subheader("Design Analysis")
                    design_analysis = analysis_result.get('design_analysis', {})
                    if design_analysis:
                        st.markdown(f"**Hull Design:** {design_analysis.get('hull_design', 'N/A')}")
                        st.markdown(f"**Cabin Layout:** {design_analysis.get('cabin_layout', 'N/A')}")
                        st.markdown(f"**Deck Features:** {design_analysis.get('deck_features', 'N/A')}")
                        st.markdown(f"**Aerodynamics:** {design_analysis.get('aerodynamics', 'N/A')}")
                    else:
                        st.info("Design analysis not available for this analysis.")
                
                with tab4:
                    st.subheader("Market Positioning")
                    market = analysis_result.get('market_positioning', {})
                    if market:
                        st.markdown(f"**Target Market:** {market.get('target_market', 'N/A')}")
                        st.markdown(f"**Competitors:** {market.get('competitors', 'N/A')}")
                        st.markdown(f"**Unique Selling Points:** {market.get('unique_selling_points', 'N/A')}")
                        st.markdown(f"**Ideal Use Cases:** {market.get('ideal_use_cases', 'N/A')}")
                    
                    st.subheader("Historical Context")
                    historical = analysis_result.get('historical_context', {})
                    if historical:
                        st.markdown(f"**Design Era:** {historical.get('design_era', 'N/A')}")
                        st.markdown(f"**Manufacturer History:** {historical.get('manufacturer_history', 'N/A')}")
                        st.markdown(f"**Model Evolution:** {historical.get('model_evolution', 'N/A')}")
                        st.markdown(f"**Market Reception:** {historical.get('market_reception', 'N/A')}")
                
                # Add to history
                st.session_state.analysis_history.insert(0, {
                    'filename': uploaded_file.name,
                    'analysis': analysis_result
                })
                
                # Limit history to last 10
                if len(st.session_state.analysis_history) > 10:
                    st.session_state.analysis_history = st.session_state.analysis_history[:10]
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.exception(e)

# Analysis history
if st.session_state.analysis_history:
    with st.expander("📜 Analysis History", expanded=False):
        for i, entry in enumerate(st.session_state.analysis_history[:5]):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{entry['filename']}** - {entry['analysis'].get('brand', 'Unknown')} {entry['analysis'].get('model', 'Unknown')}")
                with col2:
                    st.markdown(f"Confidence: {entry['analysis'].get('confidence', 0)}%")
                st.divider()

# Footer
st.markdown("---")
st.markdown("**BoataniQ** - Powered by Google Cloud Vertex AI (Gemini Flash 2.0)")
