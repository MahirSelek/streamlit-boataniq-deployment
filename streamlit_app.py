"""
BoataniQ Streamlit App - Complete Version
AI-powered boat recognition and analysis application with ALL features
"""

import streamlit as st
import os
import json
import pandas as pd
import uuid
import datetime
from PIL import Image
import io
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional

# Import all analyzers
from boat_vertex_ai_analyzer import BoatVertexAIAnalyzer
from image_preprocessor import ImagePreprocessor
from boat_database import BoatDatabase
from boat_location_analyzer import BoatLocationAnalyzer
from boat_market_analyzer import BoatMarketAnalyzer
from financial_indices_fetcher import FinancialIndicesFetcher

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
if 'boat_db' not in st.session_state:
    st.session_state.boat_db = None
if 'location_analyzer' not in st.session_state:
    st.session_state.location_analyzer = None
if 'boat_market_analyzer' not in st.session_state:
    st.session_state.boat_market_analyzer = None
if 'financial_fetcher' not in st.session_state:
    st.session_state.financial_fetcher = FinancialIndicesFetcher()
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

def initialize_analyzer():
    """Initialize the Vertex AI analyzer with credentials from Streamlit secrets"""
    try:
        credentials_dict = None
        if 'gcp_service_account' in st.secrets:
            credentials_dict = dict(st.secrets['gcp_service_account'])
        elif 'gcp_credentials' in st.secrets:
            credentials_dict = dict(st.secrets['gcp_credentials'])
        
        if credentials_dict:
            credentials_json = json.dumps(credentials_dict)
            analyzer = BoatVertexAIAnalyzer(credentials_json=credentials_json)
            return analyzer, None
        else:
            return None, "GCP credentials not found in Streamlit secrets. Please configure [gcp_service_account] in secrets.toml"
    except Exception as e:
        return None, f"Error initializing analyzer: {str(e)}"

def initialize_database():
    """Initialize boat database"""
    try:
        # Try to load database files (if available)
        csv_path = 'all_boats_data.csv'
        json_dir = 'json_boat24'
        
        if os.path.exists(csv_path):
            boat_db = BoatDatabase(csv_path, json_dir if os.path.exists(json_dir) else None)
            return boat_db, None
        else:
            return None, "Database files not found. Some features will be limited."
    except Exception as e:
        return None, f"Error initializing database: {str(e)}"

def clean_boat_data_for_display(boat_data):
    """Clean boat data for display"""
    if not boat_data:
        return None
    cleaned = {}
    for key, value in boat_data.items():
        if pd.isna(value) or value is None or str(value).lower() == 'nan':
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned

# Initialize all components
if st.session_state.analyzer is None:
    with st.spinner("Initializing AI analyzer..."):
        analyzer, error = initialize_analyzer()
        if analyzer:
            st.session_state.analyzer = analyzer
        else:
            st.error(error)
            st.stop()

if st.session_state.boat_db is None:
    with st.spinner("Initializing database..."):
        boat_db, db_error = initialize_database()
        if boat_db:
            st.session_state.boat_db = boat_db
            # Initialize market analyzer if database is available
            try:
                st.session_state.boat_market_analyzer = BoatMarketAnalyzer(boat_db.boats_df)
            except:
                pass
        else:
            if db_error:
                st.warning(db_error)

if st.session_state.location_analyzer is None:
    try:
        st.session_state.location_analyzer = BoatLocationAnalyzer()
    except:
        pass

# Sidebar Navigation
st.sidebar.title("⛵ BoataniQ")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home - Image Analysis", "🔍 Search Boats", "📊 Data Insights", "💰 Investment Comparison", "🗺️ Map", "📜 History"]
)

# Main content based on selected page
if page == "🏠 Home - Image Analysis":
    st.title("⛵ BoataniQ - AI Boat Analyzer")
    st.markdown("Upload a boat image to get detailed AI-powered analysis and identification")
    
    uploaded_file = st.file_uploader(
        "Upload a boat image",
        type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
        help="Upload a clear image of a boat for analysis"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            analyze_btn = st.button("🔍 Analyze Boat", type="primary", use_container_width=True)
        with col2:
            analyze_location_btn = st.button("🗺️ Analyze Location", use_container_width=True)
        
        if analyze_btn:
            with st.spinner("Analyzing image..."):
                try:
                    image_bytes = uploaded_file.read()
                    
                    # Validate image
                    validation_result = st.session_state.preprocessor.validate_boat_image(image_bytes)
                    
                    if not validation_result['can_proceed']:
                        st.error(f"❌ Image validation failed: {validation_result.get('rejection_reason', 'Unknown error')}")
                        st.warning("Please upload a clear, well-lit boat image from a good angle.")
                    else:
                        # Preprocess image
                        processed_bytes, preprocessing_info = st.session_state.preprocessor.preprocess_image(
                            image_bytes, enhance_quality=True
                        )
                        
                        # Analyze with AI
                        analysis_result = st.session_state.analyzer.analyze_boat_image_from_bytes(processed_bytes)
                        
                        if 'error' not in analysis_result and analysis_result.get('is_valid_image', True):
                            st.success("✅ Analysis Complete!")
                            
                            # Main metrics
                            col1, col2, col3, col4 = st.columns(4)
                            confidence = analysis_result.get('confidence', 0)
                            if isinstance(confidence, str):
                                try:
                                    confidence = float(confidence)
                                except:
                                    confidence = 0
                            
                            with col1:
                                st.metric("Boat Type", analysis_result.get('boat_type', 'Unknown'))
                            with col2:
                                st.metric("Brand", analysis_result.get('brand', 'Unknown'))
                            with col3:
                                st.metric("Model", analysis_result.get('model', 'Unknown'))
                            with col4:
                                st.metric("Confidence", f"{confidence}%")
                            
                            # Detailed tabs
                            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                                "📋 Overview", "🔧 Technical Specs", "🎨 Design", "📊 Market", "🔍 Similar Boats"
                            ])
                            
                            with tab1:
                                st.subheader("Overview")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"**Boat Type:** {analysis_result.get('boat_type', 'Unknown')}")
                                    st.markdown(f"**Brand:** {analysis_result.get('brand', 'Unknown')}")
                                    st.markdown(f"**Model:** {analysis_result.get('model', 'Unknown')}")
                                    st.markdown(f"**Model Line:** {analysis_result.get('model_line', 'Unknown')}")
                                    st.markdown(f"**Estimated Year:** {analysis_result.get('estimated_year', 'Unknown')}")
                                with col2:
                                    st.markdown(f"**Length:** {analysis_result.get('length_estimate', 'Unknown')}m")
                                    st.markdown(f"**Width:** {analysis_result.get('width_estimate', 'Unknown')}m")
                                    st.markdown(f"**Hull Material:** {analysis_result.get('hull_material', 'Unknown')}")
                                    st.markdown(f"**Engine Type:** {analysis_result.get('engine_type', 'Unknown')}")
                                    st.markdown(f"**Hull Type:** {analysis_result.get('hull_type', 'Unknown')}")
                                
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
                            
                            with tab3:
                                st.subheader("Design Analysis")
                                design = analysis_result.get('design_analysis', {})
                                if design:
                                    st.markdown(f"**Hull Design:** {design.get('hull_design', 'N/A')}")
                                    st.markdown(f"**Cabin Layout:** {design.get('cabin_layout', 'N/A')}")
                                    st.markdown(f"**Deck Features:** {design.get('deck_features', 'N/A')}")
                            
                            with tab4:
                                st.subheader("Market Positioning")
                                market = analysis_result.get('market_positioning', {})
                                if market:
                                    st.markdown(f"**Target Market:** {market.get('target_market', 'N/A')}")
                                    st.markdown(f"**Competitors:** {market.get('competitors', 'N/A')}")
                            
                            with tab5:
                                st.subheader("Similar Boats in Database")
                                if st.session_state.boat_db:
                                    similar_boats = st.session_state.boat_db.find_similar_boats(analysis_result, limit=10)
                                    if similar_boats:
                                        for boat in similar_boats:
                                            with st.expander(f"**{boat.get('title', 'Unknown')}** - {boat.get('price', 'N/A')}"):
                                                col1, col2 = st.columns(2)
                                                with col1:
                                                    st.markdown(f"**Brand:** {boat.get('brand', 'N/A')}")
                                                    st.markdown(f"**Model:** {boat.get('model', 'N/A')}")
                                                    st.markdown(f"**Year:** {boat.get('year_built', 'N/A')}")
                                                with col2:
                                                    st.markdown(f"**Dimensions:** {boat.get('dimensions', 'N/A')}")
                                                    st.markdown(f"**Price:** {boat.get('price', 'N/A')}")
                                    else:
                                        st.info("No similar boats found in database.")
                                else:
                                    st.info("Database not available. Similar boats feature requires database.")
                            
                            # Add to history
                            history_entry = {
                                'id': str(uuid.uuid4()),
                                'timestamp': datetime.datetime.now().isoformat(),
                                'filename': uploaded_file.name,
                                'analysis': analysis_result
                            }
                            st.session_state.analysis_history.insert(0, history_entry)
                            if len(st.session_state.analysis_history) > 50:
                                st.session_state.analysis_history = st.session_state.analysis_history[:50]
                        else:
                            st.error(f"❌ Analysis failed: {analysis_result.get('error', 'Unknown error')}")
                            
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.exception(e)
        
        elif analyze_location_btn:
            if st.session_state.location_analyzer:
                with st.spinner("Analyzing location..."):
                    try:
                        image_bytes = uploaded_file.read()
                        location_result = st.session_state.location_analyzer.analyze_image_location(image_bytes)
                        
                        if location_result.get('success'):
                            st.success("✅ Location Analysis Complete!")
                            st.subheader("Detected Locations")
                            
                            for loc in location_result.get('detected_locations', []):
                                st.markdown(f"**{loc['name']}**, {loc['country']}")
                                st.markdown(f"Coordinates: {loc['lat']}, {loc['lon']}")
                                
                                if st.session_state.boat_db:
                                    nearby = st.session_state.boat_db.search_by_location(
                                        loc['lat'], loc['lon'], radius_km=50, limit=10
                                    )
                                    if nearby:
                                        st.markdown(f"**Found {len(nearby)} boats nearby:**")
                                        for boat in nearby[:5]:
                                            st.markdown(f"- {boat.get('title', 'Unknown')} ({boat.get('distance_km', 0):.1f} km away)")
                                st.divider()
                        else:
                            st.error(f"Location analysis failed: {location_result.get('error')}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Location analyzer not available")

elif page == "🔍 Search Boats":
    st.title("🔍 Search Boats Database")
    
    if st.session_state.boat_db:
        search_type = st.radio("Search Type", ["Keywords", "Brand", "Model", "Year"], horizontal=True)
        
        if search_type == "Keywords":
            query = st.text_input("Enter search keywords")
            if st.button("Search", type="primary"):
                if query:
                    results = st.session_state.boat_db.search_by_keywords(query, limit=20)
                    st.success(f"Found {len(results)} boats")
                    for boat in results:
                        with st.expander(f"**{boat.get('title', 'Unknown')}** - {boat.get('price', 'N/A')}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Brand:** {boat.get('brand', 'N/A')}")
                                st.markdown(f"**Model:** {boat.get('model', 'N/A')}")
                                st.markdown(f"**Year:** {boat.get('year_built', 'N/A')}")
                            with col2:
                                st.markdown(f"**Dimensions:** {boat.get('dimensions', 'N/A')}")
                                st.markdown(f"**Price:** {boat.get('price', 'N/A')}")
        
        elif search_type == "Brand":
            query = st.text_input("Enter brand name")
            if st.button("Search", type="primary"):
                if query:
                    results = st.session_state.boat_db.search_by_brand(query, limit=20)
                    st.success(f"Found {len(results)} boats")
                    for boat in results:
                        with st.expander(f"**{boat.get('title', 'Unknown')}**"):
                            st.json(clean_boat_data_for_display(boat))
        
        elif search_type == "Model":
            query = st.text_input("Enter model name")
            if st.button("Search", type="primary"):
                if query:
                    results = st.session_state.boat_db.search_by_model(query, limit=20)
                    st.success(f"Found {len(results)} boats")
                    for boat in results:
                        st.markdown(f"**{boat.get('title', 'Unknown')}**")
        
        elif search_type == "Year":
            year = st.number_input("Enter year", min_value=1900, max_value=2030, value=2020)
            if st.button("Search", type="primary"):
                results = st.session_state.boat_db.search_by_year_range(year-5, year+5, limit=20)
                st.success(f"Found {len(results)} boats")
                for boat in results:
                    st.markdown(f"**{boat.get('title', 'Unknown')}** - {boat.get('year_built', 'N/A')}")
    else:
        st.warning("Database not available. Please ensure database files are in the deployment folder.")

elif page == "📊 Data Insights":
    st.title("📊 Data Insights Dashboard")
    
    if st.session_state.boat_db:
        df = st.session_state.boat_db.boats_df
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Boats", len(df))
        with col2:
            st.metric("Unique Brands", len(df['title'].str.split().str[0].unique()) if 'title' in df.columns else "N/A")
        with col3:
            st.metric("With Prices", len(df[df['price'].notna()]))
        with col4:
            st.metric("With Years", len(df[df['year_built'].notna()]))
        
        # Price distribution
        st.subheader("Price Distribution")
        try:
            def extract_price(price_str):
                if pd.isna(price_str) or 'Price on Request' in str(price_str):
                    return None
                try:
                    price_clean = str(price_str).replace('EUR', '').replace(',', '').replace('.', '').replace('-', '').replace(' ', '').strip()
                    import re
                    numbers = re.findall(r'\d+', price_clean)
                    if numbers:
                        return int(''.join(numbers))
                except:
                    pass
                return None
            
            prices = df['price'].apply(extract_price).dropna()
            if len(prices) > 0:
                fig = px.histogram(prices, nbins=20, title="Price Distribution")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating price chart: {e}")
        
        # Year distribution
        st.subheader("Year Distribution")
        try:
            years = pd.to_numeric(df['year_built'], errors='coerce').dropna()
            if len(years) > 0:
                fig = px.histogram(years, nbins=30, title="Year Built Distribution")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating year chart: {e}")
        
        # Top brands
        st.subheader("Top Brands")
        try:
            brands = df['title'].str.split().str[0].dropna()
            top_brands = brands.value_counts().head(10)
            fig = px.bar(x=top_brands.index, y=top_brands.values, title="Top 10 Brands")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating brand chart: {e}")
    else:
        st.warning("Database not available for insights.")

elif page == "💰 Investment Comparison":
    st.title("💰 Investment Comparison: Boats vs Financial Indices")
    
    if st.session_state.boat_market_analyzer and st.session_state.financial_fetcher:
        period = st.selectbox("Time Period", ["1y", "2y", "5y", "10y"], index=2)
        
        if st.button("Compare Performance", type="primary"):
            with st.spinner("Fetching data..."):
                # Get boat market performance
                boat_perf = st.session_state.boat_market_analyzer.calculate_market_performance()
                
                # Get financial indices
                financial_data = st.session_state.financial_fetcher.get_comparison_summary(period=period)
                
                # Display comparison
                st.subheader("Boat Market Performance")
                if 'total_return_pct' in boat_perf:
                    st.metric("Total Return", f"{boat_perf['total_return_pct']:.2f}%")
                
                st.subheader("Financial Indices Performance")
                if 'indices' in financial_data:
                    for idx_name, idx_data in financial_data['indices'].items():
                        if 'total_return_pct' in idx_data:
                            st.metric(idx_name, f"{idx_data['total_return_pct']:.2f}%")
    else:
        st.warning("Market analyzers not available.")

elif page == "🗺️ Map":
    st.title("🗺️ Boat Locations Map")
    
    if st.session_state.boat_db:
        limit = st.slider("Number of boats to show", 10, 1000, 100)
        
        if st.button("Load Map", type="primary"):
            boats = st.session_state.boat_db.get_boats_for_map(limit=limit)
            
            if boats:
                try:
                    import folium
                    from streamlit_folium import st_folium
                    
                    # Create map centered on first boat or default location
                    if boats and 'location_lat' in boats[0] and 'location_lon' in boats[0]:
                        center_lat = boats[0]['location_lat']
                        center_lon = boats[0]['location_lon']
                    else:
                        center_lat, center_lon = 43.7384, 7.4246  # Monaco
                    
                    m = folium.Map(location=[center_lat, center_lon], zoom_start=5)
                    
                    # Add markers
                    for boat in boats[:limit]:
                        if 'location_lat' in boat and 'location_lon' in boat:
                            folium.Marker(
                                [boat['location_lat'], boat['location_lon']],
                                popup=boat.get('title', 'Unknown')
                            ).add_to(m)
                    
                    st_folium(m, width=1200, height=600)
                except Exception as e:
                    st.error(f"Error creating map: {e}")
                    st.info("Map feature requires folium and streamlit-folium packages.")
    else:
        st.warning("Database not available for map.")

elif page == "📜 History":
    st.title("📜 Analysis History")
    
    if st.session_state.analysis_history:
        for entry in st.session_state.analysis_history:
            with st.expander(f"**{entry['filename']}** - {entry['timestamp'][:19]}"):
                analysis = entry['analysis']
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Boat Type:** {analysis.get('boat_type', 'Unknown')}")
                    st.markdown(f"**Brand:** {analysis.get('brand', 'Unknown')}")
                    st.markdown(f"**Model:** {analysis.get('model', 'Unknown')}")
                with col2:
                    st.markdown(f"**Confidence:** {analysis.get('confidence', 0)}%")
                    st.markdown(f"**Year:** {analysis.get('estimated_year', 'Unknown')}")
                    st.markdown(f"**Length:** {analysis.get('length_estimate', 'Unknown')}m")
    else:
        st.info("No analysis history yet. Upload and analyze images to see history here.")

# Footer
st.sidebar.markdown("---")
if st.session_state.analyzer:
    model_info = st.session_state.analyzer.get_model_info()
    st.sidebar.info(f"""
    **AI Model:** {model_info['model_name']}
    **Provider:** {model_info['provider']}
    """)
