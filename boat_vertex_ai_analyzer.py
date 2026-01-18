"""
Boat AI Analyzer for BoatanIQ App using Google Cloud Vertex AI
Uses Gemini Flash 2.0 for multimodal boat recognition and analysis
Modified for Streamlit deployment with environment variable support
"""

import os
import json
from typing import Dict, Optional
from PIL import Image
from io import BytesIO
import base64

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Part
    from google.oauth2 import service_account
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False

class BoatVertexAIAnalyzer:
    def __init__(self, credentials_path: str = None, credentials_json: str = None, project_id: str = None, location: str = "us-central1"):
        """
        Initialize the Vertex AI analyzer
        
        Args:
            credentials_path: Path to the service account JSON file (optional)
            credentials_json: JSON string of credentials (for Streamlit secrets)
            project_id: Google Cloud project ID (will be extracted from credentials if not provided)
            location: Google Cloud region for Vertex AI
        """
        if not VERTEX_AI_AVAILABLE:
            raise ImportError("Google Cloud Vertex AI libraries not installed. Run: pip install google-cloud-aiplatform")
        
        self.location = location
        credentials_data = None
        
        # Try to get credentials from different sources
        if credentials_json:
            # Use credentials from JSON string (Streamlit secrets)
            credentials_data = json.loads(credentials_json)
            self.project_id = project_id or credentials_data.get('project_id')
            # Create credentials from dict
            credentials = service_account.Credentials.from_service_account_info(credentials_data)
        elif credentials_path and os.path.exists(credentials_path):
            # Use credentials from file path
            with open(credentials_path, 'r') as f:
                credentials_data = json.load(f)
            self.project_id = project_id or credentials_data.get('project_id')
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
        else:
            raise FileNotFoundError("Credentials not found. Please provide credentials_path or credentials_json")
        
        if not self.project_id:
            raise ValueError("Project ID not found in credentials")
        
        # Initialize Vertex AI
        try:
            vertexai.init(project=self.project_id, location=self.location, credentials=credentials)
            
            # Initialize Gemini Flash 2.0 model
            self.model = GenerativeModel("gemini-2.0-flash-exp")
            
            print(f"✅ Vertex AI initialized successfully")
            print(f"   Project: {self.project_id}")
            print(f"   Location: {self.location}")
            print(f"   Model: Gemini Flash 2.0")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Vertex AI: {e}")
        
        # Define the analysis prompt optimized for Gemini Flash 2.0
        self.analysis_prompt = """
        You are an expert marine analyst specializing in boat identification and analysis. You have access to a comprehensive database of 30,000+ boats from major manufacturers worldwide. Analyze this boat image and provide detailed, accurate information with the goal of identifying the exact boat model.

        CRITICAL VALIDATION: Before analyzing, you MUST validate the image:
        1. Is this image actually a boat? (not a car, plane, building, or other object)
        2. Is the image clear and not too blurry?
        3. Is the boat clearly visible and from a reasonable angle?
        4. Can you see enough detail to identify the boat?
        
        If the image is NOT suitable (not a boat, too blurry, wrong angle, unclear), set "is_valid_image" to false and "rejection_reason" with a clear explanation. Do NOT proceed with analysis if the image is unsuitable.

        CRITICAL: Focus on identifying the EXACT boat model that might exist in our database. Look for:
        - Specific brand logos, names, or distinctive styling
        - Unique hull designs, cabin shapes, or deck layouts
        - Characteristic features of specific model lines
        - Era-appropriate design elements

        Please respond with a JSON object containing the following information:

        {
            "is_valid_image": true,
            "rejection_reason": null,
            "image_quality_assessment": "Assessment of image quality (Clear, Acceptable, Blurry, Poor)",
            "boat_type": "Specific type of boat (e.g., Sailing Yacht, Motor Yacht, Cruiser, Sport Boat, Fishing Boat, Catamaran, etc.)",
            "brand": "Manufacturer brand name if clearly identifiable (e.g., Bavaria, Beneteau, Jeanneau, Princess, Sunseeker, Sea Ray, Bayliner, etc.)",
            "model": "Specific model name if identifiable (e.g., Oceanis 40.1, Cruiser 34, Sundancer 320, etc.)",
            "model_line": "Model line or series if identifiable (e.g., Oceanis, Cruiser, Sundancer, etc.)",
            "estimated_year": "Estimated year of manufacture (provide range if uncertain, e.g., '2018-2022')",
            "length_estimate": "Estimated length in meters (e.g., '12.5', '8.2')",
            "width_estimate": "Estimated width/beam in meters (e.g., '4.2', '2.8')",
            "hull_material": "Hull material if identifiable (e.g., Fiberglass, Steel, Aluminum, Wood)",
            "engine_type": "Engine configuration if visible (e.g., Inboard Diesel, Outboard, Sail Only, Twin Engine)",
            "hull_type": "Hull design type (e.g., Monohull, Catamaran, Trimaran, Deep V, Planing Hull)",
            "key_features": ["List of distinctive features visible in the image that help identify the model"],
            "distinctive_elements": ["Unique design elements, logos, or features that are model-specific"],
            "condition": "Overall condition assessment (Excellent, Good, Fair, Poor)",
            "price_estimate": "Estimated market value range if possible (e.g., '€150,000-€200,000')",
            "confidence": "Confidence level (0-100) of the overall analysis. MUST be honest - if image is unclear or not a boat, set confidence below 30",
            "detailed_description": "Comprehensive, detailed description of what you observe in the image. Include: hull design analysis, cabin structure, deck layout, distinctive features, design era characteristics, potential use case, and any unique elements that make this boat special. Write 3-4 paragraphs with rich detail.",
            "identification_clues": "Specific visual clues that help identify this exact boat model",
            "technical_specs": {
                "sail_area": "Estimated sail area if sailing vessel",
                "engine_power": "Estimated engine power if visible",
                "fuel_capacity": "Estimated fuel capacity if determinable",
                "water_capacity": "Estimated water capacity if determinable",
                "max_speed": "Estimated maximum speed if determinable",
                "cruising_range": "Estimated cruising range if determinable",
                "berths": "Number of sleeping berths if visible",
                "headroom": "Estimated headroom in cabin if visible"
            },
            "design_analysis": {
                "hull_design": "Detailed analysis of hull shape, keel type, and design philosophy",
                "cabin_layout": "Description of cabin arrangement and interior design",
                "deck_features": "Analysis of deck layout, cockpit design, and outdoor spaces",
                "aerodynamics": "Analysis of wind resistance and performance characteristics"
            },
            "market_positioning": {
                "target_market": "Primary target market (luxury, family, racing, etc.)",
                "competitors": "Similar boats in the same category",
                "unique_selling_points": "What makes this boat special or different",
                "ideal_use_cases": "Best use cases for this boat type"
            },
            "historical_context": {
                "design_era": "When this design style was popular",
                "manufacturer_history": "Brief history of the manufacturer if known",
                "model_evolution": "How this model fits in the manufacturer's lineup",
                "market_reception": "How this model was received in the market"
            }
        }

        Guidelines:
        - FIRST: Validate if this is actually a boat image. If not, set is_valid_image=false and provide rejection_reason
        - If image is blurry, unclear, or from a bad angle, set is_valid_image=false with appropriate rejection_reason
        - Be honest about confidence - if uncertain, set confidence below 50. If very uncertain or image is poor quality, set below 30
        - Be as specific and accurate as possible in identifying the exact model
        - Look for brand-specific design elements, logos, and styling cues
        - Consider the boat's design era and how it fits with known model lines
        - Analyze hull shape, cabin design, deck layout, and other identifying features
        - If you can identify a specific model, provide high confidence (70+)
        - Focus on features that would help match this boat to our database entries
        - Consider popular brands like Bavaria, Beneteau, Jeanneau, Princess, Sunseeker, Sea Ray, Bayliner, etc.
        - REJECTION REASONS should be clear and helpful: "Image is too blurry", "This does not appear to be a boat", "Boat is not clearly visible", "Image angle is too extreme", etc.
        """
    
    def analyze_boat_image_from_bytes(self, image_bytes: bytes) -> Dict:
        """
        Analyze a boat image from bytes (for web uploads)
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Dictionary containing analyzed boat features
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert image to JPEG bytes for Vertex AI
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=95)
            processed_bytes = buffer.getvalue()
            
            # Create image part for Vertex AI
            image_part = Part.from_data(
                data=processed_bytes,
                mime_type="image/jpeg"
            )
            
            # Generate analysis using Vertex AI
            response = self.model.generate_content([self.analysis_prompt, image_part])
            
            # Parse response
            analysis_text = response.text.strip()
            
            # Try to extract JSON from response
            try:
                # Look for JSON in the response
                start_idx = analysis_text.find('{')
                end_idx = analysis_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_str = analysis_text[start_idx:end_idx]
                    analysis_result = json.loads(json_str)
                else:
                    # Fallback: create structured response from text
                    analysis_result = self._parse_text_response(analysis_text)
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                # Fallback parsing if JSON is malformed
                analysis_result = self._parse_text_response(analysis_text)
            
            # Add metadata
            analysis_result['raw_response'] = analysis_text
            analysis_result['model_used'] = 'gemini-2.0-flash-exp'
            analysis_result['analyzer_type'] = 'vertex_ai'
            
            # Handle validation fields (default to valid if not present)
            if 'is_valid_image' not in analysis_result:
                analysis_result['is_valid_image'] = True
            if 'rejection_reason' not in analysis_result:
                analysis_result['rejection_reason'] = None
            
            # Check confidence threshold
            confidence = analysis_result.get('confidence', 0)
            if isinstance(confidence, str):
                try:
                    confidence = float(confidence)
                except:
                    confidence = 0
            
            # If confidence is very low or image is invalid, mark as rejected
            if not analysis_result.get('is_valid_image', True) or confidence < 30:
                if not analysis_result.get('rejection_reason'):
                    if confidence < 30:
                        analysis_result['rejection_reason'] = 'AI confidence too low - image may be unclear, not a boat, or from a poor angle'
                    else:
                        analysis_result['rejection_reason'] = 'Image validation failed'
            
            return analysis_result
            
        except Exception as e:
            return {
                'error': f"Error analyzing image: {str(e)}",
                'confidence': 0,
                'model_used': 'gemini-2.0-flash-exp',
                'analyzer_type': 'vertex_ai'
            }
    
    def _parse_text_response(self, text: str) -> Dict:
        """
        Parse text response when JSON parsing fails
        
        Args:
            text: Raw response text
            
        Returns:
            Structured dictionary
        """
        result = {
            'boat_type': 'Unknown',
            'brand': 'Unknown',
            'model': 'Unknown',
            'estimated_year': 'Unknown',
            'length_estimate': 'Unknown',
            'width_estimate': 'Unknown',
            'hull_material': 'Unknown',
            'engine_type': 'Unknown',
            'hull_type': 'Unknown',
            'key_features': [],
            'condition': 'Unknown',
            'price_estimate': 'Unknown',
            'confidence': 50,
            'detailed_description': text,
            'technical_specs': {}
        }
        
        # Try to extract some information using keyword matching
        text_lower = text.lower()
        
        # Extract boat type
        boat_types = ['sailing yacht', 'motor yacht', 'cruiser', 'sport boat', 'fishing boat', 'catamaran', 'speedboat', 'motorboat']
        for boat_type in boat_types:
            if boat_type in text_lower:
                result['boat_type'] = boat_type.title()
                break
        
        # Extract brand
        brands = ['bavaria', 'beneteau', 'jeanneau', 'princess', 'sunseeker', 'azimut', 'ferretti', 'pershing', 'riva', 'sea ray']
        for brand in brands:
            if brand in text_lower:
                result['brand'] = brand.title()
                break
        
        # Extract year
        import re
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match:
            result['estimated_year'] = year_match.group()
        
        return result
    
    def get_analysis_summary(self, analysis_result: Dict) -> str:
        """
        Generate a human-readable summary of the analysis
        
        Args:
            analysis_result: Result from analyze_boat_image
            
        Returns:
            Formatted summary string
        """
        if 'error' in analysis_result:
            return f"Analysis Error: {analysis_result['error']}"
        
        summary_parts = []
        
        # Basic info
        if analysis_result.get('boat_type') != 'Unknown':
            summary_parts.append(f"**Boat Type:** {analysis_result['boat_type']}")
        
        if analysis_result.get('brand') != 'Unknown':
            summary_parts.append(f"**Brand:** {analysis_result['brand']}")
        
        if analysis_result.get('model') != 'Unknown':
            summary_parts.append(f"**Model:** {analysis_result['model']}")
        
        if analysis_result.get('estimated_year') != 'Unknown':
            summary_parts.append(f"**Estimated Year:** {analysis_result['estimated_year']}")
        
        # Dimensions
        if analysis_result.get('length_estimate') != 'Unknown':
            summary_parts.append(f"**Estimated Length:** {analysis_result['length_estimate']}m")
        
        if analysis_result.get('width_estimate') != 'Unknown':
            summary_parts.append(f"**Estimated Width:** {analysis_result['width_estimate']}m")
        
        # Hull and engine info
        if analysis_result.get('hull_type') != 'Unknown':
            summary_parts.append(f"**Hull Type:** {analysis_result['hull_type']}")
        
        if analysis_result.get('engine_type') != 'Unknown':
            summary_parts.append(f"**Engine Type:** {analysis_result['engine_type']}")
        
        # Features
        if analysis_result.get('key_features'):
            features = analysis_result['key_features']
            if isinstance(features, list) and features:
                summary_parts.append(f"**Key Features:** {', '.join(features)}")
        
        # Condition and price
        if analysis_result.get('condition') != 'Unknown':
            summary_parts.append(f"**Condition:** {analysis_result['condition']}")
        
        if analysis_result.get('price_estimate') != 'Unknown':
            summary_parts.append(f"**Price Estimate:** {analysis_result['price_estimate']}")
        
        # Technical specs
        if analysis_result.get('technical_specs'):
            tech_specs = analysis_result['technical_specs']
            if tech_specs:
                summary_parts.append("**Technical Specifications:**")
                for spec, value in tech_specs.items():
                    if value and value != 'Unknown':
                        summary_parts.append(f"  - {spec.replace('_', ' ').title()}: {value}")
        
        # Confidence
        confidence = analysis_result.get('confidence', 0)
        summary_parts.append(f"**Analysis Confidence:** {confidence}%")
        
        # Model info
        summary_parts.append(f"**AI Model:** {analysis_result.get('model_used', 'Unknown')}")
        
        # Detailed description
        if analysis_result.get('detailed_description'):
            summary_parts.append(f"\n**Detailed Analysis:**\n{analysis_result['detailed_description']}")
        
        return '\n'.join(summary_parts)
    
    def get_model_info(self) -> Dict:
        """
        Get information about the AI model being used
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': 'gemini-2.0-flash-exp',
            'provider': 'Google Cloud Vertex AI',
            'location': self.location,
            'project_id': self.project_id,
            'analyzer_type': 'vertex_ai',
            'capabilities': [
                'Multimodal analysis (text + images)',
                'High accuracy boat identification',
                'Detailed technical specifications',
                'Confidence scoring',
                'Real-time processing'
            ]
        }
