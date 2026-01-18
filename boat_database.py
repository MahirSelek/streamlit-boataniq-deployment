"""
Boat Database Layer for BoatanIQ App
Handles loading and querying boat data from CSV and JSON files
"""

import pandas as pd
import json
import os
import random
from typing import List, Dict, Optional
from fuzzywuzzy import fuzz, process
import re
from geopy.distance import geodesic

class BoatDatabase:
    def __init__(self, csv_path: str, json_dir: str = None):
        """
        Initialize the boat database
        
        Args:
            csv_path: Path to the main CSV file with all boat data
            json_dir: Optional path to directory with individual JSON files
        """
        self.csv_path = csv_path
        self.json_dir = json_dir
        self.boats_df = None
        self.json_boats = {}
        
        # Popular boat locations for generating sample data
        self.popular_locations = [
            {"name": "Monaco", "lat": 43.7384, "lon": 7.4246, "country": "Monaco"},
            {"name": "Antibes", "lat": 43.5804, "lon": 7.1258, "country": "France"},
            {"name": "Portofino", "lat": 44.3039, "lon": 9.2089, "country": "Italy"},
            {"name": "Marina del Rey", "lat": 33.9759, "lon": -118.4481, "country": "USA"},
            {"name": "Fort Lauderdale", "lat": 26.1224, "lon": -80.1373, "country": "USA"},
            {"name": "Miami Beach", "lat": 25.7907, "lon": -80.1300, "country": "USA"},
            {"name": "Barcelona", "lat": 41.3759, "lon": 2.1825, "country": "Spain"},
            {"name": "Porto Cervo", "lat": 41.1306, "lon": 9.5306, "country": "Italy"},
            {"name": "Saint-Tropez", "lat": 43.2692, "lon": 6.6389, "country": "France"},
            {"name": "Cannes", "lat": 43.5528, "lon": 7.0174, "country": "France"},
            {"name": "Nice", "lat": 43.7102, "lon": 7.2620, "country": "France"},
            {"name": "Ibiza", "lat": 38.9067, "lon": 1.4206, "country": "Spain"},
            {"name": "Mykonos", "lat": 37.4467, "lon": 25.3289, "country": "Greece"},
            {"name": "Santorini", "lat": 36.3932, "lon": 25.4615, "country": "Greece"},
            {"name": "Dubai Marina", "lat": 25.0764, "lon": 55.1322, "country": "UAE"},
            {"name": "Sydney Harbour", "lat": -33.8568, "lon": 151.2153, "country": "Australia"},
            {"name": "Cape Town", "lat": -33.9249, "lon": 18.4241, "country": "South Africa"},
            {"name": "Newport RI", "lat": 41.4901, "lon": -71.3128, "country": "USA"},
            {"name": "Hamble", "lat": 50.8594, "lon": -1.3158, "country": "UK"},
            {"name": "Southampton", "lat": 50.9097, "lon": -1.4044, "country": "UK"},
        ]
        
        self._load_data()
    
    def _load_data(self):
        """Load boat data from CSV and JSON files"""
        try:
            # Load main CSV data
            self.boats_df = pd.read_csv(self.csv_path)
            print(f"Loaded {len(self.boats_df)} boats from CSV")
            
            # Add location data if not present
            self._add_location_data()
            
            # Initialize extracted columns for search
            self._initialize_extracted_columns()
            
            # Load individual JSON files if directory provided
            if self.json_dir and os.path.exists(self.json_dir):
                self._load_json_files()
                
        except Exception as e: 
            print(f"Error loading boat data: {e}")
            raise
    
    def _load_json_files(self):
        """Load individual JSON boat files"""
        json_files = [f for f in os.listdir(self.json_dir) if f.endswith('.json')]
        
        for json_file in json_files:
            try:
                with open(os.path.join(self.json_dir, json_file), 'r', encoding='utf-8') as f:
                    boat_data = json.load(f)
                    # Use filename as key for detailed data
                    self.json_boats[json_file] = boat_data
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        print(f"Loaded {len(self.json_boats)} detailed boat records from JSON")
    
    def _add_location_data(self):
        """Add location data to boats if not present"""
        if 'location_name' not in self.boats_df.columns:
            # Add random locations to boats
            self.boats_df['location_name'] = [random.choice(self.popular_locations)['name'] for _ in range(len(self.boats_df))]
            self.boats_df['location_lat'] = [random.choice(self.popular_locations)['lat'] for _ in range(len(self.boats_df))]
            self.boats_df['location_lon'] = [random.choice(self.popular_locations)['lon'] for _ in range(len(self.boats_df))]
            self.boats_df['location_country'] = [random.choice(self.popular_locations)['country'] for _ in range(len(self.boats_df))]
            print("Added location data to boats")
    
    def _initialize_extracted_columns(self):
        """Initialize extracted columns for search functionality"""
        if self.boats_df is None:
            return
        
        # Extract numeric year from year_built column
        def extract_year(year_str):
            if pd.isna(year_str):
                return None
            year_match = re.search(r'\b(19|20)\d{2}\b', str(year_str))
            return float(year_match.group()) if year_match else None
        
        # Extract dimensions (length and width)
        def extract_length_width(dimensions_str):
            if pd.isna(dimensions_str):
                return None, None
            dim_match = re.search(r'(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*m', str(dimensions_str))
            if dim_match:
                return float(dim_match.group(1)), float(dim_match.group(2))
            return None, None
        
        # Initialize extracted columns
        self.boats_df['extracted_year'] = self.boats_df['year_built'].apply(extract_year)
        
        # Extract length and width
        dimensions_extracted = self.boats_df['dimensions'].apply(extract_length_width)
        self.boats_df['length'] = [x[0] if x[0] is not None else None for x in dimensions_extracted]
        self.boats_df['width'] = [x[1] if x[1] is not None else None for x in dimensions_extracted]
        
        print("Initialized extracted columns for search")
    
    def search_by_brand(self, brand: str, limit: int = 10) -> List[Dict]:
        """
        Search boats by brand name using fuzzy matching
        
        Args:
            brand: Brand name to search for
            limit: Maximum number of results to return
            
        Returns:
            List of boat dictionaries matching the brand
        """
        if self.boats_df is None:
            return []
        
        # Extract brand from title using fuzzy matching
        brands = []
        for title in self.boats_df['title'].dropna():
            # Extract potential brand (first word or two)
            words = title.split()
            if len(words) >= 1:
                potential_brand = words[0]
                if len(words) >= 2:
                    potential_brand = f"{words[0]} {words[1]}"
                
                brands.append(potential_brand)
        
        # Find best brand matches
        best_matches = process.extract(brand, list(set(brands)), limit=5)
        
        results = []
        for match_brand, score in best_matches:
            if score >= 60:  # Minimum similarity threshold
                matching_boats = self.boats_df[
                    self.boats_df['title'].str.contains(match_brand, case=False, na=False)
                ].head(limit)
                
                for _, boat in matching_boats.iterrows():
                    results.append(self._boat_row_to_dict(boat))
        
        return results[:limit]
    
    def search_by_model(self, model: str, limit: int = 10) -> List[Dict]:
        """
        Search boats by model name
        
        Args:
            model: Model name to search for
            limit: Maximum number of results to return
            
        Returns:
            List of boat dictionaries matching the model
        """
        if self.boats_df is None:
            return []
        
        # Search for boats containing the model name
        matching_boats = self.boats_df[
            self.boats_df['title'].str.contains(model, case=False, na=False)
        ].head(limit)
        
        results = []
        for _, boat in matching_boats.iterrows():
            results.append(self._boat_row_to_dict(boat))
        
        return results
    
    def search_by_year_range(self, min_year: int, max_year: int, limit: int = 10) -> List[Dict]:
        """
        Search boats by year range
        
        Args:
            min_year: Minimum year
            max_year: Maximum year
            limit: Maximum number of results to return
            
        Returns:
            List of boat dictionaries in the year range
        """
        if self.boats_df is None:
            return []
        
        # Use pre-extracted year column
        
        # Filter by year range (handle None values)
        matching_boats = self.boats_df[
            (self.boats_df['extracted_year'].notna()) &
            (self.boats_df['extracted_year'] >= min_year) & 
            (self.boats_df['extracted_year'] <= max_year)
        ].head(limit)
        
        results = []
        for _, boat in matching_boats.iterrows():
            results.append(self._boat_row_to_dict(boat))
        
        return results
    
    def search_by_dimensions(self, length_range: tuple, width_range: tuple, limit: int = 10) -> List[Dict]:
        """
        Search boats by dimensions
        
        Args:
            length_range: Tuple of (min_length, max_length) in meters
            width_range: Tuple of (min_width, max_width) in meters
            limit: Maximum number of results to return
            
        Returns:
            List of boat dictionaries matching the dimensions
        """
        if self.boats_df is None:
            return []
        
        # Use pre-extracted dimension columns
        
        # Filter by dimensions (handle None values)
        length_mask = (self.boats_df['length'].notna()) & (self.boats_df['length'] >= length_range[0]) & (self.boats_df['length'] <= length_range[1])
        width_mask = (self.boats_df['width'].notna()) & (self.boats_df['width'] >= width_range[0]) & (self.boats_df['width'] <= width_range[1])
        
        matching_boats = self.boats_df[length_mask & width_mask].head(limit)
        
        results = []
        for _, boat in matching_boats.iterrows():
            results.append(self._boat_row_to_dict(boat))
        
        return results
    
    def find_similar_boats(self, detected_features: Dict, limit: int = 10) -> List[Dict]:
        """
        Find similar boats based on detected features from AI analysis
        
        Args:
            detected_features: Dictionary containing detected boat features
            limit: Maximum number of results to return
            
        Returns:
            List of similar boat dictionaries
        """
        results = []
        
        # Search by brand if detected
        if 'brand' in detected_features and detected_features['brand']:
            brand_results = self.search_by_brand(detected_features['brand'], limit=limit//2)
            results.extend(brand_results)
        
        # Search by model if detected
        if 'model' in detected_features and detected_features['model']:
            model_results = self.search_by_model(detected_features['model'], limit=limit//2)
            results.extend(model_results)
        
        # Search by year if detected
        if 'year' in detected_features and detected_features['year']:
            year = detected_features['year']
            year_results = self.search_by_year_range(year-2, year+2, limit=limit//3)
            results.extend(year_results)
        
        # Remove duplicates based on title
        seen_titles = set()
        unique_results = []
        for boat in results:
            if boat['title'] not in seen_titles:
                seen_titles.add(boat['title'])
                unique_results.append(boat)
        
        return unique_results[:limit]
    
    def get_all_boats(self, limit: int = 100) -> List[Dict]:
        """
        Get all boats with a limit
        
        Args:
            limit: Maximum number of boats to return
            
        Returns:
            List of all boat dictionaries
        """
        if self.boats_df is None:
            return []
        
        results = []
        for _, boat in self.boats_df.head(limit).iterrows():
            results.append(self._boat_row_to_dict(boat))
        
        return results
    
    def get_boat_by_id(self, boat_id: str) -> Optional[Dict]:
        """
        Get a specific boat by ID or title
        
        Args:
            boat_id: Boat ID or title to search for
            
        Returns:
            Boat dictionary or None if not found
        """
        if self.boats_df is None:
            return None
        
        # Try to find by exact title match first
        boat = self.boats_df[self.boats_df['title'] == boat_id]
        if not boat.empty:
            return self._boat_row_to_dict(boat.iloc[0])
        
        # Try fuzzy matching on title
        titles = self.boats_df['title'].tolist()
        best_match = process.extractOne(boat_id, titles, scorer=fuzz.ratio)
        
        if best_match and best_match[1] > 70:  # 70% similarity threshold
            boat = self.boats_df[self.boats_df['title'] == best_match[0]]
            if not boat.empty:
                return self._boat_row_to_dict(boat.iloc[0])
        
        return None
    
    def search_by_keywords(self, keywords: str, limit: int = 20) -> List[Dict]:
        """
        Search boats by keywords with EXACT matching only for flawless results
        
        Args:
            keywords: Keywords to search for
            limit: Maximum number of results to return
            
        Returns:
            List of boat dictionaries matching the keywords EXACTLY
        """
        if self.boats_df is None:
            return []
        
        keywords_lower = keywords.lower().strip()
        if not keywords_lower:
            return []
        
        results = []
        
        # 1. EXACT title matches (highest priority) - must contain ALL keywords
        keywords_list = keywords_lower.split()
        
        # Find boats that contain ALL keywords in the title
        exact_matches = self.boats_df[
            self.boats_df['title'].str.lower().str.contains('|'.join(keywords_list), na=False)
        ]
        
        # Filter to ensure ALL keywords are present
        for _, boat in exact_matches.iterrows():
            title_lower = str(boat['title']).lower()
            if all(keyword in title_lower for keyword in keywords_list):
                results.append(self._boat_row_to_dict(boat))
        
        # 2. EXACT brand matches (second priority)
        if len(results) < limit and 'brand' in self.boats_df.columns:
            try:
                brand_matches = self.boats_df[
                    self.boats_df['brand'].str.lower().str.contains(keywords_lower, na=False)
                ]
                
                for _, boat in brand_matches.iterrows():
                    if not any(r['title'] == boat['title'] for r in results):
                        results.append(self._boat_row_to_dict(boat))
                        if len(results) >= limit:
                            break
            except (KeyError, AttributeError, Exception):
                pass
        
        # 3. EXACT model matches (third priority)
        if len(results) < limit and 'model' in self.boats_df.columns:
            try:
                model_matches = self.boats_df[
                    self.boats_df['model'].str.lower().str.contains(keywords_lower, na=False)
                ]
                
                for _, boat in model_matches.iterrows():
                    if not any(r['title'] == boat['title'] for r in results):
                        results.append(self._boat_row_to_dict(boat))
                        if len(results) >= limit:
                            break
            except (KeyError, AttributeError, Exception):
                pass
        
        # 4. EXACT boat type matches (fourth priority)
        if len(results) < limit and 'boat_type' in self.boats_df.columns:
            try:
                type_matches = self.boats_df[
                    self.boats_df['boat_type'].str.lower().str.contains(keywords_lower, na=False)
                ]
                
                for _, boat in type_matches.iterrows():
                    if not any(r['title'] == boat['title'] for r in results):
                        results.append(self._boat_row_to_dict(boat))
                        if len(results) >= limit:
                            break
            except (KeyError, AttributeError, Exception):
                pass
        
        return results[:limit]
    
    def search_with_filters(self, keywords: str = "", filters: Dict = None, limit: int = 20) -> List[Dict]:
        """
        Search boats with advanced filters for precise results
        
        Args:
            keywords: Keywords to search for (optional)
            filters: Dictionary of filter criteria
            limit: Maximum number of results to return
            
        Returns:
            List of boat dictionaries matching the criteria
        """
        if self.boats_df is None:
            return []
        
        # Start with all boats
        filtered_df = self.boats_df.copy()
        
        # Apply keyword search if provided
        if keywords and keywords.strip():
            keywords_lower = keywords.lower().strip()
            keywords_list = keywords_lower.split()
            
            # Find boats that contain ALL keywords in the title
            keyword_mask = filtered_df['title'].str.lower().str.contains('|'.join(keywords_list), na=False)
            
            # Filter to ensure ALL keywords are present
            for _, boat in filtered_df[keyword_mask].iterrows():
                title_lower = str(boat['title']).lower()
                if not all(keyword in title_lower for keyword in keywords_list):
                    keyword_mask.loc[boat.name] = False
            
            filtered_df = filtered_df[keyword_mask]
        
        # Apply filters if provided
        if filters:
            # Brand filter
            if filters.get('brand') and 'brand' in filtered_df.columns:
                try:
                    brand_filter = filtered_df['brand'].str.lower().str.contains(filters['brand'].lower(), na=False)
                    filtered_df = filtered_df[brand_filter]
                except (KeyError, AttributeError, Exception):
                    pass
            
            # Model filter
            if filters.get('model') and 'model' in filtered_df.columns:
                try:
                    model_filter = filtered_df['model'].str.lower().str.contains(filters['model'].lower(), na=False)
                    filtered_df = filtered_df[model_filter]
                except (KeyError, AttributeError, Exception):
                    pass
            
            # Boat type filter
            if filters.get('boat_type') and 'boat_type' in filtered_df.columns:
                try:
                    type_filter = filtered_df['boat_type'].str.lower().str.contains(filters['boat_type'].lower(), na=False)
                    filtered_df = filtered_df[type_filter]
                except (KeyError, AttributeError, Exception):
                    pass
            
            # Year range filter
            if filters.get('year_min') or filters.get('year_max'):
                year_min = filters.get('year_min', 1900)
                year_max = filters.get('year_max', 2030)
                year_filter = (filtered_df['year_built'] >= year_min) & (filtered_df['year_built'] <= year_max)
                filtered_df = filtered_df[year_filter]
            
            # Price range filter
            if filters.get('price_min') or filters.get('price_max'):
                # Extract numeric price values
                price_min = filters.get('price_min', 0)
                price_max = filters.get('price_max', 10000000)
                
                # Convert price column to numeric (remove EUR, commas, etc.)
                price_numeric = filtered_df['price'].str.replace('EUR', '').str.replace(',', '').str.replace('.', '').str.replace('-', '').str.strip()
                price_numeric = pd.to_numeric(price_numeric, errors='coerce')
                
                price_filter = (price_numeric >= price_min) & (price_numeric <= price_max)
                filtered_df = filtered_df[price_filter]
            
            # Length range filter
            if filters.get('length_min') or filters.get('length_max'):
                length_min = filters.get('length_min', 0)
                length_max = filters.get('length_max', 200)
                length_filter = (filtered_df['length'] >= length_min) & (filtered_df['length'] <= length_max)
                filtered_df = filtered_df[length_filter]
            
            # Width range filter
            if filters.get('width_min') or filters.get('width_max'):
                width_min = filters.get('width_min', 0)
                width_max = filters.get('width_max', 50)
                width_filter = (filtered_df['width'] >= width_min) & (filtered_df['width'] <= width_max)
                filtered_df = filtered_df[width_filter]
            
            # Location filter
            if filters.get('location'):
                location_filter = filtered_df['location_name'].str.lower().str.contains(filters['location'].lower(), na=False)
                filtered_df = filtered_df[location_filter]
        
        # Convert to list of dictionaries
        results = []
        for _, boat in filtered_df.head(limit).iterrows():
            results.append(self._boat_row_to_dict(boat))
        
        return results
    
    def get_filter_options(self) -> Dict:
        """
        Get available filter options for the UI
        
        Returns:
            Dictionary with filter options
        """
        if self.boats_df is None or self.boats_df.empty:
            return {
                'brands': [],
                'models': [],
                'boat_types': [],
                'locations': [],
                'year_range': {'min': 1900, 'max': 2030},
                'length_range': {'min': 0, 'max': 200},
                'width_range': {'min': 0, 'max': 50}
            }
        
        try:
            # Get unique values for dropdown filters - check if columns exist
            brands = []
            try:
                if 'brand' in self.boats_df.columns:
                    brands = sorted(self.boats_df['brand'].dropna().unique().tolist())
                elif 'title' in self.boats_df.columns:
                    # Extract brands from title if brand column doesn't exist
                    titles = self.boats_df['title'].dropna().unique().tolist()
                    brand_set = set()
                    for title in titles:
                        if title and isinstance(title, str):
                            words = title.split()
                            if words:
                                brand_set.add(words[0])
                    brands = sorted(list(brand_set))
            except (KeyError, AttributeError, Exception) as e:
                print(f"Warning: Error extracting brands: {e}")
                brands = []
            
            models = []
            try:
                if 'model' in self.boats_df.columns:
                    models = sorted(self.boats_df['model'].dropna().unique().tolist())
            except (KeyError, AttributeError, Exception) as e:
                print(f"Warning: Error extracting models: {e}")
                models = []
            
            boat_types = []
            try:
                if 'boat_type' in self.boats_df.columns:
                    boat_types = sorted(self.boats_df['boat_type'].dropna().unique().tolist())
            except (KeyError, AttributeError, Exception) as e:
                print(f"Warning: Error extracting boat_types: {e}")
                boat_types = []
            
            locations = []
            try:
                if 'location_name' in self.boats_df.columns:
                    locations = sorted(self.boats_df['location_name'].dropna().unique().tolist())
            except (KeyError, AttributeError, Exception) as e:
                print(f"Warning: Error extracting locations: {e}")
                locations = []
            
            # Get numeric ranges with error handling
            year_range = {'min': 1900, 'max': 2030}
            try:
                if 'year_built' in self.boats_df.columns:
                    year_col = self.boats_df['year_built'].dropna()
                    if not year_col.empty:
                        try:
                            year_range = {
                                'min': int(year_col.min()),
                                'max': int(year_col.max())
                            }
                        except (ValueError, TypeError):
                            pass
            except (KeyError, AttributeError, Exception) as e:
                print(f"Warning: Error extracting year range: {e}")
            
            length_range = {'min': 0, 'max': 200}
            try:
                if 'length' in self.boats_df.columns:
                    length_col = self.boats_df['length'].dropna()
                    if not length_col.empty:
                        try:
                            length_range = {
                                'min': float(length_col.min()),
                                'max': float(length_col.max())
                            }
                        except (ValueError, TypeError):
                            pass
            except (KeyError, AttributeError, Exception) as e:
                print(f"Warning: Error extracting length range: {e}")
            
            width_range = {'min': 0, 'max': 50}
            try:
                if 'width' in self.boats_df.columns:
                    width_col = self.boats_df['width'].dropna()
                    if not width_col.empty:
                        try:
                            width_range = {
                                'min': float(width_col.min()),
                                'max': float(width_col.max())
                            }
                        except (ValueError, TypeError):
                            pass
            except (KeyError, AttributeError, Exception) as e:
                print(f"Warning: Error extracting width range: {e}")
            
            return {
                'brands': brands[:50],  # Limit to top 50 for performance
                'models': models[:50],
                'boat_types': boat_types[:20],
                'locations': locations[:50],
                'year_range': year_range,
                'length_range': length_range,
                'width_range': width_range
            }
        except Exception as e:
            print(f"❌ Filter options error: {e}")
            import traceback
            traceback.print_exc()
            # Return safe defaults
            return {
                'brands': [],
                'models': [],
                'boat_types': [],
                'locations': [],
                'year_range': {'min': 1900, 'max': 2030},
                'length_range': {'min': 0, 'max': 200},
                'width_range': {'min': 0, 'max': 50}
            }
    
    def _boat_row_to_dict(self, row) -> Dict:
        """Convert pandas row to dictionary"""
        def clean_value(value):
            """Clean NaN and None values for JSON serialization"""
            if pd.isna(value) or value is None or str(value).lower() == 'nan':
                return None
            return value
        
        def extract_brand_model(title):
            """Extract brand and model from title"""
            if not title:
                return None, None
            
            # Common boat brands to look for
            brands = [
                'Bavaria', 'Beneteau', 'Jeanneau', 'Princess', 'Sunseeker', 'Azimut', 'Ferretti',
                'Pershing', 'Riva', 'Sea Ray', 'Bayliner', 'Crownline', 'Malibu', 'Mastercraft',
                'Boston Whaler', 'Grady-White', 'Regal', 'Formula', 'Cobalt', 'Chaparral',
                'Four Winns', 'Maxum', 'Larson', 'Glastron', 'Crestliner', 'Lund', 'Tracker',
                'Ranger', 'Nitro', 'Bass Pro', 'Skeeter', 'Stratos', 'Triton', 'Champion',
                'Mayland', 'Quicksilver', 'Cranchi', 'Galaxy', 'Seaco', 'Crownline', 'Axis',
                'Sea Storm', 'Galeasen', 'Focus', 'Plastex', 'Mercury'
            ]
            
            title_lower = title.lower()
            brand = None
            model = None
            
            # Find brand in title
            for b in brands:
                if b.lower() in title_lower:
                    brand = b
                    break
            
            # Extract model (usually after brand)
            if brand:
                brand_pos = title_lower.find(brand.lower())
                model_part = title[brand_pos + len(brand):].strip()
                if model_part:
                    # Take first part as model (before any additional text)
                    model = model_part.split()[0] if model_part.split() else None
            
            return brand, model
        
        def determine_boat_type(title, dimensions):
            """Determine boat type from title and dimensions"""
            if not title:
                return None
            
            title_lower = title.lower()
            
            # Sailing boats
            if any(word in title_lower for word in ['sail', 'yacht', 'sloop', 'catamaran', 'trimaran', 'ketch', 'schooner']):
                return 'Sailing Yacht'
            
            # Motor boats
            if any(word in title_lower for word in ['motor', 'cruiser', 'bowrider', 'deck', 'cuddy', 'walkaround']):
                return 'Motor Yacht'
            
            # Fishing boats
            if any(word in title_lower for word in ['fishing', 'bass', 'center console', 'console']):
                return 'Fishing Boat'
            
            # Speed boats
            if any(word in title_lower for word in ['speed', 'racing', 'sport', 'runabout']):
                return 'Speed Boat'
            
            # Try to determine from dimensions if available
            if dimensions:
                try:
                    # Extract length from dimensions (e.g., "12.43 x 4.20 m")
                    length_str = dimensions.split('x')[0].strip()
                    length = float(length_str.replace('m', '').strip())
                    
                    if length < 6:
                        return 'Small Boat'
                    elif length < 12:
                        return 'Motor Yacht'
                    else:
                        return 'Sailing Yacht'
                except:
                    pass
            
            return 'Motor Yacht'  # Default
        
        def extract_dimensions(dimensions):
            """Extract length and width from dimensions string"""
            if not dimensions:
                return None, None
            
            try:
                # Format: "12.43 x 4.20 m"
                parts = dimensions.split('x')
                if len(parts) >= 2:
                    length = parts[0].strip().replace('m', '').strip()
                    width = parts[1].strip().replace('m', '').strip()
                    return length, width
            except:
                pass
            
            return None, None
        
        # Extract basic info
        title = clean_value(row['title'])
        brand, model = extract_brand_model(title)
        length, width = extract_dimensions(clean_value(row['dimensions']))
        boat_type = determine_boat_type(title, clean_value(row['dimensions']))
        
        return {
            'title': title,
            'price': clean_value(row['price']),
            'dimensions': clean_value(row['dimensions']),
            'engine_performance': clean_value(row['engine_performance']),
            'year_built': clean_value(row['year_built']),
            'brand': brand,
            'model': model,
            'boat_type': boat_type,
            'length': length,
            'width': width,
            'hull_material': None,  # Not available in CSV
            'engine_type': None,    # Not available in CSV
            'hull_type': None,      # Not available in CSV
            'features': None,       # Not available in CSV
            'equipment': None,      # Not available in CSV
            'description': None     # Not available in CSV
        }
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        if self.boats_df is None:
            return {}
        
        # Extract numeric years for statistics
        def extract_year_for_stats(year_str):
            if pd.isna(year_str):
                return None
            year_match = re.search(r'\b(19|20)\d{2}\b', str(year_str))
            return int(year_match.group()) if year_match else None
        
        years = self.boats_df['year_built'].apply(extract_year_for_stats).dropna()
        
        # Handle price range safely
        price_min = self.boats_df['price'].min()
        price_max = self.boats_df['price'].max()
        
        # Handle year range safely
        year_min = 'N/A'
        year_max = 'N/A'
        if len(years) > 0:
            try:
                year_min = int(years.min())
                year_max = int(years.max())
            except (ValueError, TypeError):
                year_min = 'N/A'
                year_max = 'N/A'
        
        stats = {
            'total_boats': len(self.boats_df),
            'unique_brands': len(self.boats_df['title'].str.split().str[0].unique()),
            'year_range': {
                'min': year_min,
                'max': year_max
            },
            'price_range': {
                'min': str(price_min) if pd.notna(price_min) else 'N/A',
                'max': str(price_max) if pd.notna(price_max) else 'N/A'
            }
        }
        
        return stats
    
    def search_by_location(self, lat: float, lon: float, radius_km: float = 50, limit: int = 20) -> List[Dict]:
        """
        Search boats within a specified radius of given coordinates
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate  
            radius_km: Search radius in kilometers
            limit: Maximum number of results to return
            
        Returns:
            List of boat dictionaries within the radius
        """
        if self.boats_df is None:
            return []
        
        nearby_boats = []
        
        for _, boat in self.boats_df.iterrows():
            if pd.notna(boat.get('location_lat')) and pd.notna(boat.get('location_lon')):
                distance = geodesic((lat, lon), (boat['location_lat'], boat['location_lon'])).kilometers
                
                if distance <= radius_km:
                    boat_dict = boat.to_dict()
                    boat_dict['distance_km'] = round(distance, 2)
                    nearby_boats.append(boat_dict)
        
        # Sort by distance and limit results
        nearby_boats.sort(key=lambda x: x['distance_km'])
        return nearby_boats[:limit]
    
    def search_by_location_name(self, location_name: str, radius_km: float = 50, limit: int = 20) -> List[Dict]:
        """
        Search boats near a location by name
        
        Args:
            location_name: Name of the location to search near
            radius_km: Search radius in kilometers
            limit: Maximum number of results to return
            
        Returns:
            List of boat dictionaries near the location
        """
        # Find matching location in our popular locations
        matching_location = None
        for loc in self.popular_locations:
            if location_name.lower() in loc['name'].lower():
                matching_location = loc
                break
        
        if matching_location:
            return self.search_by_location(
                matching_location['lat'], 
                matching_location['lon'], 
                radius_km, 
                limit
            )
        
        return []
    
    def get_boats_for_map(self, limit: int = 1000) -> List[Dict]:
        """
        Get boats with location data for map visualization
        
        Args:
            limit: Maximum number of boats to return
            
        Returns:
            List of boat dictionaries with location data
        """
        if self.boats_df is None:
            return []
        
        boats_with_location = []
        
        for _, boat in self.boats_df.iterrows():
            if (pd.notna(boat.get('location_lat')) and 
                pd.notna(boat.get('location_lon')) and 
                pd.notna(boat.get('location_name'))):
                
                boat_dict = boat.to_dict()
                boats_with_location.append(boat_dict)
                
                if len(boats_with_location) >= limit:
                    break
        
        return boats_with_location
    
    def get_location_statistics(self) -> Dict:
        """
        Get statistics about boat locations
        
        Returns:
            Dictionary with location statistics
        """
        if self.boats_df is None:
            return {}
        
        location_stats = {}
        
        # Count boats by country
        if 'location_country' in self.boats_df.columns:
            country_counts = self.boats_df['location_country'].value_counts().to_dict()
            location_stats['boats_by_country'] = country_counts
        
        # Count boats by location
        if 'location_name' in self.boats_df.columns:
            location_counts = self.boats_df['location_name'].value_counts().head(10).to_dict()
            location_stats['top_locations'] = location_counts
        
        return location_stats
