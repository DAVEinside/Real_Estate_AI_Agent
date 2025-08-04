"""Real estate data sources implementation."""

import asyncio
import logging
import pandas as pd
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aiohttp
from fredapi import Fred
from census import Census
import geopandas as gpd
import osmnx as ox

from ..config import settings

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    def __init__(self, name: str):
        self.name = name
        self.last_updated: Optional[datetime] = None
    
    @abstractmethod
    async def fetch_data(self, **kwargs) -> pd.DataFrame:
        """Fetch data from the source."""
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate the fetched data."""
        pass


class ZillowDataSource(DataSource):
    """Zillow Research Data source for housing market data."""
    
    BASE_URL = "https://files.zillowstatic.com/research/public_csvs/"
    
    DATASETS = {
        'home_values': 'zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
        'rentals': 'rentals/Metro_zori_sm_month.csv',
        'inventory': 'inventory/Metro_invt_fs_uc_sfrcondo_sm_month.csv',
        'sales': 'sales/Metro_median_sale_price_uc_sfrcondo_sm_month.csv',
        'price_cuts': 'price_cuts/Metro_perc_listings_price_cut_uc_sfrcondo_sm_month.csv'
    }
    
    def __init__(self):
        super().__init__("Zillow Research Data")
    
    async def fetch_data(self, dataset: str = 'home_values', **kwargs) -> pd.DataFrame:
        """Fetch Zillow research data."""
        try:
            if dataset not in self.DATASETS:
                raise ValueError(f"Unknown dataset: {dataset}")
            
            url = f"{self.BASE_URL}{self.DATASETS[dataset]}"
            logger.info(f"Fetching Zillow {dataset} data from {url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        # Use StringIO to read CSV content
                        from io import StringIO
                        df = pd.read_csv(StringIO(content))
                        
                        # Clean and standardize column names
                        df = self._clean_data(df, dataset)
                        self.last_updated = datetime.now()
                        logger.info(f"Successfully fetched {len(df)} records from Zillow {dataset}")
                        return df
                    else:
                        raise Exception(f"HTTP {response.status}: {await response.text()}")
                        
        except Exception as e:
            logger.error(f"Error fetching Zillow data: {e}")
            raise
    
    def _clean_data(self, df: pd.DataFrame, dataset: str) -> pd.DataFrame:
        """Clean and standardize Zillow data."""
        # Melt time series data from wide to long format
        id_cols = ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName']
        id_cols = [col for col in id_cols if col in df.columns]
        
        date_cols = [col for col in df.columns if col not in id_cols]
        
        if date_cols:
            df_long = df.melt(
                id_vars=id_cols,
                value_vars=date_cols,
                var_name='date',
                value_name=f'{dataset}_value'
            )
            
            # Convert date column to datetime
            df_long['date'] = pd.to_datetime(df_long['date'])
            df_long = df_long.dropna(subset=[f'{dataset}_value'])
            
            return df_long
        
        return df
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate Zillow data."""
        required_columns = ['RegionName', 'date']
        return all(col in data.columns for col in required_columns) and len(data) > 0


class FREDDataSource(DataSource):
    """Federal Reserve Economic Data source."""
    
    SERIES_IDS = {
        'zhvi_national': 'USAUCSFRCONDOSMSAMID',
        'mortgage_rates': 'MORTGAGE30US',
        'unemployment': 'UNRATE',
        'cpi': 'CPIAUCSL',
        'gdp': 'GDP',
        'housing_starts': 'HOUST',
        'building_permits': 'PERMIT'
    }
    
    def __init__(self):
        super().__init__("FRED Economic Data")
        if settings.fred_api_key:
            self.fred = Fred(api_key=settings.fred_api_key)
        else:
            logger.warning("FRED API key not provided")
            self.fred = None
    
    async def fetch_data(self, series: str = 'zhvi_national', start: str = '2000-01-01', **kwargs) -> pd.DataFrame:
        """Fetch FRED economic data."""
        if not self.fred:
            raise Exception("FRED API key required")
        
        try:
            if series not in self.SERIES_IDS:
                raise ValueError(f"Unknown series: {series}")
            
            series_id = self.SERIES_IDS[series]
            logger.info(f"Fetching FRED series {series_id}")
            
            # Run in thread pool since fredapi is synchronous
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None, 
                self.fred.get_series, 
                series_id, 
                start
            )
            
            df = data.reset_index()
            df.columns = ['date', f'{series}_value']
            df['series_id'] = series_id
            df['series_name'] = series
            
            self.last_updated = datetime.now()
            logger.info(f"Successfully fetched {len(df)} records from FRED {series}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching FRED data: {e}")
            raise
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate FRED data."""
        required_columns = ['date', 'series_id']
        return all(col in data.columns for col in required_columns) and len(data) > 0


class CensusDataSource(DataSource):
    """US Census Bureau data source."""
    
    # American Community Survey variables for housing
    ACS_VARIABLES = {
        'total_population': 'B01001_001E',
        'median_household_income': 'B19013_001E',
        'median_home_value': 'B25077_001E',
        'median_rent': 'B25064_001E',
        'total_housing_units': 'B25001_001E',
        'owner_occupied': 'B25003_002E',
        'renter_occupied': 'B25003_003E',
        'vacancy_rate': 'B25002_003E'
    }
    
    def __init__(self):
        super().__init__("US Census Bureau")
        # Census API doesn't require key for basic usage
        self.census = Census()
    
    async def fetch_data(self, geography: str = 'county', state: str = '06', **kwargs) -> pd.DataFrame:
        """Fetch Census ACS data."""
        try:
            logger.info(f"Fetching Census ACS data for {geography} in state {state}")
            
            # Get variable names and codes
            variables = list(self.ACS_VARIABLES.values())
            
            # Run in thread pool since census library is synchronous
            loop = asyncio.get_event_loop()
            
            if geography == 'county':
                data = await loop.run_in_executor(
                    None,
                    self.census.acs5.state_county,
                    variables,
                    state,
                    Census.ALL
                )
            elif geography == 'tract':
                county = kwargs.get('county', Census.ALL)
                data = await loop.run_in_executor(
                    None,
                    self.census.acs5.state_county_tract,
                    variables,
                    state,
                    county,
                    Census.ALL
                )
            else:
                raise ValueError(f"Unsupported geography: {geography}")
            
            df = pd.DataFrame(data)
            
            # Rename columns to meaningful names
            column_mapping = {v: k for k, v in self.ACS_VARIABLES.items()}
            df = df.rename(columns=column_mapping)
            
            # Convert numeric columns
            numeric_cols = list(self.ACS_VARIABLES.keys())
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Create geographic identifiers
            if 'state' in df.columns and 'county' in df.columns:
                df['fips_code'] = df['state'] + df['county']
                if 'tract' in df.columns:
                    df['fips_code'] = df['fips_code'] + df['tract']
            
            df['fetch_date'] = datetime.now()
            
            self.last_updated = datetime.now()
            logger.info(f"Successfully fetched {len(df)} records from Census ACS")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Census data: {e}")
            raise
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate Census data."""
        required_columns = ['state', 'total_population']
        return all(col in data.columns for col in required_columns) and len(data) > 0


class RentCastDataSource(DataSource):
    """RentCast API for property rental data."""
    
    BASE_URL = "https://api.rentcast.io/v1"
    
    def __init__(self):
        super().__init__("RentCast API")
        self.api_key = settings.rentcast_api_key
    
    async def fetch_data(self, address: str = None, city: str = None, state: str = None, **kwargs) -> pd.DataFrame:
        """Fetch RentCast property data."""
        if not self.api_key:
            logger.warning("RentCast API key not provided, returning empty DataFrame")
            return pd.DataFrame()
        
        try:
            headers = {'X-Api-Key': self.api_key}
            
            async with aiohttp.ClientSession(headers=headers) as session:
                if address:
                    # Get specific property data
                    url = f"{self.BASE_URL}/properties"
                    params = {'address': address}
                else:
                    # Get market data for city/state
                    url = f"{self.BASE_URL}/markets"
                    params = {
                        'city': city or settings.default_city,
                        'state': state or settings.default_state
                    }
                
                logger.info(f"Fetching RentCast data from {url}")
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if isinstance(data, list):
                            df = pd.DataFrame(data)
                        else:
                            df = pd.DataFrame([data])
                        
                        df['fetch_date'] = datetime.now()
                        df['data_source'] = 'rentcast'
                        
                        self.last_updated = datetime.now()
                        logger.info(f"Successfully fetched {len(df)} records from RentCast")
                        return df
                    else:
                        error_msg = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_msg}")
                        
        except Exception as e:
            logger.error(f"Error fetching RentCast data: {e}")
            raise
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate RentCast data."""
        return len(data) > 0 and 'data_source' in data.columns


class OpenStreetMapDataSource(DataSource):
    """OpenStreetMap data source for geographic and amenity data."""
    
    AMENITY_TYPES = [
        'school', 'hospital', 'restaurant', 'cafe', 'bank', 'atm',
        'pharmacy', 'supermarket', 'fuel', 'post_office', 'library',
        'police', 'fire_station', 'bus_station', 'subway_entrance'
    ]
    
    def __init__(self):
        super().__init__("OpenStreetMap")
    
    async def fetch_data(self, location: str = None, amenity_type: str = 'all', **kwargs) -> pd.DataFrame:
        """Fetch OpenStreetMap amenity data."""
        try:
            location = location or f"{settings.default_city}, {settings.default_state}, USA"
            logger.info(f"Fetching OSM data for {location}")
            
            # Run in thread pool since osmnx is synchronous
            loop = asyncio.get_event_loop()
            
            if amenity_type == 'all':
                # Get all amenity types
                all_amenities = []
                for amenity in self.AMENITY_TYPES:
                    try:
                        amenities = await loop.run_in_executor(
                            None,
                            ox.geometries_from_place,
                            location,
                            {'amenity': amenity}
                        )
                        if not amenities.empty:
                            amenities['amenity_type'] = amenity
                            all_amenities.append(amenities)
                    except Exception as e:
                        logger.warning(f"Could not fetch {amenity} amenities: {e}")
                        continue
                
                if all_amenities:
                    df = pd.concat(all_amenities, ignore_index=True)
                else:
                    df = pd.DataFrame()
            else:
                # Get specific amenity type
                amenities = await loop.run_in_executor(
                    None,
                    ox.geometries_from_place,
                    location,
                    {'amenity': amenity_type}
                )
                df = pd.DataFrame(amenities)
                if not df.empty:
                    df['amenity_type'] = amenity_type
            
            if not df.empty:
                # Extract coordinates from geometry
                if 'geometry' in df.columns:
                    df['latitude'] = df.geometry.centroid.y
                    df['longitude'] = df.geometry.centroid.x
                
                # Add metadata
                df['location_query'] = location
                df['fetch_date'] = datetime.now()
                df['data_source'] = 'openstreetmap'
                
                # Select relevant columns
                columns_to_keep = [
                    'amenity_type', 'name', 'latitude', 'longitude',
                    'location_query', 'fetch_date', 'data_source'
                ]
                columns_to_keep = [col for col in columns_to_keep if col in df.columns]
                df = df[columns_to_keep]
            
            self.last_updated = datetime.now()
            logger.info(f"Successfully fetched {len(df)} amenities from OSM")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OSM data: {e}")
            # Return empty DataFrame instead of raising
            return pd.DataFrame()
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate OSM data."""
        if data.empty:
            return True  # Empty is valid for OSM data
        required_columns = ['amenity_type', 'latitude', 'longitude']
        return all(col in data.columns for col in required_columns)