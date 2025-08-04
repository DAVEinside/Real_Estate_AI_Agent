"""Data ingestion pipeline for coordinating multiple real estate data sources."""

import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

from .sources import (
    ZillowDataSource,
    FREDDataSource, 
    CensusDataSource,
    RentCastDataSource,
    OpenStreetMapDataSource
)
from ..utils.database import DatabaseManager
from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of a data ingestion operation."""
    source_name: str
    success: bool
    records_processed: int
    error_message: Optional[str] = None
    execution_time_seconds: float = 0.0
    data_timestamp: Optional[datetime] = None


class DataIngestionPipeline:
    """Coordinates ingestion from multiple real estate data sources."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.sources = {
            'zillow': ZillowDataSource(),
            'fred': FREDDataSource(),
            'census': CensusDataSource(),
            'rentcast': RentCastDataSource(),
            'osm': OpenStreetMapDataSource()
        }
        
    async def run_full_ingestion(self) -> List[IngestionResult]:
        """Run complete data ingestion from all sources."""
        logger.info("Starting full data ingestion pipeline")
        
        results = []
        
        # Ingest Zillow data for multiple datasets
        zillow_datasets = ['home_values', 'rentals', 'inventory', 'sales']
        for dataset in zillow_datasets:
            result = await self._ingest_source_data(
                'zillow', 
                dataset=dataset,
                table_suffix=f"_{dataset}"
            )
            results.append(result)
        
        # Ingest FRED economic indicators
        fred_series = ['zhvi_national', 'mortgage_rates', 'unemployment', 'housing_starts']
        for series in fred_series:
            result = await self._ingest_source_data(
                'fred',
                series=series,
                table_suffix=f"_{series}"
            )
            results.append(result)
        
        # Ingest Census data for major states
        major_states = ['06', '36', '12', '48', '17']  # CA, NY, FL, TX, IL
        for state in major_states:
            result = await self._ingest_source_data(
                'census',
                geography='county',
                state=state,
                table_suffix=f"_state_{state}"
            )
            results.append(result)
        
        # Ingest RentCast market data for major cities
        major_cities = [
            ('San Francisco', 'CA'),
            ('New York', 'NY'),
            ('Los Angeles', 'CA'),
            ('Chicago', 'IL'),
            ('Miami', 'FL')
        ]
        for city, state in major_cities:
            result = await self._ingest_source_data(
                'rentcast',
                city=city,
                state=state,
                table_suffix=f"_{city.lower().replace(' ', '_')}"
            )
            results.append(result)
        
        # Ingest OSM amenity data for major cities
        for city, state in major_cities:
            location = f"{city}, {state}, USA"
            result = await self._ingest_source_data(
                'osm',
                location=location,
                amenity_type='all',
                table_suffix=f"_{city.lower().replace(' ', '_')}"
            )
            results.append(result)
        
        # Log summary
        successful = sum(1 for r in results if r.success)
        total_records = sum(r.records_processed for r in results if r.success)
        
        logger.info(f"Ingestion complete: {successful}/{len(results)} sources successful, {total_records} total records")
        
        return results
    
    async def _ingest_source_data(
        self, 
        source_name: str, 
        table_suffix: str = "",
        **kwargs
    ) -> IngestionResult:
        """Ingest data from a single source."""
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting ingestion for {source_name}{table_suffix}")
            
            source = self.sources[source_name]
            data = await source.fetch_data(**kwargs)
            
            if data.empty:
                logger.warning(f"No data returned from {source_name}")
                return IngestionResult(
                    source_name=f"{source_name}{table_suffix}",
                    success=True,
                    records_processed=0,
                    execution_time_seconds=(datetime.now() - start_time).total_seconds()
                )
            
            # Validate data
            if not source.validate_data(data):
                raise Exception("Data validation failed")
            
            # Store in database
            table_name = f"{source_name}_data{table_suffix}"
            await self.db_manager.store_raw_data(table_name, data)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Successfully ingested {len(data)} records for {source_name}{table_suffix} in {execution_time:.2f}s")
            
            return IngestionResult(
                source_name=f"{source_name}{table_suffix}",
                success=True,
                records_processed=len(data),
                execution_time_seconds=execution_time,
                data_timestamp=source.last_updated
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            
            logger.error(f"Failed to ingest {source_name}{table_suffix}: {error_msg}")
            
            return IngestionResult(
                source_name=f"{source_name}{table_suffix}",
                success=False,
                records_processed=0,
                error_message=error_msg,
                execution_time_seconds=execution_time
            )
    
    async def run_incremental_ingestion(self, hours_back: int = 24) -> List[IngestionResult]:
        """Run incremental data ingestion for sources that support it."""
        logger.info(f"Starting incremental ingestion for last {hours_back} hours")
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        results = []
        
        # Check which sources need updates
        sources_to_update = []
        for name, source in self.sources.items():
            if source.last_updated is None or source.last_updated < cutoff_time:
                sources_to_update.append(name)
        
        logger.info(f"Sources requiring update: {sources_to_update}")
        
        # Update FRED data (daily updates)
        if 'fred' in sources_to_update:
            fred_series = ['zhvi_national', 'mortgage_rates', 'unemployment']
            for series in fred_series:
                result = await self._ingest_source_data(
                    'fred',
                    series=series,
                    start=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                    table_suffix=f"_{series}_incremental"
                )
                results.append(result)
        
        # Update RentCast data (if API limits allow)
        if 'rentcast' in sources_to_update:
            result = await self._ingest_source_data(
                'rentcast',
                city=settings.default_city,
                state=settings.default_state,
                table_suffix="_incremental"
            )
            results.append(result)
        
        return results
    
    async def get_ingestion_status(self) -> Dict[str, Any]:
        """Get status of all data sources."""
        status = {
            'sources': {},
            'last_full_ingestion': None,
            'total_records': 0
        }
        
        for name, source in self.sources.items():
            status['sources'][name] = {
                'last_updated': source.last_updated.isoformat() if source.last_updated else None,
                'status': 'healthy' if source.last_updated and 
                         source.last_updated > datetime.now() - timedelta(days=7) else 'stale'
            }
        
        # Get record counts from database
        try:
            table_counts = await self.db_manager.get_table_record_counts()
            status['table_counts'] = table_counts
            status['total_records'] = sum(table_counts.values())
        except Exception as e:
            logger.error(f"Error getting table counts: {e}")
            status['table_counts'] = {}
        
        return status
    
    async def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old raw data to manage storage."""
        logger.info(f"Cleaning up data older than {days_to_keep} days")
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        try:
            deleted_count = await self.db_manager.cleanup_old_raw_data(cutoff_date)
            logger.info(f"Cleaned up {deleted_count} old records")
            return deleted_count
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise


class SyntheticDataGenerator:
    """Generate synthetic real estate data for testing and development."""
    
    @staticmethod
    def generate_property_listings(count: int = 1000) -> pd.DataFrame:
        """Generate synthetic property listings."""
        import numpy as np
        from faker import Faker
        
        fake = Faker()
        
        data = []
        for _ in range(count):
            property_data = {
                'id': fake.uuid4(),
                'address': fake.address(),
                'city': fake.city(),
                'state': fake.state_abbr(),
                'zip_code': fake.zipcode(),
                'price': np.random.randint(200000, 2000000),
                'bedrooms': np.random.randint(1, 6),
                'bathrooms': np.random.randint(1, 4),
                'square_feet': np.random.randint(800, 4000),
                'lot_size': np.random.randint(5000, 20000),
                'year_built': np.random.randint(1950, 2024),
                'property_type': np.random.choice(['Single Family', 'Condo', 'Townhouse']),
                'listing_date': fake.date_between(start_date='-1y', end_date='today'),
                'days_on_market': np.random.randint(1, 180),
                'latitude': fake.latitude(),
                'longitude': fake.longitude()
            }
            data.append(property_data)
        
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_market_trends(regions: List[str], months: int = 24) -> pd.DataFrame:
        """Generate synthetic market trend data."""
        import numpy as np
        from datetime import datetime, timedelta
        
        data = []
        base_date = datetime.now() - timedelta(days=30 * months)
        
        for region in regions:
            base_price = np.random.randint(400000, 1200000)
            
            for month in range(months):
                date = base_date + timedelta(days=30 * month)
                # Add some realistic trend and noise
                trend = 1 + (month * 0.002)  # 0.2% monthly growth
                noise = np.random.normal(1, 0.05)  # 5% noise
                price = base_price * trend * noise
                
                trend_data = {
                    'region': region,
                    'date': date,
                    'median_price': price,
                    'inventory_count': np.random.randint(50, 500),
                    'days_on_market': np.random.randint(20, 60) + np.random.normal(0, 10),
                    'price_per_sqft': price / np.random.randint(1200, 2000)
                }
                data.append(trend_data)
        
        return pd.DataFrame(data)