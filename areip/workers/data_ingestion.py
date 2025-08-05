"""Data ingestion Celery tasks."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

from celery import Task
from ..celery_app import celery_app
from ..data.ingestion import DataIngestionPipeline
from ..data.sources import DataSourceOrchestrator
from ..utils.database import DatabaseManager
from ..utils.cache import CacheManager

logger = logging.getLogger(__name__)


class AsyncTask(Task):
    """Base task class for async operations."""
    
    def __call__(self, *args, **kwargs):
        """Execute async task in event loop."""
        return asyncio.run(self.run(*args, **kwargs))
    
    async def run(self, *args, **kwargs):
        """Override this method in subclasses."""
        raise NotImplementedError


@celery_app.task(bind=True, base=AsyncTask)
async def ingest_zillow_data(self) -> Dict[str, Any]:
    """Ingest Zillow research data."""
    
    logger.info("Starting Zillow data ingestion task")
    start_time = datetime.now()
    
    try:
        # Initialize components
        db_manager = DatabaseManager()
        await db_manager.initialize_database()
        
        cache_manager = CacheManager()
        orchestrator = DataSourceOrchestrator(cache_manager=cache_manager)
        pipeline = DataIngestionPipeline(db_manager)
        
        async with orchestrator:
            # Fetch Zillow research data
            zillow_source = await orchestrator.get_source('zillow')
            
            datasets = ['home_values', 'rental_data', 'inventory', 'sales']
            total_records = 0
            
            for dataset in datasets:
                try:
                    data = await zillow_source.fetch_data(data_type=dataset)
                    if not data.empty:
                        # Store in database
                        await db_manager.store_raw_data(
                            f"zillow_{dataset}",
                            data,
                            metadata={'source': 'zillow_research', 'dataset': dataset}
                        )
                        total_records += len(data)
                        logger.info(f"Ingested {len(data)} records for {dataset}")
                    
                except Exception as e:
                    logger.error(f"Error ingesting {dataset}: {e}")
            
            # Update ingestion log
            await pipeline.log_ingestion_run(
                source_id="zillow_research",
                status="completed",
                records_processed=total_records,
                start_time=start_time,
                end_time=datetime.now()
            )
            
            return {
                'status': 'success',
                'total_records': total_records,
                'datasets_processed': len(datasets),
                'duration_minutes': (datetime.now() - start_time).total_seconds() / 60
            }
    
    except Exception as e:
        logger.error(f"Zillow data ingestion failed: {e}")
        
        # Log failure
        try:
            await pipeline.log_ingestion_run(
                source_id="zillow_research",
                status="failed",
                error_message=str(e),
                start_time=start_time,
                end_time=datetime.now()
            )
        except:
            pass
        
        return {
            'status': 'failed',
            'error': str(e),
            'duration_minutes': (datetime.now() - start_time).total_seconds() / 60
        }


@celery_app.task(bind=True, base=AsyncTask)
async def ingest_fred_data(self) -> Dict[str, Any]:
    """Ingest FRED economic data."""
    
    logger.info("Starting FRED data ingestion task")
    start_time = datetime.now()
    
    try:
        # Initialize components
        db_manager = DatabaseManager()
        await db_manager.initialize_database()
        
        cache_manager = CacheManager()
        orchestrator = DataSourceOrchestrator(cache_manager=cache_manager)
        pipeline = DataIngestionPipeline(db_manager)
        
        async with orchestrator:
            # Fetch FRED data
            fred_source = await orchestrator.get_source('fred')
            
            # Get housing indicators
            indicators_data = await fred_source.fetch_housing_indicators()
            total_records = 0
            
            for indicator_name, data in indicators_data.items():
                if not data.empty:
                    # Store in database
                    await db_manager.store_raw_data(
                        f"fred_{indicator_name}",
                        data,
                        metadata={'source': 'fred', 'indicator': indicator_name}
                    )
                    total_records += len(data)
                    logger.info(f"Ingested {len(data)} records for {indicator_name}")
            
            # Update ingestion log
            await pipeline.log_ingestion_run(
                source_id="fred",
                status="completed",
                records_processed=total_records,
                start_time=start_time,
                end_time=datetime.now()
            )
            
            return {
                'status': 'success',
                'total_records': total_records,
                'indicators_processed': len(indicators_data),
                'duration_minutes': (datetime.now() - start_time).total_seconds() / 60
            }
    
    except Exception as e:
        logger.error(f"FRED data ingestion failed: {e}")
        
        return {
            'status': 'failed',
            'error': str(e),
            'duration_minutes': (datetime.now() - start_time).total_seconds() / 60
        }


@celery_app.task(bind=True, base=AsyncTask)
async def ingest_census_data(self) -> Dict[str, Any]:
    """Ingest Census demographic data."""
    
    logger.info("Starting Census data ingestion task")
    start_time = datetime.now()
    
    try:
        # Initialize components
        db_manager = DatabaseManager()
        await db_manager.initialize_database()
        
        cache_manager = CacheManager()
        orchestrator = DataSourceOrchestrator(cache_manager=cache_manager)
        pipeline = DataIngestionPipeline(db_manager)
        
        async with orchestrator:
            # Fetch Census data
            census_source = await orchestrator.get_source('census')
            
            # Get ACS data for major states
            states = ['06', '36', '12', '48', '17']  # CA, NY, FL, TX, IL
            total_records = 0
            
            for state in states:
                try:
                    data = await census_source.fetch_data(state=state)
                    if not data.empty:
                        # Store in database
                        await db_manager.store_raw_data(
                            f"census_state_{state}",
                            data,
                            metadata={'source': 'census', 'state': state}
                        )
                        total_records += len(data)
                        logger.info(f"Ingested {len(data)} records for state {state}")
                
                except Exception as e:
                    logger.error(f"Error ingesting Census data for state {state}: {e}")
            
            # Update ingestion log
            await pipeline.log_ingestion_run(
                source_id="census",
                status="completed",
                records_processed=total_records,
                start_time=start_time,
                end_time=datetime.now()
            )
            
            return {
                'status': 'success',
                'total_records': total_records,
                'states_processed': len(states),
                'duration_minutes': (datetime.now() - start_time).total_seconds() / 60
            }
    
    except Exception as e:
        logger.error(f"Census data ingestion failed: {e}")
        
        return {
            'status': 'failed',
            'error': str(e),
            'duration_minutes': (datetime.now() - start_time).total_seconds() / 60
        }


@celery_app.task(bind=True, base=AsyncTask)
async def ingest_rental_data(self) -> Dict[str, Any]:
    """Ingest rental market data."""
    
    logger.info("Starting rental data ingestion task")
    start_time = datetime.now()
    
    try:
        # Initialize components
        db_manager = DatabaseManager()
        await db_manager.initialize_database()
        
        cache_manager = CacheManager()
        orchestrator = DataSourceOrchestrator(cache_manager=cache_manager)
        pipeline = DataIngestionPipeline(db_manager)
        
        async with orchestrator:
            # Fetch rental data
            rental_source = await orchestrator.get_source('rentcast')
            
            # Major metro areas
            metros = [
                ('San Francisco', 'CA'),
                ('Los Angeles', 'CA'),
                ('New York', 'NY'),
                ('Chicago', 'IL'),
                ('Miami', 'FL')
            ]
            
            total_records = 0
            
            for city, state in metros:
                try:
                    data = await rental_source.fetch_data(city=city, state=state, limit=200)
                    if not data.empty:
                        # Store in database
                        await db_manager.store_raw_data(
                            f"rentals_{city.lower().replace(' ', '_')}",
                            data,
                            metadata={'source': 'rental_market', 'city': city, 'state': state}
                        )
                        total_records += len(data)
                        logger.info(f"Ingested {len(data)} rental records for {city}, {state}")
                
                except Exception as e:
                    logger.error(f"Error ingesting rental data for {city}, {state}: {e}")
            
            # Update ingestion log
            await pipeline.log_ingestion_run(
                source_id="rental_market",
                status="completed",
                records_processed=total_records,
                start_time=start_time,
                end_time=datetime.now()
            )
            
            return {
                'status': 'success',
                'total_records': total_records,
                'metros_processed': len(metros),
                'duration_minutes': (datetime.now() - start_time).total_seconds() / 60
            }
    
    except Exception as e:
        logger.error(f"Rental data ingestion failed: {e}")
        
        return {
            'status': 'failed',
            'error': str(e),
            'duration_minutes': (datetime.now() - start_time).total_seconds() / 60
        }


@celery_app.task(bind=True, base=AsyncTask)
async def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, Any]:
    """Clean up old data from the database."""
    
    logger.info(f"Starting data cleanup task - keeping last {days_to_keep} days")
    start_time = datetime.now()
    
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize_database()
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Cleanup queries
        cleanup_queries = [
            ("raw_data.property_listings", "listing_date", cutoff_date),
            ("raw_data.market_trends", "date", cutoff_date),
            ("raw_data.economic_indicators", "date", cutoff_date),
            ("agent_state.agent_executions", "created_at", cutoff_date),
        ]
        
        total_deleted = 0
        
        for table, date_column, cutoff in cleanup_queries:
            try:
                query = f"DELETE FROM {table} WHERE {date_column} < $1"
                result = await db_manager.execute_query(query, cutoff)
                deleted_count = result if isinstance(result, int) else 0
                total_deleted += deleted_count
                
                logger.info(f"Deleted {deleted_count} old records from {table}")
                
            except Exception as e:
                logger.error(f"Error cleaning up {table}: {e}")
        
        # Vacuum analyze for performance
        await db_manager.execute_query("VACUUM ANALYZE")
        
        return {
            'status': 'success',
            'total_deleted': total_deleted,
            'cutoff_date': cutoff_date.isoformat(),
            'duration_minutes': (datetime.now() - start_time).total_seconds() / 60
        }
    
    except Exception as e:
        logger.error(f"Data cleanup failed: {e}")
        
        return {
            'status': 'failed',
            'error': str(e),
            'duration_minutes': (datetime.now() - start_time).total_seconds() / 60
        }