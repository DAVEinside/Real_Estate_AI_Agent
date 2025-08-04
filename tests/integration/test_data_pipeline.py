"""Integration tests for data pipeline."""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd

from areip.data.ingestion import DataIngestionPipeline
from areip.data.sources import ZillowDataSource, FREDDataSource, CensusDataSource
from areip.utils.database import DatabaseManager


@pytest.fixture
def mock_db_manager():
    """Mock database manager for testing."""
    db_manager = Mock(spec=DatabaseManager)
    db_manager.store_raw_data = AsyncMock()
    db_manager.get_table_record_counts = AsyncMock(return_value={"test_table": 100})
    db_manager.cleanup_old_raw_data = AsyncMock(return_value=50)
    return db_manager


@pytest.fixture
def ingestion_pipeline(mock_db_manager):
    """Create ingestion pipeline for testing."""
    return DataIngestionPipeline(mock_db_manager)


@pytest.mark.asyncio
class TestDataSources:
    """Test individual data sources."""
    
    async def test_zillow_data_source(self):
        """Test Zillow data source."""
        source = ZillowDataSource()
        
        # Mock HTTP response
        mock_csv_data = """RegionID,SizeRank,RegionName,RegionType,StateName,2023-01-31,2023-02-28
        102001,1,United States,Country,,"1234567.89","1245678.90"
        394913,2,New York NY,Msa,NY,"2345678.90","2356789.01"
        """
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value=mock_csv_data)
            mock_get.return_value.__aenter__.return_value = mock_response
            
            data = await source.fetch_data(dataset='home_values')
            
            assert not data.empty
            assert 'RegionName' in data.columns
            assert source.validate_data(data) is True
    
    async def test_fred_data_source(self):
        """Test FRED data source."""
        with patch('areip.config.settings.fred_api_key', 'test_key'):
            source = FREDDataSource()
            
            # Mock FRED API response
            mock_series_data = pd.Series([100.0, 101.0, 102.0], 
                                       index=pd.date_range('2023-01-01', periods=3, freq='M'))
            
            with patch.object(source.fred, 'get_series', return_value=mock_series_data):
                data = await source.fetch_data(series='zhvi_national')
                
                assert not data.empty
                assert 'date' in data.columns
                assert source.validate_data(data) is True
    
    async def test_census_data_source(self):
        """Test Census data source."""
        source = CensusDataSource()
        
        mock_census_data = [
            {'state': '06', 'county': '001', 'B01001_001E': 1000000, 'B19013_001E': 75000},
            {'state': '06', 'county': '002', 'B01001_001E': 500000, 'B19013_001E': 65000}
        ]
        
        with patch.object(source.census.acs5, 'state_county', return_value=mock_census_data):
            data = await source.fetch_data(geography='county', state='06')
            
            assert not data.empty
            assert 'total_population' in data.columns
            assert source.validate_data(data) is True


@pytest.mark.asyncio
class TestDataIngestionPipeline:
    """Test data ingestion pipeline."""
    
    async def test_pipeline_initialization(self, ingestion_pipeline):
        """Test pipeline initialization."""
        assert ingestion_pipeline.db_manager is not None
        assert 'zillow' in ingestion_pipeline.sources
        assert 'fred' in ingestion_pipeline.sources
        assert 'census' in ingestion_pipeline.sources
    
    async def test_single_source_ingestion(self, ingestion_pipeline):
        """Test ingestion from a single source."""
        # Mock successful data fetch
        mock_data = pd.DataFrame({
            'RegionName': ['Test Region'],
            'date': ['2023-01-01'],
            'home_values_value': [500000]
        })
        
        with patch.object(ingestion_pipeline.sources['zillow'], 'fetch_data', return_value=mock_data):
            with patch.object(ingestion_pipeline.sources['zillow'], 'validate_data', return_value=True):
                result = await ingestion_pipeline._ingest_source_data('zillow', dataset='home_values')
                
                assert result.success is True
                assert result.records_processed == 1
                assert result.source_name == 'zillow'
    
    async def test_failed_source_ingestion(self, ingestion_pipeline):
        """Test handling of failed ingestion."""
        with patch.object(ingestion_pipeline.sources['zillow'], 'fetch_data', side_effect=Exception("API Error")):
            result = await ingestion_pipeline._ingest_source_data('zillow', dataset='home_values')
            
            assert result.success is False
            assert result.error_message == "API Error"
            assert result.records_processed == 0
    
    async def test_incremental_ingestion(self, ingestion_pipeline):
        """Test incremental ingestion logic."""
        # Mock that sources need updates
        for source in ingestion_pipeline.sources.values():
            source.last_updated = None
        
        # Mock successful ingestion
        mock_data = pd.DataFrame({'test': [1, 2, 3]})
        
        with patch.object(ingestion_pipeline, '_ingest_source_data') as mock_ingest:
            mock_ingest.return_value = Mock(success=True, records_processed=3)
            
            results = await ingestion_pipeline.run_incremental_ingestion(hours_back=24)
            
            assert len(results) > 0
            assert all(result.success for result in results if hasattr(result, 'success'))
    
    async def test_ingestion_status(self, ingestion_pipeline):
        """Test ingestion status reporting."""
        status = await ingestion_pipeline.get_ingestion_status()
        
        assert 'sources' in status
        assert 'total_records' in status
        assert status['total_records'] == 100  # From mock
    
    async def test_data_cleanup(self, ingestion_pipeline):
        """Test old data cleanup."""
        deleted_count = await ingestion_pipeline.cleanup_old_data(days_to_keep=90)
        
        assert deleted_count == 50  # From mock
        ingestion_pipeline.db_manager.cleanup_old_raw_data.assert_called_once()


@pytest.mark.asyncio
class TestDataValidation:
    """Test data validation and quality checks."""
    
    async def test_data_quality_validation(self):
        """Test data quality validation."""
        # Create test data with quality issues
        test_data = pd.DataFrame({
            'price': [100000, 200000, None, 300000, 999999999],  # Missing value and outlier
            'region': ['A', 'B', 'C', 'D', 'E'],
            'date': pd.date_range('2023-01-01', periods=5)
        })
        
        from areip.agents.validation import StatisticalValidator
        
        # Test outlier detection
        outliers = StatisticalValidator.detect_outliers(test_data, ['price'])
        
        assert 'price' in outliers
        assert outliers['price']['iqr_outliers']['count'] > 0
    
    async def test_trend_validation(self):
        """Test trend validation functionality."""
        # Create test data with clear trend
        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        prices = [100000 + i * 5000 for i in range(12)]  # Increasing trend
        
        test_data = pd.DataFrame({
            'date': dates,
            'price': prices
        })
        
        from areip.agents.validation import StatisticalValidator
        
        validation_result = StatisticalValidator.validate_price_trends(test_data, 'price', 'date')
        
        assert 'trend_test' in validation_result
        assert validation_result['trend_test']['trend_direction'] == 'increasing'
        assert validation_result['trend_test']['significant'] is True


@pytest.mark.asyncio 
class TestEndToEndDataFlow:
    """Test complete end-to-end data flow."""
    
    async def test_full_pipeline_execution(self, mock_db_manager):
        """Test complete pipeline execution."""
        pipeline = DataIngestionPipeline(mock_db_manager)
        
        # Mock all data sources to return valid data
        mock_data = pd.DataFrame({
            'test_column': [1, 2, 3],
            'date': pd.date_range('2023-01-01', periods=3)
        })
        
        for source_name, source in pipeline.sources.items():
            source.fetch_data = AsyncMock(return_value=mock_data)
            source.validate_data = Mock(return_value=True)
            source.last_updated = None
        
        # Run incremental ingestion
        results = await pipeline.run_incremental_ingestion()
        
        # Verify results
        assert len(results) > 0
        successful_results = [r for r in results if r.success]
        assert len(successful_results) > 0
        
        # Verify database interactions
        assert mock_db_manager.store_raw_data.call_count > 0


if __name__ == "__main__":
    pytest.main([__file__])