"""Database management utilities for AREIP."""

import asyncio
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import asyncpg
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import json

from ..config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations for AREIP."""
    
    def __init__(self):
        self.engine = create_async_engine(settings.async_postgres_url)
        self.sync_engine = create_engine(settings.postgres_url)
        self.async_session = sessionmaker(
            self.engine, 
            class_=AsyncSession, 
            expire_on_commit=False
        )
    
    async def initialize_database(self):
        """Initialize database tables and schemas."""
        logger.info("Initializing database schema")
        
        schema_sql = """
        -- Raw data tables
        CREATE SCHEMA IF NOT EXISTS raw_data;
        
        -- Processed data tables  
        CREATE SCHEMA IF NOT EXISTS processed_data;
        
        -- Agent state and memory
        CREATE SCHEMA IF NOT EXISTS agent_state;
        
        -- Ingestion metadata table
        CREATE TABLE IF NOT EXISTS raw_data.ingestion_metadata (
            id SERIAL PRIMARY KEY,
            table_name VARCHAR(255) NOT NULL,
            source_name VARCHAR(100) NOT NULL,
            ingestion_timestamp TIMESTAMP DEFAULT NOW(),
            record_count INTEGER,
            data_timestamp TIMESTAMP,
            metadata JSONB
        );
        
        -- Agent execution logs
        CREATE TABLE IF NOT EXISTS agent_state.execution_logs (
            id SERIAL PRIMARY KEY,
            agent_name VARCHAR(100) NOT NULL,
            execution_id VARCHAR(255) NOT NULL,
            start_time TIMESTAMP DEFAULT NOW(),
            end_time TIMESTAMP,
            status VARCHAR(50),
            input_data JSONB,
            output_data JSONB,
            error_message TEXT,
            metadata JSONB
        );
        
        -- Knowledge graph entities
        CREATE TABLE IF NOT EXISTS processed_data.entities (
            id SERIAL PRIMARY KEY,
            entity_id VARCHAR(255) UNIQUE NOT NULL,
            entity_type VARCHAR(100) NOT NULL,
            properties JSONB,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
        
        -- Knowledge graph relationships
        CREATE TABLE IF NOT EXISTS processed_data.relationships (
            id SERIAL PRIMARY KEY,
            relationship_id VARCHAR(255) UNIQUE NOT NULL,
            source_entity_id VARCHAR(255) NOT NULL,
            target_entity_id VARCHAR(255) NOT NULL,
            relationship_type VARCHAR(100) NOT NULL,
            properties JSONB,
            confidence_score FLOAT,
            created_at TIMESTAMP DEFAULT NOW(),
            FOREIGN KEY (source_entity_id) REFERENCES processed_data.entities(entity_id),
            FOREIGN KEY (target_entity_id) REFERENCES processed_data.entities(entity_id)
        );
        
        -- Model training data
        CREATE TABLE IF NOT EXISTS processed_data.model_training_data (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(100) NOT NULL,
            feature_vector JSONB,
            target_value FLOAT,
            data_source VARCHAR(100),
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_ingestion_metadata_table_name ON raw_data.ingestion_metadata(table_name);
        CREATE INDEX IF NOT EXISTS idx_execution_logs_agent_name ON agent_state.execution_logs(agent_name);
        CREATE INDEX IF NOT EXISTS idx_entities_type ON processed_data.entities(entity_type);
        CREATE INDEX IF NOT EXISTS idx_relationships_type ON processed_data.relationships(relationship_type);
        """
        
        async with self.engine.begin() as conn:
            await conn.execute(text(schema_sql))
        
        logger.info("Database schema initialized successfully")
    
    async def store_raw_data(self, table_name: str, data: pd.DataFrame, metadata: Optional[Dict] = None):
        """Store raw data in the database."""
        try:
            full_table_name = f"raw_data.{table_name}"
            
            # Use pandas to_sql with the sync engine for bulk insert
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: data.to_sql(
                    table_name,
                    self.sync_engine,
                    schema='raw_data',
                    if_exists='replace',
                    index=False,
                    method='multi'
                )
            )
            
            # Store metadata
            await self._store_ingestion_metadata(
                table_name=full_table_name,
                source_name=table_name.split('_')[0],
                record_count=len(data),
                metadata=metadata
            )
            
            logger.info(f"Stored {len(data)} records in {full_table_name}")
            
        except Exception as e:
            logger.error(f"Error storing raw data in {table_name}: {e}")
            raise
    
    async def _store_ingestion_metadata(
        self, 
        table_name: str, 
        source_name: str, 
        record_count: int,
        metadata: Optional[Dict] = None
    ):
        """Store ingestion metadata."""
        sql = """
        INSERT INTO raw_data.ingestion_metadata 
        (table_name, source_name, record_count, metadata)
        VALUES ($1, $2, $3, $4)
        """
        
        async with self.engine.begin() as conn:
            await conn.execute(
                text(sql),
                table_name,
                source_name,
                record_count,
                json.dumps(metadata) if metadata else None
            )
    
    async def get_raw_data(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Retrieve raw data from the database."""
        try:
            sql = f"SELECT * FROM raw_data.{table_name}"
            if limit:
                sql += f" LIMIT {limit}"
            
            # Use pandas read_sql with sync engine
            df = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: pd.read_sql(sql, self.sync_engine)
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving data from {table_name}: {e}")
            raise
    
    async def get_table_record_counts(self) -> Dict[str, int]:
        """Get record counts for all raw data tables."""
        sql = """
        SELECT table_name, record_count, ingestion_timestamp
        FROM raw_data.ingestion_metadata
        WHERE ingestion_timestamp = (
            SELECT MAX(ingestion_timestamp) 
            FROM raw_data.ingestion_metadata m2 
            WHERE m2.table_name = raw_data.ingestion_metadata.table_name
        )
        ORDER BY table_name
        """
        
        async with self.engine.begin() as conn:
            result = await conn.execute(text(sql))
            rows = result.fetchall()
            
            return {row[0]: row[1] for row in rows}
    
    async def cleanup_old_raw_data(self, cutoff_date: datetime) -> int:
        """Clean up old raw data based on cutoff date."""
        # Get tables to clean up
        sql = """
        SELECT DISTINCT table_name 
        FROM raw_data.ingestion_metadata 
        WHERE ingestion_timestamp < $1
        """
        
        async with self.engine.begin() as conn:
            result = await conn.execute(text(sql), cutoff_date)
            old_tables = [row[0] for row in result.fetchall()]
            
            deleted_count = 0
            for table_name in old_tables:
                # Delete old metadata
                delete_sql = """
                DELETE FROM raw_data.ingestion_metadata 
                WHERE table_name = $1 AND ingestion_timestamp < $2
                """
                result = await conn.execute(text(delete_sql), table_name, cutoff_date)
                deleted_count += result.rowcount
            
            return deleted_count
    
    async def store_agent_execution(
        self,
        agent_name: str,
        execution_id: str,
        status: str,
        input_data: Optional[Dict] = None,
        output_data: Optional[Dict] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ):
        """Store agent execution information."""
        sql = """
        INSERT INTO agent_state.execution_logs 
        (agent_name, execution_id, status, input_data, output_data, error_message, metadata, start_time, end_time)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """
        
        async with self.engine.begin() as conn:
            await conn.execute(
                text(sql),
                agent_name,
                execution_id,
                status,
                json.dumps(input_data) if input_data else None,
                json.dumps(output_data) if output_data else None,
                error_message,
                json.dumps(metadata) if metadata else None,
                start_time or datetime.now(),
                end_time
            )
    
    async def get_agent_execution_history(
        self, 
        agent_name: Optional[str] = None, 
        limit: int = 100
    ) -> List[Dict]:
        """Get agent execution history."""
        sql = """
        SELECT agent_name, execution_id, status, start_time, end_time, 
               input_data, output_data, error_message, metadata
        FROM agent_state.execution_logs
        """
        
        params = []
        if agent_name:
            sql += " WHERE agent_name = $1"
            params.append(agent_name)
        
        sql += f" ORDER BY start_time DESC LIMIT {limit}"
        
        async with self.engine.begin() as conn:
            result = await conn.execute(text(sql), *params)
            rows = result.fetchall()
            
            return [
                {
                    'agent_name': row[0],
                    'execution_id': row[1], 
                    'status': row[2],
                    'start_time': row[3],
                    'end_time': row[4],
                    'input_data': json.loads(row[5]) if row[5] else None,
                    'output_data': json.loads(row[6]) if row[6] else None,
                    'error_message': row[7],
                    'metadata': json.loads(row[8]) if row[8] else None
                }
                for row in rows
            ]
    
    async def store_entity(
        self,
        entity_id: str,
        entity_type: str,
        properties: Dict
    ):
        """Store or update a knowledge graph entity."""
        sql = """
        INSERT INTO processed_data.entities (entity_id, entity_type, properties, updated_at)
        VALUES ($1, $2, $3, NOW())
        ON CONFLICT (entity_id) 
        DO UPDATE SET 
            entity_type = EXCLUDED.entity_type,
            properties = EXCLUDED.properties,
            updated_at = NOW()
        """
        
        async with self.engine.begin() as conn:
            await conn.execute(
                text(sql),
                entity_id,
                entity_type,
                json.dumps(properties)
            )
    
    async def store_relationship(
        self,
        relationship_id: str,
        source_entity_id: str,
        target_entity_id: str,
        relationship_type: str,
        properties: Optional[Dict] = None,
        confidence_score: Optional[float] = None
    ):
        """Store a knowledge graph relationship."""
        sql = """
        INSERT INTO processed_data.relationships 
        (relationship_id, source_entity_id, target_entity_id, relationship_type, properties, confidence_score)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (relationship_id)
        DO UPDATE SET
            relationship_type = EXCLUDED.relationship_type,
            properties = EXCLUDED.properties,
            confidence_score = EXCLUDED.confidence_score
        """
        
        async with self.engine.begin() as conn:
            await conn.execute(
                text(sql),
                relationship_id,
                source_entity_id,
                target_entity_id,
                relationship_type,
                json.dumps(properties) if properties else None,
                confidence_score
            )
    
    async def close(self):
        """Close database connections."""
        await self.engine.dispose()
        self.sync_engine.dispose()