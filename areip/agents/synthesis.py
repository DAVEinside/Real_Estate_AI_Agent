"""Knowledge Synthesis Agent with Graph-RAG for dynamic relationship mapping."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import json
import hashlib

from neo4j import AsyncGraphDatabase
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
import pandas as pd
import numpy as np

from .base import BaseAgent, AgentTask, AgentResult
from ..config import settings
from ..utils.database import DatabaseManager

logger = logging.getLogger(__name__)


class GraphRAGKnowledgeBase:
    """Graph-RAG knowledge base combining Neo4j graph database with vector search."""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    async def initialize_schema(self):
        """Initialize Neo4j schema for real estate knowledge graph."""
        schema_queries = [
            # Create indexes
            "CREATE INDEX IF NOT EXISTS FOR (p:Property) ON (p.id)",
            "CREATE INDEX IF NOT EXISTS FOR (m:Market) ON (m.id)", 
            "CREATE INDEX IF NOT EXISTS FOR (n:Neighborhood) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (e:EconomicIndicator) ON (e.id)",
            "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.id)",
            
            # Create constraints
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Property) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Market) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Neighborhood) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:EconomicIndicator) REQUIRE e.id IS UNIQUE",
            
            # Create vector index for embeddings
            """
            CALL db.index.vector.createNodeIndex(
                'document_embeddings',
                'Document', 
                'embedding',
                1536,
                'cosine'
            )
            """
        ]
        
        async with self.driver.session() as session:
            for query in schema_queries:
                try:
                    await session.run(query)
                    logger.info(f"Executed schema query: {query[:50]}...")
                except Exception as e:
                    logger.warning(f"Schema query failed (may already exist): {e}")
    
    async def ingest_property_data(self, property_data: pd.DataFrame):
        """Ingest property data into the knowledge graph."""
        async with self.driver.session() as session:
            for _, row in property_data.iterrows():
                try:
                    # Create property node
                    await session.run("""
                        MERGE (p:Property {id: $property_id})
                        SET p.address = $address,
                            p.price = $price,
                            p.bedrooms = $bedrooms,
                            p.bathrooms = $bathrooms,
                            p.square_feet = $square_feet,
                            p.year_built = $year_built,
                            p.property_type = $property_type,
                            p.latitude = $latitude,
                            p.longitude = $longitude,
                            p.updated_at = datetime()
                    """, 
                        property_id=str(row.get('id', f"prop_{hash(str(row))}")),
                        address=str(row.get('address', '')),
                        price=float(row.get('price', 0)) if pd.notna(row.get('price')) else None,
                        bedrooms=int(row.get('bedrooms', 0)) if pd.notna(row.get('bedrooms')) else None,
                        bathrooms=float(row.get('bathrooms', 0)) if pd.notna(row.get('bathrooms')) else None,
                        square_feet=int(row.get('square_feet', 0)) if pd.notna(row.get('square_feet')) else None,
                        year_built=int(row.get('year_built', 0)) if pd.notna(row.get('year_built')) else None,
                        property_type=str(row.get('property_type', '')),
                        latitude=float(row.get('latitude', 0)) if pd.notna(row.get('latitude')) else None,
                        longitude=float(row.get('longitude', 0)) if pd.notna(row.get('longitude')) else None
                    )
                    
                    # Create neighborhood relationship if location data exists
                    if pd.notna(row.get('city')) and pd.notna(row.get('state')):
                        neighborhood_id = f"{row.get('city', '')}_{row.get('state', '')}"
                        await session.run("""
                            MERGE (n:Neighborhood {id: $neighborhood_id})
                            SET n.city = $city,
                                n.state = $state,
                                n.updated_at = datetime()
                            WITH n
                            MATCH (p:Property {id: $property_id})
                            MERGE (p)-[:LOCATED_IN]->(n)
                        """,
                            neighborhood_id=neighborhood_id,
                            city=str(row.get('city', '')),
                            state=str(row.get('state', '')),
                            property_id=str(row.get('id', f"prop_{hash(str(row))}"))
                        )
                        
                except Exception as e:
                    logger.error(f"Error ingesting property data: {e}")
    
    async def ingest_market_data(self, market_data: pd.DataFrame):
        """Ingest market trend data into the knowledge graph."""
        async with self.driver.session() as session:
            for _, row in market_data.iterrows():
                try:
                    market_id = f"market_{row.get('RegionName', 'unknown')}_{row.get('date', 'unknown')}"
                    
                    # Create market data node
                    await session.run("""
                        MERGE (m:Market {id: $market_id})
                        SET m.region_name = $region_name,
                            m.date = date($date),
                            m.value = $value,
                            m.region_type = $region_type,
                            m.state_name = $state_name,
                            m.updated_at = datetime()
                    """,
                        market_id=market_id,
                        region_name=str(row.get('RegionName', '')),
                        date=str(row.get('date', ''))[:10],  # YYYY-MM-DD format
                        value=float(row.get('home_values_value', 0)) if pd.notna(row.get('home_values_value')) else None,
                        region_type=str(row.get('RegionType', '')),
                        state_name=str(row.get('StateName', ''))
                    )
                    
                    # Create neighborhood relationship
                    if pd.notna(row.get('RegionName')):
                        neighborhood_id = f"{row.get('RegionName', '')}_{row.get('StateName', '')}"
                        await session.run("""
                            MERGE (n:Neighborhood {id: $neighborhood_id})
                            SET n.name = $region_name,
                                n.state = $state_name,
                                n.updated_at = datetime()
                            WITH n
                            MATCH (m:Market {id: $market_id})
                            MERGE (m)-[:REPRESENTS]->(n)
                        """,
                            neighborhood_id=neighborhood_id,
                            region_name=str(row.get('RegionName', '')),
                            state_name=str(row.get('StateName', '')),
                            market_id=market_id
                        )
                        
                except Exception as e:
                    logger.error(f"Error ingesting market data: {e}")
    
    async def store_document_with_embedding(self, content: str, metadata: Dict[str, Any]) -> str:
        """Store document with vector embedding in graph."""
        try:
            # Generate embedding
            embedding = await asyncio.get_event_loop().run_in_executor(
                None,
                self.embeddings.embed_query,
                content
            )
            
            doc_id = hashlib.md5(content.encode()).hexdigest()
            
            async with self.driver.session() as session:
                await session.run("""
                    MERGE (d:Document {id: $doc_id})
                    SET d.content = $content,
                        d.metadata = $metadata,
                        d.embedding = $embedding,
                        d.created_at = datetime(),
                        d.updated_at = datetime()
                """,
                    doc_id=doc_id,
                    content=content,
                    metadata=json.dumps(metadata),
                    embedding=embedding
                )
            
            return doc_id
            
        except Exception as e:
            logger.error(f"Error storing document: {e}")
            raise
    
    async def graph_rag_query(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform Graph-RAG query combining vector search with graph traversal."""
        try:
            # Get query embedding
            query_embedding = await asyncio.get_event_loop().run_in_executor(
                None,
                self.embeddings.embed_query,
                query
            )
            
            async with self.driver.session() as session:
                # Vector similarity search
                vector_results = await session.run("""
                    CALL db.index.vector.queryNodes('document_embeddings', $k, $query_embedding)
                    YIELD node, score
                    RETURN node.id as doc_id, node.content as content, 
                           node.metadata as metadata, score
                    ORDER BY score DESC
                """, k=k, query_embedding=query_embedding)
                
                vector_docs = []
                async for record in vector_results:
                    vector_docs.append({
                        "doc_id": record["doc_id"],
                        "content": record["content"],
                        "metadata": json.loads(record["metadata"]) if record["metadata"] else {},
                        "similarity_score": record["score"]
                    })
                
                # Graph traversal for related entities
                graph_results = await session.run("""
                    // Find related properties and markets based on content similarity
                    MATCH (d:Document)
                    WHERE d.id IN $doc_ids
                    OPTIONAL MATCH (d)-[:RELATES_TO]->(p:Property)-[:LOCATED_IN]->(n:Neighborhood)
                    OPTIONAL MATCH (m:Market)-[:REPRESENTS]->(n)
                    RETURN DISTINCT p.id as property_id, p.address as address, p.price as price,
                           n.id as neighborhood_id, n.city as city, n.state as state,
                           m.id as market_id, m.value as market_value, m.date as market_date
                    LIMIT 20
                """, doc_ids=[doc["doc_id"] for doc in vector_docs])
                
                graph_context = []
                async for record in graph_results:
                    if record["property_id"]:
                        graph_context.append({
                            "type": "property",
                            "id": record["property_id"],
                            "address": record["address"],
                            "price": record["price"],
                            "neighborhood": record["city"]
                        })
                    if record["market_id"]:
                        graph_context.append({
                            "type": "market",
                            "id": record["market_id"],
                            "value": record["market_value"],
                            "date": str(record["market_date"]),
                            "neighborhood": record["city"]
                        })
                
                return {
                    "vector_results": vector_docs,
                    "graph_context": graph_context
                }
                
        except Exception as e:
            logger.error(f"Graph-RAG query failed: {e}")
            return {"vector_results": [], "graph_context": []}
    
    async def discover_relationships(self) -> List[Dict[str, Any]]:
        """Discover new relationships in the knowledge graph using graph algorithms."""
        relationships = []
        
        async with self.driver.session() as session:
            try:
                # Find properties with similar characteristics
                similar_properties = await session.run("""
                    MATCH (p1:Property), (p2:Property)
                    WHERE p1.id <> p2.id 
                    AND abs(p1.price - p2.price) < 50000
                    AND abs(p1.square_feet - p2.square_feet) < 200
                    AND p1.bedrooms = p2.bedrooms
                    WITH p1, p2, 
                         abs(p1.price - p2.price) + abs(p1.square_feet - p2.square_feet) as similarity_score
                    ORDER BY similarity_score
                    LIMIT 50
                    RETURN p1.id as prop1_id, p2.id as prop2_id, similarity_score
                """)
                
                async for record in similar_properties:
                    relationships.append({
                        "type": "similar_properties",
                        "source": record["prop1_id"],
                        "target": record["prop2_id"],
                        "score": record["similarity_score"]
                    })
                
                # Find market trends by neighborhood
                neighborhood_trends = await session.run("""
                    MATCH (m:Market)-[:REPRESENTS]->(n:Neighborhood)
                    WHERE m.date >= date('2023-01-01')
                    WITH n, collect(m.value) as values, collect(m.date) as dates
                    WHERE size(values) >= 3
                    RETURN n.id as neighborhood_id, values, dates
                    LIMIT 20
                """)
                
                async for record in neighborhood_trends:
                    values = record["values"]
                    if len(values) >= 2:
                        trend = "increasing" if values[-1] > values[0] else "decreasing"
                        relationships.append({
                            "type": "market_trend",
                            "neighborhood": record["neighborhood_id"],
                            "trend": trend,
                            "value_change": values[-1] - values[0] if values[-1] and values[0] else 0
                        })
                
            except Exception as e:
                logger.error(f"Error discovering relationships: {e}")
        
        return relationships
    
    async def close(self):
        """Close database connection."""
        await self.driver.close()


class KnowledgeSynthesisAgent(BaseAgent):
    """
    Graph-RAG powered agent for dynamic knowledge synthesis and relationship mapping
    in real estate data and market intelligence.
    """
    
    def __init__(self, db_manager: DatabaseManager, **kwargs):
        super().__init__("KnowledgeSynthesisAgent", db_manager, **kwargs)
        
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            api_key=settings.openai_api_key
        )
        
        # Initialize Graph-RAG knowledge base
        self.knowledge_base = GraphRAGKnowledgeBase(
            neo4j_uri=settings.neo4j_uri,
            neo4j_user=settings.neo4j_username,
            neo4j_password=settings.neo4j_password
        )
        
        # Initialize on first use
        self._initialized = False
        
        logger.info("Knowledge Synthesis Agent initialized with Graph-RAG")
    
    async def _ensure_initialized(self):
        """Ensure knowledge base is initialized."""
        if not self._initialized:
            await self.knowledge_base.initialize_schema()
            self._initialized = True
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute knowledge synthesis task."""
        try:
            await self._ensure_initialized()
            
            task_type = task.task_type
            input_data = task.input_data
            
            logger.info(f"Knowledge Synthesis Agent executing: {task_type}")
            
            if task_type == "data_ingestion":
                result = await self._ingest_data(input_data)
            elif task_type == "knowledge_query":
                result = await self._query_knowledge(input_data)
            elif task_type == "relationship_discovery":
                result = await self._discover_relationships(input_data)
            elif task_type == "synthesis_analysis":
                result = await self._synthesize_knowledge(input_data)
            else:
                result = await self._general_synthesis(input_data)
            
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=True,
                output_data=result,
                confidence_score=self._calculate_confidence_score(result)
            )
            
        except Exception as e:
            logger.error(f"Knowledge Synthesis execution failed: {e}")
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=False,
                output_data={},
                error_message=str(e)
            )
    
    async def _ingest_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest data into the knowledge graph."""
        logger.info("Ingesting data into knowledge graph")
        
        ingested_entities = 0
        ingested_relationships = 0
        
        try:
            # Get property data from database
            if "ingest_properties" in input_data and input_data["ingest_properties"]:
                try:
                    # Try to get synthetic property data first, then real data
                    property_data = await self._get_property_data()
                    if not property_data.empty:
                        await self.knowledge_base.ingest_property_data(property_data)
                        ingested_entities += len(property_data)
                        logger.info(f"Ingested {len(property_data)} properties")
                except Exception as e:
                    logger.warning(f"Could not ingest property data: {e}")
            
            # Get market data from database
            if "ingest_market_data" in input_data and input_data["ingest_market_data"]:
                try:
                    market_data = await self.db_manager.get_raw_data("zillow_data_home_values", limit=1000)
                    if not market_data.empty:
                        await self.knowledge_base.ingest_market_data(market_data)
                        ingested_entities += len(market_data)
                        logger.info(f"Ingested {len(market_data)} market records")
                except Exception as e:
                    logger.warning(f"Could not ingest market data: {e}")
            
            # Store analysis documents
            if "documents" in input_data:
                for doc in input_data["documents"]:
                    doc_id = await self.knowledge_base.store_document_with_embedding(
                        content=doc["content"],
                        metadata=doc.get("metadata", {})
                    )
                    ingested_entities += 1
                    logger.info(f"Stored document: {doc_id}")
            
            return {
                "analysis_type": "data_ingestion",
                "entities_ingested": ingested_entities,
                "relationships_created": ingested_relationships,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            raise
    
    async def _query_knowledge(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Query the knowledge graph using Graph-RAG."""
        query = input_data.get("query", "What are the current market trends?")
        k = input_data.get("top_k", 5)
        
        logger.info(f"Querying knowledge graph: {query}")
        
        # Perform Graph-RAG query
        rag_results = await self.knowledge_base.graph_rag_query(query, k)
        
        # Use LLM to synthesize results
        context_docs = rag_results.get("vector_results", [])
        graph_context = rag_results.get("graph_context", [])
        
        synthesis_prompt = f"""
        Based on the following knowledge graph data, provide a comprehensive answer to: {query}
        
        Document Context:
        {json.dumps(context_docs, indent=2)}
        
        Graph Context:
        {json.dumps(graph_context, indent=2)}
        
        Provide a detailed, analytical response that synthesizes information from both the documents and graph relationships.
        """
        
        from langchain.schema import HumanMessage
        response = await self.llm.ainvoke([HumanMessage(content=synthesis_prompt)])
        
        return {
            "analysis_type": "knowledge_query",
            "query": query,
            "synthesis": response.content,
            "source_documents": len(context_docs),
            "graph_entities": len(graph_context),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _discover_relationships(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Discover new relationships in the knowledge graph."""
        logger.info("Discovering relationships in knowledge graph")
        
        relationships = await self.knowledge_base.discover_relationships()
        
        # Analyze relationships using LLM
        analysis_prompt = f"""
        Analyze the following discovered relationships in real estate data:
        
        {json.dumps(relationships[:20], indent=2)}
        
        Provide insights about:
        1. Most significant relationship patterns
        2. Market implications of these relationships
        3. Opportunities or risks identified
        4. Recommendations for further analysis
        """
        
        from langchain.schema import HumanMessage
        response = await self.llm.ainvoke([HumanMessage(content=analysis_prompt)])
        
        return {
            "analysis_type": "relationship_discovery",
            "relationships_found": len(relationships),
            "relationship_types": list(set(r["type"] for r in relationships)),
            "analysis": response.content,
            "sample_relationships": relationships[:10],
            "timestamp": datetime.now().isoformat()
        }
    
    async def _synthesize_knowledge(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize knowledge from multiple sources and analyses."""
        logger.info("Synthesizing knowledge from multiple sources")
        
        # Get recent analysis results
        focus_areas = input_data.get("focus_areas", ["market trends", "property values", "investment opportunities"])
        
        synthesis_results = []
        
        for area in focus_areas:
            # Query knowledge base for this focus area
            rag_results = await self.knowledge_base.graph_rag_query(f"Analysis of {area}", k=3)
            
            if rag_results["vector_results"] or rag_results["graph_context"]:
                synthesis_results.append({
                    "focus_area": area,
                    "document_sources": len(rag_results["vector_results"]),
                    "graph_entities": len(rag_results["graph_context"]),
                    "key_insights": rag_results
                })
        
        # Use LLM to create comprehensive synthesis
        synthesis_prompt = f"""
        Create a comprehensive synthesis of real estate market intelligence based on:
        
        Focus Areas Analysis:
        {json.dumps(synthesis_results, indent=2)}
        
        Provide:
        1. Executive summary of key findings
        2. Cross-domain insights and connections
        3. Strategic recommendations
        4. Risk factors and opportunities
        5. Data quality and confidence assessment
        """
        
        from langchain.schema import HumanMessage
        response = await self.llm.ainvoke([HumanMessage(content=synthesis_prompt)])
        
        return {
            "analysis_type": "knowledge_synthesis",
            "focus_areas": focus_areas,
            "synthesis": response.content,
            "data_sources_analyzed": len(synthesis_results),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _general_synthesis(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform general knowledge synthesis."""
        description = input_data.get("description", "General knowledge synthesis")
        
        # Discover relationships
        relationships = await self.knowledge_base.discover_relationships()
        
        # Query for general market insights
        market_query = await self.knowledge_base.graph_rag_query("market insights and trends", k=5)
        
        synthesis_prompt = f"""
        Provide a general synthesis of real estate market knowledge based on:
        
        Task Description: {description}
        
        Discovered Relationships:
        {json.dumps(relationships[:10], indent=2)}
        
        Market Knowledge:
        {json.dumps(market_query, indent=2)}
        
        Synthesize this information into actionable insights.
        """
        
        from langchain.schema import HumanMessage
        response = await self.llm.ainvoke([HumanMessage(content=synthesis_prompt)])
        
        return {
            "analysis_type": "general_synthesis",
            "synthesis": response.content,
            "relationships_analyzed": len(relationships),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _get_property_data(self) -> pd.DataFrame:
        """Get property data for ingestion."""
        try:
            # Try to get real property data from database
            property_data = await self.db_manager.get_raw_data("rentcast_data", limit=500)
            
            if not property_data.empty:
                return property_data
                
            # Generate synthetic property data if no real data available
            from ..data.ingestion import SyntheticDataGenerator
            synthetic_data = SyntheticDataGenerator.generate_property_listings(count=100)
            return synthetic_data
            
        except Exception as e:
            logger.warning(f"Could not get property data: {e}")
            return pd.DataFrame()
    
    def _calculate_confidence_score(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score based on synthesis results."""
        base_score = 0.6
        
        # Increase confidence based on data sources
        if "entities_ingested" in result and result["entities_ingested"] > 0:
            base_score += 0.2
        
        if "synthesis" in result and len(result["synthesis"]) > 200:
            base_score += 0.2
        
        return min(base_score, 1.0)
    
    async def close(self):
        """Close knowledge base connections."""
        await self.knowledge_base.close()