"""Market Discovery Agent using CrewAI for autonomous data mining and pattern recognition."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import json

from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .base import BaseAgent, AgentTask, AgentResult
from ..config import settings
from ..utils.database import DatabaseManager

logger = logging.getLogger(__name__)


class PropertyAnalysisTool(BaseTool):
    """Tool for analyzing property market data."""
    
    name: str = "property_analysis_tool"
    description: str = "Analyze property market data to identify trends and anomalies"
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__()
        self.db_manager = db_manager
    
    def _run(self, query: str) -> str:
        """Execute property analysis."""
        try:
            # This would be called synchronously by CrewAI
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new event loop for synchronous execution
                import threading
                result = None
                exception = None
                
                def run_async():
                    nonlocal result, exception
                    try:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        result = new_loop.run_until_complete(self._analyze_properties(query))
                        new_loop.close()
                    except Exception as e:
                        exception = e
                
                thread = threading.Thread(target=run_async)
                thread.start()
                thread.join()
                
                if exception:
                    raise exception
                return result
            else:
                return loop.run_until_complete(self._analyze_properties(query))
                
        except Exception as e:
            logger.error(f"Property analysis tool error: {e}")
            return f"Error: {str(e)}"
    
    async def _analyze_properties(self, query: str) -> str:
        """Analyze properties based on query."""
        try:
            # Get Zillow data
            zillow_data = await self.db_manager.get_raw_data("zillow_data_home_values", limit=1000)
            
            if zillow_data.empty:
                return "No property data available for analysis"
            
            # Perform basic analysis
            analysis = {
                "total_regions": len(zillow_data['RegionName'].unique()) if 'RegionName' in zillow_data.columns else 0,
                "date_range": {
                    "start": str(zillow_data['date'].min()) if 'date' in zillow_data.columns else "unknown",
                    "end": str(zillow_data['date'].max()) if 'date' in zillow_data.columns else "unknown"
                },
                "price_stats": {}
            }
            
            # Price statistics if value column exists
            value_cols = [col for col in zillow_data.columns if 'value' in col.lower()]
            if value_cols:
                value_col = value_cols[0]
                analysis["price_stats"] = {
                    "median": float(zillow_data[value_col].median()),
                    "mean": float(zillow_data[value_col].mean()),
                    "std": float(zillow_data[value_col].std()),
                    "min": float(zillow_data[value_col].min()),
                    "max": float(zillow_data[value_col].max())
                }
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            logger.error(f"Property analysis failed: {e}")
            return f"Analysis failed: {str(e)}"


class MarketTrendTool(BaseTool):
    """Tool for analyzing market trends and patterns."""
    
    name: str = "market_trend_tool" 
    description: str = "Identify trends and patterns in real estate market data"
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__()
        self.db_manager = db_manager
    
    def _run(self, time_period: str = "6m") -> str:
        """Analyze market trends."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Handle async in sync context
                import threading
                result = None
                exception = None
                
                def run_async():
                    nonlocal result, exception
                    try:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        result = new_loop.run_until_complete(self._analyze_trends(time_period))
                        new_loop.close()
                    except Exception as e:
                        exception = e
                
                thread = threading.Thread(target=run_async)
                thread.start()
                thread.join()
                
                if exception:
                    raise exception
                return result
            else:
                return loop.run_until_complete(self._analyze_trends(time_period))
                
        except Exception as e:
            logger.error(f"Market trend tool error: {e}")
            return f"Error: {str(e)}"
    
    async def _analyze_trends(self, time_period: str) -> str:
        """Analyze market trends over specified period."""
        try:
            # Get recent market data
            zillow_data = await self.db_manager.get_raw_data("zillow_data_home_values", limit=5000)
            
            if zillow_data.empty:
                return "No market data available for trend analysis"
            
            # Parse time period
            months = 6  # default
            if time_period.endswith('m'):
                months = int(time_period[:-1])
            elif time_period.endswith('y'):
                months = int(time_period[:-1]) * 12
            
            # Filter to recent data if date column exists
            if 'date' in zillow_data.columns:
                zillow_data['date'] = pd.to_datetime(zillow_data['date'])
                cutoff_date = datetime.now() - timedelta(days=months * 30)
                recent_data = zillow_data[zillow_data['date'] >= cutoff_date]
            else:
                recent_data = zillow_data
            
            trends = {
                "time_period": time_period,
                "data_points": len(recent_data),
                "regions_analyzed": len(recent_data['RegionName'].unique()) if 'RegionName' in recent_data.columns else 0,
                "trends": []
            }
            
            # Analyze trends by region if possible
            value_cols = [col for col in recent_data.columns if 'value' in col.lower()]
            if value_cols and 'RegionName' in recent_data.columns and 'date' in recent_data.columns:
                value_col = value_cols[0]
                
                # Get top 10 regions by data availability
                top_regions = recent_data['RegionName'].value_counts().head(10).index
                
                for region in top_regions:
                    region_data = recent_data[recent_data['RegionName'] == region].sort_values('date')
                    
                    if len(region_data) >= 2:
                        first_value = region_data[value_col].iloc[0]
                        last_value = region_data[value_col].iloc[-1]
                        
                        if first_value > 0:
                            change_pct = ((last_value - first_value) / first_value) * 100
                            
                            trends["trends"].append({
                                "region": region,
                                "price_change_pct": round(change_pct, 2),
                                "trend": "increasing" if change_pct > 5 else "decreasing" if change_pct < -5 else "stable",
                                "current_value": round(last_value, 2) if pd.notna(last_value) else None
                            })
            
            return json.dumps(trends, indent=2)
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return f"Trend analysis failed: {str(e)}"


class AnomalyDetectionTool(BaseTool):
    """Tool for detecting anomalies in real estate data."""
    
    name: str = "anomaly_detection_tool"
    description: str = "Detect anomalies and outliers in real estate market data using machine learning"
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__()
        self.db_manager = db_manager
    
    def _run(self, sensitivity: str = "medium") -> str:
        """Detect anomalies in market data."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import threading
                result = None
                exception = None
                
                def run_async():
                    nonlocal result, exception
                    try:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        result = new_loop.run_until_complete(self._detect_anomalies(sensitivity))
                        new_loop.close()
                    except Exception as e:
                        exception = e
                
                thread = threading.Thread(target=run_async)
                thread.start()
                thread.join()
                
                if exception:
                    raise exception
                return result
            else:
                return loop.run_until_complete(self._detect_anomalies(sensitivity))
                
        except Exception as e:
            logger.error(f"Anomaly detection tool error: {e}")
            return f"Error: {str(e)}"
    
    async def _detect_anomalies(self, sensitivity: str) -> str:
        """Detect anomalies using DBSCAN clustering."""
        try:
            # Get market data
            zillow_data = await self.db_manager.get_raw_data("zillow_data_home_values", limit=2000)
            
            if zillow_data.empty:
                return "No data available for anomaly detection"
            
            # Prepare features for anomaly detection
            value_cols = [col for col in zillow_data.columns if 'value' in col.lower()]
            if not value_cols:
                return "No value columns found for anomaly detection"
            
            value_col = value_cols[0]
            
            # Filter out null values
            clean_data = zillow_data.dropna(subset=[value_col])
            
            if len(clean_data) < 10:
                return "Insufficient data for anomaly detection"
            
            # Prepare features
            features = []
            if 'RegionName' in clean_data.columns:
                # Add region-based features (encoded)
                region_counts = clean_data['RegionName'].value_counts()
                clean_data['region_frequency'] = clean_data['RegionName'].map(region_counts)
                features.append('region_frequency')
            
            features.append(value_col)
            
            # Add date-based features if available
            if 'date' in clean_data.columns:
                clean_data['date'] = pd.to_datetime(clean_data['date'])
                clean_data['month'] = clean_data['date'].dt.month
                clean_data['year'] = clean_data['date'].dt.year
                features.extend(['month', 'year'])
            
            # Prepare feature matrix
            feature_data = clean_data[features].copy()
            
            # Handle any remaining nulls
            feature_data = feature_data.fillna(feature_data.mean())
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_data)
            
            # Set DBSCAN parameters based on sensitivity
            eps_map = {"low": 1.5, "medium": 1.0, "high": 0.5}
            min_samples_map = {"low": 10, "medium": 5, "high": 3}
            
            eps = eps_map.get(sensitivity, 1.0)
            min_samples = min_samples_map.get(sensitivity, 5)
            
            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(scaled_features)
            
            # Identify anomalies (cluster label -1)
            anomaly_mask = cluster_labels == -1
            anomalies = clean_data[anomaly_mask]
            
            # Prepare results
            results = {
                "sensitivity": sensitivity,
                "total_data_points": len(clean_data),
                "anomalies_detected": len(anomalies),
                "anomaly_percentage": round((len(anomalies) / len(clean_data)) * 100, 2),
                "clusters_found": len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                "anomaly_examples": []
            }
            
            # Add examples of anomalies
            if len(anomalies) > 0:
                for i, (idx, row) in enumerate(anomalies.head(5).iterrows()):
                    example = {
                        "index": int(idx),
                        "value": float(row[value_col]) if pd.notna(row[value_col]) else None
                    }
                    
                    if 'RegionName' in row:
                        example["region"] = str(row['RegionName'])
                    if 'date' in row:
                        example["date"] = str(row['date'])
                    
                    results["anomaly_examples"].append(example)
            
            return json.dumps(results, indent=2)
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return f"Anomaly detection failed: {str(e)}"


class MarketDiscoveryAgent(BaseAgent):
    """
    CrewAI-powered autonomous data mining and pattern recognition agent
    for discovering market opportunities and anomalies.
    """
    
    def __init__(self, db_manager: DatabaseManager, **kwargs):
        super().__init__("MarketDiscoveryAgent", db_manager, **kwargs)
        
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.2,
            api_key=settings.openai_api_key
        )
        
        # Initialize tools
        self.tools = [
            PropertyAnalysisTool(db_manager),
            MarketTrendTool(db_manager),
            AnomalyDetectionTool(db_manager)
        ]
        
        # Create CrewAI agents
        self.data_analyst = Agent(
            role="Real Estate Data Analyst",
            goal="Analyze real estate market data to identify trends, patterns, and opportunities",
            backstory="""You are an experienced real estate data analyst with expertise in 
            market research, statistical analysis, and pattern recognition. You excel at
            finding insights hidden in large datasets and translating them into actionable
            intelligence for real estate professionals.""",
            tools=self.tools,
            llm=self.llm,
            verbose=True
        )
        
        self.pattern_detective = Agent(
            role="Market Pattern Detective", 
            goal="Discover unusual patterns, anomalies, and emerging trends in real estate data",
            backstory="""You are a specialized pattern detective with an exceptional ability
            to spot anomalies and emerging trends in real estate markets. Your expertise lies
            in identifying signals that others miss and recognizing patterns that indicate
            market shifts or opportunities.""",
            tools=self.tools,
            llm=self.llm,
            verbose=True
        )
        
        self.opportunity_hunter = Agent(
            role="Market Opportunity Hunter",
            goal="Identify and validate potential real estate investment and market opportunities",
            backstory="""You are a savvy opportunity hunter with a keen eye for identifying
            profitable real estate opportunities. You combine data analysis with market
            intuition to find undervalued markets, emerging neighborhoods, and investment
            prospects that others overlook.""",
            tools=self.tools,
            llm=self.llm,
            verbose=True
        )
        
        logger.info("Market Discovery Agent initialized with CrewAI agents")
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute market discovery task using CrewAI crew."""
        try:
            task_type = task.task_type
            input_data = task.input_data
            
            logger.info(f"Market Discovery Agent executing: {task_type}")
            
            # Create tasks based on input
            if task_type == "market_discovery":
                result = await self._run_market_discovery(input_data)
            elif task_type == "anomaly_detection":
                result = await self._run_anomaly_detection(input_data)
            elif task_type == "opportunity_identification":
                result = await self._run_opportunity_identification(input_data)
            else:
                result = await self._run_general_analysis(input_data)
            
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=True,
                output_data=result,
                confidence_score=self._calculate_confidence_score(result)
            )
            
        except Exception as e:
            logger.error(f"Market Discovery execution failed: {e}")
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=False,
                output_data={},
                error_message=str(e)
            )
    
    async def _run_market_discovery(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive market discovery analysis."""
        
        # Create CrewAI tasks
        analysis_task = Task(
            description=f"""
            Analyze the current real estate market data to identify key trends and patterns.
            Focus on: {input_data.get('focus_areas', 'general market trends')}
            Time period: {input_data.get('time_period', '6 months')}
            
            Provide comprehensive analysis including:
            1. Overall market condition assessment
            2. Key trends and patterns identified
            3. Regional variations and hotspots
            4. Price movement analysis
            5. Inventory and demand patterns
            """,
            agent=self.data_analyst,
            expected_output="Detailed market analysis report with key findings and trends"
        )
        
        pattern_task = Task(
            description="""
            Search for unusual patterns, anomalies, and emerging trends that might indicate
            market shifts or opportunities. Look for:
            1. Unexpected price movements
            2. Unusual inventory patterns
            3. Emerging market segments
            4. Regional anomalies
            5. Temporal patterns (seasonal, cyclical)
            
            Use anomaly detection tools to identify statistical outliers.
            """,
            agent=self.pattern_detective,
            expected_output="Report on detected patterns and anomalies with statistical evidence"
        )
        
        opportunity_task = Task(
            description="""
            Based on the market analysis and pattern detection, identify potential
            investment and market opportunities. Consider:
            1. Undervalued markets or regions
            2. Emerging growth areas
            3. Market inefficiencies
            4. Timing opportunities
            5. Risk-adjusted return potential
            
            Validate opportunities with data-driven evidence.
            """,
            agent=self.opportunity_hunter,
            expected_output="List of validated market opportunities with supporting analysis"
        )
        
        # Create and run crew
        crew = Crew(
            agents=[self.data_analyst, self.pattern_detective, self.opportunity_hunter],
            tasks=[analysis_task, pattern_task, opportunity_task],
            verbose=True
        )
        
        # Execute crew in thread pool (CrewAI is synchronous)
        loop = asyncio.get_event_loop()
        crew_result = await loop.run_in_executor(None, crew.kickoff)
        
        return {
            "analysis_type": "market_discovery",
            "crew_result": str(crew_result),
            "timestamp": datetime.now().isoformat(),
            "input_parameters": input_data
        }
    
    async def _run_anomaly_detection(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run focused anomaly detection analysis."""
        
        anomaly_task = Task(
            description=f"""
            Perform comprehensive anomaly detection on real estate market data.
            Sensitivity level: {input_data.get('sensitivity', 'medium')}
            Focus areas: {input_data.get('focus_areas', 'price and inventory anomalies')}
            
            Use machine learning anomaly detection tools to identify:
            1. Statistical outliers in pricing
            2. Unusual market behavior patterns
            3. Regional anomalies
            4. Temporal anomalies
            5. Multi-variate outliers
            
            Provide detailed analysis of each anomaly found.
            """,
            agent=self.pattern_detective,
            expected_output="Comprehensive anomaly detection report with findings and explanations"
        )
        
        validation_task = Task(
            description="""
            Validate and interpret the detected anomalies to determine their significance:
            1. Assess whether anomalies represent opportunities or risks
            2. Investigate potential causes of anomalies
            3. Determine actionability of findings
            4. Recommend follow-up analysis or actions
            """,
            agent=self.data_analyst,
            expected_output="Validation report with interpretation and recommendations for each anomaly"
        )
        
        crew = Crew(
            agents=[self.pattern_detective, self.data_analyst],
            tasks=[anomaly_task, validation_task],
            verbose=True
        )
        
        loop = asyncio.get_event_loop()
        crew_result = await loop.run_in_executor(None, crew.kickoff)
        
        return {
            "analysis_type": "anomaly_detection",
            "crew_result": str(crew_result),
            "timestamp": datetime.now().isoformat(),
            "input_parameters": input_data
        }
    
    async def _run_opportunity_identification(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run focused opportunity identification analysis."""
        
        opportunity_task = Task(
            description=f"""
            Identify and analyze potential real estate opportunities based on:
            Investment type: {input_data.get('investment_type', 'any')}
            Budget range: {input_data.get('budget_range', 'any')}
            Risk tolerance: {input_data.get('risk_tolerance', 'medium')}
            Geographic focus: {input_data.get('geographic_focus', 'any')}
            
            Analyze market data to find:
            1. Undervalued properties or markets
            2. Emerging growth areas
            3. Market timing opportunities
            4. Arbitrage opportunities
            5. Development prospects
            """,
            agent=self.opportunity_hunter,
            expected_output="Detailed opportunity analysis with specific recommendations"
        )
        
        crew = Crew(
            agents=[self.opportunity_hunter],
            tasks=[opportunity_task],
            verbose=True
        )
        
        loop = asyncio.get_event_loop()
        crew_result = await loop.run_in_executor(None, crew.kickoff)
        
        return {
            "analysis_type": "opportunity_identification",
            "crew_result": str(crew_result),
            "timestamp": datetime.now().isoformat(),
            "input_parameters": input_data
        }
    
    async def _run_general_analysis(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run general market analysis."""
        
        general_task = Task(
            description=f"""
            Perform general real estate market analysis based on available data.
            Analysis focus: {input_data.get('description', 'comprehensive market overview')}
            
            Provide insights on:
            1. Current market conditions
            2. Key trends and patterns
            3. Notable findings or anomalies
            4. Market opportunities or risks
            5. Recommendations for further analysis
            """,
            agent=self.data_analyst,
            expected_output="Comprehensive market analysis report"
        )
        
        crew = Crew(
            agents=[self.data_analyst],
            tasks=[general_task],
            verbose=True
        )
        
        loop = asyncio.get_event_loop()
        crew_result = await loop.run_in_executor(None, crew.kickoff)
        
        return {
            "analysis_type": "general_analysis",
            "crew_result": str(crew_result),
            "timestamp": datetime.now().isoformat(),
            "input_parameters": input_data
        }
    
    def _calculate_confidence_score(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score based on analysis results."""
        # Simple heuristic based on result completeness and data availability
        base_score = 0.7
        
        if "crew_result" in result and len(str(result["crew_result"])) > 100:
            base_score += 0.2
        
        if "analysis_type" in result:
            base_score += 0.1
        
        return min(base_score, 1.0)