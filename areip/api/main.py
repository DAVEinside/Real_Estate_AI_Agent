"""FastAPI application for AREIP."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..agents.coordinator import AgentCoordinator
from ..data.ingestion import DataIngestionPipeline
from ..utils.database import DatabaseManager
from ..utils.logging import setup_logging
from ..config import settings

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global variables for application state
db_manager: Optional[DatabaseManager] = None
coordinator: Optional[AgentCoordinator] = None
ingestion_pipeline: Optional[DataIngestionPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global db_manager, coordinator, ingestion_pipeline
    
    logger.info("Starting AREIP application...")
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager()
        await db_manager.initialize_database()
        
        # Initialize agent coordinator
        coordinator = AgentCoordinator(db_manager)
        
        # Initialize data ingestion pipeline
        ingestion_pipeline = DataIngestionPipeline(db_manager)
        
        logger.info("AREIP application started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start AREIP application: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down AREIP application...")
        
        if coordinator:
            await coordinator.close()
        
        if db_manager:
            await db_manager.close()
        
        logger.info("AREIP application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Autonomous Real Estate Intelligence Platform (AREIP)",
    description="Multi-agent system for autonomous real estate market analysis and decision making",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class AnalysisRequest(BaseModel):
    """Request model for market analysis."""
    analysis_type: str = Field(default="market_analysis", description="Type of analysis to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Analysis parameters")
    regions: Optional[List[str]] = Field(None, description="Geographic regions to focus on")
    time_period: str = Field(default="6m", description="Time period for analysis")
    include_properties: bool = Field(default=True, description="Include property data")
    include_market_data: bool = Field(default=True, description="Include market trend data")


class PropertyEvaluationRequest(BaseModel):
    """Request model for property evaluation."""
    property_data: Dict[str, Any] = Field(..., description="Property information")
    analysis_depth: str = Field(default="comprehensive", description="Depth of analysis")
    include_comparables: bool = Field(default=True, description="Include comparable properties")


class WorkflowResponse(BaseModel):
    """Response model for workflow operations."""
    workflow_id: str
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None


class AnalysisResponse(BaseModel):
    """Response model for analysis results."""
    workflow_id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    execution_time: Optional[float] = None


# Dependency injection
async def get_db_manager() -> DatabaseManager:
    """Get database manager dependency."""
    if db_manager is None:
        raise HTTPException(status_code=500, detail="Database manager not initialized")
    return db_manager


async def get_coordinator() -> AgentCoordinator:
    """Get agent coordinator dependency."""
    if coordinator is None:
        raise HTTPException(status_code=500, detail="Agent coordinator not initialized")
    return coordinator


async def get_ingestion_pipeline() -> DataIngestionPipeline:
    """Get ingestion pipeline dependency."""
    if ingestion_pipeline is None:
        raise HTTPException(status_code=500, detail="Ingestion pipeline not initialized")
    return ingestion_pipeline


# API Routes

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environment": settings.environment,
        "timestamp": "2024-01-01T00:00:00Z"
    }


@app.get("/status")
async def get_system_status(coord: AgentCoordinator = Depends(get_coordinator)):
    """Get system status including agent states."""
    try:
        agent_status = await coord.get_agent_status()
        active_workflows = await coord.list_active_workflows()
        
        return {
            "system_status": "operational",
            "agents": agent_status,
            "active_workflows": len(active_workflows),
            "workflows": active_workflows
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analysis/market", response_model=WorkflowResponse)
async def start_market_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    coord: AgentCoordinator = Depends(get_coordinator)
):
    """Start a comprehensive market analysis workflow."""
    try:
        # Create workflow
        workflow_id = await coord.create_market_analysis_workflow(
            parameters={
                "regions": request.regions or [],
                "time_period": request.time_period,
                "include_properties": request.include_properties,
                "include_market_data": request.include_market_data,
                "focus_areas": request.parameters.get("focus_areas", ["market trends", "pricing patterns"]),
                "analysis_depth": request.parameters.get("analysis_depth", "comprehensive")
            }
        )
        
        # Execute workflow in background
        background_tasks.add_task(coord.execute_workflow, workflow_id)
        
        return WorkflowResponse(
            workflow_id=workflow_id,
            status="started",
            message="Market analysis workflow started successfully",
            details={"analysis_type": request.analysis_type}
        )
        
    except Exception as e:
        logger.error(f"Error starting market analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analysis/property", response_model=WorkflowResponse)
async def start_property_evaluation(
    request: PropertyEvaluationRequest,
    background_tasks: BackgroundTasks,
    coord: AgentCoordinator = Depends(get_coordinator)
):
    """Start a property evaluation workflow."""
    try:
        # Create workflow
        workflow_id = await coord.create_property_evaluation_workflow(request.property_data)
        
        # Execute workflow in background
        background_tasks.add_task(coord.execute_workflow, workflow_id)
        
        return WorkflowResponse(
            workflow_id=workflow_id,
            status="started", 
            message="Property evaluation workflow started successfully",
            details={"property_id": request.property_data.get("id", "unknown")}
        )
        
    except Exception as e:
        logger.error(f"Error starting property evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workflow/{workflow_id}/status")
async def get_workflow_status(
    workflow_id: str,
    coord: AgentCoordinator = Depends(get_coordinator)
):
    """Get the status of a specific workflow."""
    try:
        status = await coord.get_workflow_status(workflow_id)
        
        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workflow/{workflow_id}/results", response_model=AnalysisResponse)
async def get_workflow_results(
    workflow_id: str,
    coord: AgentCoordinator = Depends(get_coordinator)
):
    """Get the results of a completed workflow."""
    try:
        status = await coord.get_workflow_status(workflow_id)
        
        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])
        
        # Check if workflow is completed
        if status["status"] != "completed":
            return AnalysisResponse(
                workflow_id=workflow_id,
                status=status["status"],
                results=None,
                summary="Workflow not yet completed"
            )
        
        # Get workflow results
        workflow = coord.active_workflows.get(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Prepare results
        results = {}
        for task_id, result in workflow.results.items():
            results[task_id] = {
                "success": result.success,
                "output_data": result.output_data,
                "confidence_score": result.confidence_score,
                "execution_time": result.execution_time_seconds
            }
        
        return AnalysisResponse(
            workflow_id=workflow_id,
            status="completed",
            results=results,
            summary="Analysis completed successfully",
            execution_time=(workflow.completed_at - workflow.started_at).total_seconds() if workflow.completed_at and workflow.started_at else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/workflow/{workflow_id}")
async def cancel_workflow(
    workflow_id: str,
    coord: AgentCoordinator = Depends(get_coordinator)
):
    """Cancel a running workflow."""
    try:
        success = await coord.cancel_workflow(workflow_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Workflow not found or cannot be cancelled")
        
        return {"message": f"Workflow {workflow_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workflows")
async def list_workflows(coord: AgentCoordinator = Depends(get_coordinator)):
    """List all active workflows."""
    try:
        workflows = await coord.list_active_workflows()
        return {"workflows": workflows}
        
    except Exception as e:
        logger.error(f"Error listing workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/ingest")
async def trigger_data_ingestion(
    background_tasks: BackgroundTasks,
    full_ingestion: bool = False,
    pipeline: DataIngestionPipeline = Depends(get_ingestion_pipeline)
):
    """Trigger data ingestion from all sources."""
    try:
        if full_ingestion:
            background_tasks.add_task(pipeline.run_full_ingestion)
            message = "Full data ingestion started"
        else:
            background_tasks.add_task(pipeline.run_incremental_ingestion)
            message = "Incremental data ingestion started"
        
        return {"message": message, "status": "started"}
        
    except Exception as e:
        logger.error(f"Error triggering data ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/status")
async def get_data_status(pipeline: DataIngestionPipeline = Depends(get_ingestion_pipeline)):
    """Get data ingestion status."""
    try:
        status = await pipeline.get_ingestion_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting data status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents")
async def get_agents_info(coord: AgentCoordinator = Depends(get_coordinator)):
    """Get information about available agents."""
    try:
        agent_status = await coord.get_agent_status()
        
        agent_info = {}
        for name, agent in coord.agents.items():
            agent_info[name] = {
                "name": agent.agent_name,
                "id": agent.agent_id,
                "status": agent_status[name]["status"],
                "description": f"{agent.__class__.__name__} - {agent.__doc__}" if agent.__doc__ else agent.__class__.__name__
            }
        
        return {"agents": agent_info}
        
    except Exception as e:
        logger.error(f"Error getting agents info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "areip.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower()
    )