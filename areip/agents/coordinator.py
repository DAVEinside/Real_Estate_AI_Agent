"""Agent Coordinator for managing multi-agent workflows and communication."""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .base import BaseAgent, AgentTask, AgentResult, AgentStatus
from .orchestrator import IntelligenceOrchestratorAgent
from .discovery import MarketDiscoveryAgent
from .synthesis import KnowledgeSynthesisAgent
from .validation import ValidationRiskAgent
from ..utils.database import DatabaseManager
from ..config import settings

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending" 
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class Workflow:
    """Workflow definition for multi-agent coordination."""
    workflow_id: str
    name: str
    description: str
    tasks: List[AgentTask]
    dependencies: Dict[str, List[str]]  # task_id -> [dependency_task_ids]
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Dict[str, AgentResult] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.results is None:
            self.results = {}


class AgentCoordinator:
    """
    Coordinates multiple agents to execute complex workflows
    with proper dependency management and result aggregation.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
        # Initialize all agents
        self.orchestrator = IntelligenceOrchestratorAgent(db_manager)
        self.discovery_agent = MarketDiscoveryAgent(db_manager)
        self.synthesis_agent = KnowledgeSynthesisAgent(db_manager)
        self.validation_agent = ValidationRiskAgent(db_manager)
        
        # Register agents with orchestrator
        self.orchestrator.register_agent(self.discovery_agent)
        self.orchestrator.register_agent(self.synthesis_agent)
        self.orchestrator.register_agent(self.validation_agent)
        
        # Agent registry
        self.agents = {
            "orchestrator": self.orchestrator,
            "discovery": self.discovery_agent,
            "synthesis": self.synthesis_agent,
            "validation": self.validation_agent
        }
        
        # Active workflows
        self.active_workflows: Dict[str, Workflow] = {}
        
        logger.info("Agent Coordinator initialized with all agents")
    
    async def create_market_analysis_workflow(self, parameters: Dict[str, Any]) -> str:
        """Create a comprehensive market analysis workflow."""
        workflow_id = f"market_analysis_{uuid.uuid4().hex[:8]}"
        
        # Create tasks for the workflow
        tasks = []
        
        # 1. Data ingestion and preparation
        ingestion_task = AgentTask(
            task_id=f"{workflow_id}_ingestion",
            task_type="data_ingestion",
            description="Ingest and prepare market data for analysis",
            input_data={
                "ingest_properties": parameters.get("include_properties", True),
                "ingest_market_data": parameters.get("include_market_data", True),
                "focus_regions": parameters.get("regions", [])
            }
        )
        tasks.append(ingestion_task)
        
        # 2. Market discovery and pattern analysis
        discovery_task = AgentTask(
            task_id=f"{workflow_id}_discovery",
            task_type="market_discovery",
            description="Discover market patterns and opportunities",
            input_data={
                "focus_areas": parameters.get("focus_areas", ["market trends", "pricing patterns"]),
                "time_period": parameters.get("time_period", "6m"),
                "analysis_depth": parameters.get("analysis_depth", "comprehensive")
            }
        )
        tasks.append(discovery_task)
        
        # 3. Knowledge synthesis
        synthesis_task = AgentTask(
            task_id=f"{workflow_id}_synthesis",
            task_type="synthesis_analysis",
            description="Synthesize findings from market discovery",
            input_data={
                "focus_areas": parameters.get("focus_areas", ["market trends", "investment opportunities"]),
                "synthesis_type": "market_intelligence"
            }
        )
        tasks.append(synthesis_task)
        
        # 4. Risk assessment and validation
        validation_task = AgentTask(
            task_id=f"{workflow_id}_validation",
            task_type="assess_risk",
            description="Validate findings and assess risks",
            input_data={
                "risk_type": "comprehensive",
                "validation_level": parameters.get("validation_level", "thorough")
            }
        )
        tasks.append(validation_task)
        
        # 5. Final orchestration and decision making
        orchestration_task = AgentTask(
            task_id=f"{workflow_id}_orchestration",
            task_type="strategic_coordination",
            description="Coordinate findings and make strategic recommendations",
            input_data={
                "decision_scope": parameters.get("decision_scope", "investment_recommendations"),
                "risk_tolerance": parameters.get("risk_tolerance", "moderate")
            }
        )
        tasks.append(orchestration_task)
        
        # Define dependencies
        dependencies = {
            discovery_task.task_id: [ingestion_task.task_id],
            synthesis_task.task_id: [discovery_task.task_id],
            validation_task.task_id: [synthesis_task.task_id, discovery_task.task_id],
            orchestration_task.task_id: [validation_task.task_id, synthesis_task.task_id]
        }
        
        # Create workflow
        workflow = Workflow(
            workflow_id=workflow_id,
            name="Market Analysis Workflow",
            description="Comprehensive market analysis with discovery, synthesis, and validation",
            tasks=tasks,
            dependencies=dependencies
        )
        
        self.active_workflows[workflow_id] = workflow
        
        logger.info(f"Created market analysis workflow: {workflow_id}")
        return workflow_id
    
    async def create_property_evaluation_workflow(self, property_data: Dict[str, Any]) -> str:
        """Create a property evaluation workflow."""
        workflow_id = f"property_eval_{uuid.uuid4().hex[:8]}"
        
        tasks = []
        
        # 1. Property data analysis
        analysis_task = AgentTask(
            task_id=f"{workflow_id}_analysis",
            task_type="property_analysis",
            description="Analyze individual property characteristics",
            input_data={
                "property_data": property_data,
                "analysis_type": "comprehensive"
            }
        )
        tasks.append(analysis_task)
        
        # 2. Market comparison
        comparison_task = AgentTask(
            task_id=f"{workflow_id}_comparison",
            task_type="market_discovery",
            description="Compare property against market conditions",
            input_data={
                "focus_areas": ["comparable properties", "market pricing"],
                "property_location": property_data.get("location", ""),
                "property_type": property_data.get("property_type", "")
            }
        )
        tasks.append(comparison_task)
        
        # 3. Risk assessment
        risk_task = AgentTask(
            task_id=f"{workflow_id}_risk",
            task_type="assess_risk",
            description="Assess property-specific risks",
            input_data={
                "risk_type": "property",
                "target_data": {"property_data": property_data}
            }
        )
        tasks.append(risk_task)
        
        # 4. Investment recommendation
        recommendation_task = AgentTask(
            task_id=f"{workflow_id}_recommendation",
            task_type="investment_analysis",
            description="Generate investment recommendation",
            input_data={
                "property_data": property_data,
                "analysis_scope": "investment_decision"
            }
        )
        tasks.append(recommendation_task)
        
        dependencies = {
            comparison_task.task_id: [analysis_task.task_id],
            risk_task.task_id: [analysis_task.task_id],
            recommendation_task.task_id: [comparison_task.task_id, risk_task.task_id]
        }
        
        workflow = Workflow(
            workflow_id=workflow_id,
            name="Property Evaluation Workflow",
            description="Comprehensive property evaluation and investment analysis",
            tasks=tasks,
            dependencies=dependencies
        )
        
        self.active_workflows[workflow_id] = workflow
        
        logger.info(f"Created property evaluation workflow: {workflow_id}")
        return workflow_id
    
    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute a workflow with proper dependency management."""
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.active_workflows[workflow_id]
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now()
        
        logger.info(f"Starting workflow execution: {workflow_id}")
        
        try:
            # Track task completion
            completed_tasks = set()
            task_results = {}
            
            # Execute tasks respecting dependencies
            while len(completed_tasks) < len(workflow.tasks):
                ready_tasks = []
                
                for task in workflow.tasks:
                    if task.task_id in completed_tasks:
                        continue
                    
                    # Check if all dependencies are completed
                    dependencies = workflow.dependencies.get(task.task_id, [])
                    if all(dep_id in completed_tasks for dep_id in dependencies):
                        ready_tasks.append(task)
                
                if not ready_tasks:
                    raise Exception("Circular dependency or blocked workflow detected")
                
                # Execute ready tasks in parallel
                execution_tasks = []
                for task in ready_tasks:
                    agent = self._get_agent_for_task(task)
                    if agent:
                        execution_tasks.append(self._execute_task_with_context(agent, task, task_results))
                
                # Wait for all ready tasks to complete
                results = await asyncio.gather(*execution_tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    task = ready_tasks[i]
                    
                    if isinstance(result, Exception):
                        logger.error(f"Task {task.task_id} failed: {result}")
                        workflow.status = WorkflowStatus.FAILED
                        raise result
                    else:
                        task_results[task.task_id] = result
                        completed_tasks.add(task.task_id)
                        workflow.results[task.task_id] = result
                        logger.info(f"Task {task.task_id} completed successfully")
            
            # Workflow completed successfully
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.now()
            
            # Generate workflow summary
            summary = await self._generate_workflow_summary(workflow)
            
            logger.info(f"Workflow {workflow_id} completed successfully")
            
            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "execution_time": (workflow.completed_at - workflow.started_at).total_seconds(),
                "tasks_completed": len(completed_tasks),
                "results": task_results,
                "summary": summary
            }
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            logger.error(f"Workflow {workflow_id} failed: {e}")
            
            return {
                "workflow_id": workflow_id,
                "status": "failed", 
                "error": str(e),
                "tasks_completed": len(completed_tasks) if 'completed_tasks' in locals() else 0
            }
    
    async def _execute_task_with_context(self, agent: BaseAgent, task: AgentTask, previous_results: Dict[str, Any]) -> AgentResult:
        """Execute a task with context from previous results."""
        # Add context from previous results
        if previous_results:
            task.input_data["previous_results"] = previous_results
            
            # Add specific context based on task type
            if task.task_type == "synthesis_analysis":
                # Add discovery results for synthesis
                discovery_results = {k: v for k, v in previous_results.items() if "discovery" in k}
                if discovery_results:
                    task.input_data["discovery_context"] = discovery_results
            
            elif task.task_type == "assess_risk":
                # Add analysis results for risk assessment
                analysis_results = {k: v for k, v in previous_results.items() if "analysis" in k or "discovery" in k}
                if analysis_results:
                    task.input_data["analysis_results"] = analysis_results
        
        return await agent.run(task)
    
    def _get_agent_for_task(self, task: AgentTask) -> Optional[BaseAgent]:
        """Get the appropriate agent for a task."""
        task_type = task.task_type
        
        # Task routing logic
        if task_type in ["data_ingestion", "synthesis_analysis", "knowledge_query"]:
            return self.synthesis_agent
        elif task_type in ["market_discovery", "property_analysis", "opportunity_identification"]:
            return self.discovery_agent
        elif task_type in ["assess_risk", "validate_analysis", "validate_trends"]:
            return self.validation_agent
        elif task_type in ["strategic_coordination", "investment_analysis"]:
            return self.orchestrator
        else:
            # Default to orchestrator for unknown tasks
            return self.orchestrator
    
    async def _generate_workflow_summary(self, workflow: Workflow) -> str:
        """Generate a summary of workflow execution results."""
        from langchain_openai import ChatOpenAI
        from langchain.schema import HumanMessage, SystemMessage
        
        llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            api_key=settings.openai_api_key
        )
        
        # Prepare results for summary
        results_summary = {}
        for task_id, result in workflow.results.items():
            results_summary[task_id] = {
                "success": result.success,
                "output_keys": list(result.output_data.keys()) if result.output_data else [],
                "confidence_score": result.confidence_score,
                "execution_time": result.execution_time_seconds
            }
        
        system_prompt = """
        You are an expert analyst summarizing the results of a multi-agent real estate analysis workflow.
        Provide a concise, executive-level summary of the key findings and recommendations.
        """
        
        human_prompt = f"""
        Workflow: {workflow.name}
        Description: {workflow.description}
        
        Task Results Summary:
        {results_summary}
        
        Provide a comprehensive summary covering:
        1. Key findings and insights
        2. Strategic recommendations
        3. Risk factors identified
        4. Data quality and confidence assessment
        5. Next steps or follow-up actions needed
        """
        
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        response = await llm.ainvoke(messages)
        
        return response.content
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get the current status of a workflow."""
        if workflow_id not in self.active_workflows:
            return {"error": f"Workflow {workflow_id} not found"}
        
        workflow = self.active_workflows[workflow_id]
        
        # Calculate progress
        completed_tasks = len(workflow.results)
        total_tasks = len(workflow.tasks)
        progress = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
        
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "status": workflow.status.value,
            "progress": progress,
            "tasks_completed": completed_tasks,
            "total_tasks": total_tasks,
            "created_at": workflow.created_at.isoformat(),
            "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
            "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None
        }
    
    async def list_active_workflows(self) -> List[Dict[str, Any]]:
        """List all active workflows."""
        workflows = []
        
        for workflow_id, workflow in self.active_workflows.items():
            status = await self.get_workflow_status(workflow_id)
            workflows.append(status)
        
        return workflows
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        if workflow_id not in self.active_workflows:
            return False
        
        workflow = self.active_workflows[workflow_id]
        
        if workflow.status == WorkflowStatus.RUNNING:
            workflow.status = WorkflowStatus.PAUSED
            logger.info(f"Workflow {workflow_id} has been cancelled")
            return True
        
        return False
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all managed agents."""
        agent_status = {}
        
        for name, agent in self.agents.items():
            memory_summary = await agent.get_memory_summary()
            agent_status[name] = {
                "status": agent.status.value,
                "current_task": agent.current_task.task_id if agent.current_task else None,
                "memory_summary": memory_summary
            }
        
        return agent_status
    
    async def close(self):
        """Close all agents and cleanup resources."""
        logger.info("Shutting down Agent Coordinator")
        
        # Close knowledge synthesis agent (has Neo4j connection)
        await self.synthesis_agent.close()
        
        # Close database connections
        await self.db_manager.close()
        
        logger.info("Agent Coordinator shutdown complete")