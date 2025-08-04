"""Intelligence Orchestrator Agent using LangGraph for workflow coordination."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, TypedDict
import json

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from .base import BaseAgent, AgentTask, AgentResult, AgentStatus
from ..config import settings
from ..utils.database import DatabaseManager

logger = logging.getLogger(__name__)


class OrchestratorState(TypedDict):
    """State structure for the orchestrator workflow."""
    task_queue: List[Dict[str, Any]]
    active_agents: Dict[str, str]
    completed_tasks: List[Dict[str, Any]]
    failed_tasks: List[Dict[str, Any]]
    current_analysis: Dict[str, Any]
    market_context: Dict[str, Any]
    decisions: List[Dict[str, Any]]
    next_action: str


class IntelligenceOrchestratorAgent(BaseAgent):
    """
    Master coordinator agent using LangGraph for complex decision trees
    and workflow management across multiple specialized agents.
    """
    
    def __init__(self, db_manager: DatabaseManager, **kwargs):
        super().__init__("IntelligenceOrchestrator", db_manager, **kwargs)
        
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            api_key=settings.openai_api_key
        )
        
        # Initialize workflow graph
        self.workflow = self._create_workflow()
        self.memory_saver = SqliteSaver.from_conn_string(":memory:")
        self.app = self.workflow.compile(checkpointer=self.memory_saver)
        
        # Agent registry for coordination
        self.agent_registry: Dict[str, BaseAgent] = {}
        
        logger.info("Intelligence Orchestrator Agent initialized with LangGraph workflow")
    
    def register_agent(self, agent: BaseAgent):
        """Register a specialized agent for coordination."""
        self.agent_registry[agent.agent_name] = agent
        logger.info(f"Registered agent: {agent.agent_name}")
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for orchestration."""
        workflow = StateGraph(OrchestratorState)
        
        # Define workflow nodes
        workflow.add_node("analyze_market_context", self._analyze_market_context)
        workflow.add_node("prioritize_tasks", self._prioritize_tasks)
        workflow.add_node("allocate_agents", self._allocate_agents)
        workflow.add_node("monitor_execution", self._monitor_execution)
        workflow.add_node("synthesize_results", self._synthesize_results)
        workflow.add_node("make_decisions", self._make_decisions)
        workflow.add_node("update_strategy", self._update_strategy)
        
        # Define workflow edges and conditions
        workflow.set_entry_point("analyze_market_context")
        
        workflow.add_edge("analyze_market_context", "prioritize_tasks")
        workflow.add_edge("prioritize_tasks", "allocate_agents")
        workflow.add_edge("allocate_agents", "monitor_execution")
        
        # Conditional edges based on execution status
        workflow.add_conditional_edges(
            "monitor_execution",
            self._should_continue_monitoring,
            {
                "continue": "monitor_execution",
                "synthesize": "synthesize_results",
                "reallocate": "allocate_agents"
            }
        )
        
        workflow.add_edge("synthesize_results", "make_decisions")
        
        workflow.add_conditional_edges(
            "make_decisions",
            self._should_update_strategy,
            {
                "update": "update_strategy",
                "complete": END
            }
        )
        
        workflow.add_edge("update_strategy", "prioritize_tasks")
        
        return workflow
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute orchestration task using LangGraph workflow."""
        try:
            logger.info(f"Orchestrator executing task: {task.task_type}")
            
            # Initialize workflow state
            initial_state: OrchestratorState = {
                "task_queue": [task.__dict__],
                "active_agents": {},
                "completed_tasks": [],
                "failed_tasks": [],
                "current_analysis": {},
                "market_context": {},
                "decisions": [],
                "next_action": "start"
            }
            
            # Run the workflow
            config = {"configurable": {"thread_id": f"orchestrator_{task.task_id}"}}
            final_state = await self.app.ainvoke(initial_state, config)
            
            # Prepare result
            success = len(final_state["failed_tasks"]) == 0
            output_data = {
                "completed_tasks": final_state["completed_tasks"],
                "failed_tasks": final_state["failed_tasks"],
                "decisions": final_state["decisions"],
                "market_analysis": final_state["current_analysis"]
            }
            
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=success,
                output_data=output_data,
                confidence_score=self._calculate_confidence(final_state)
            )
            
        except Exception as e:
            logger.error(f"Orchestrator execution failed: {e}")
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=False,
                output_data={},
                error_message=str(e)
            )
    
    async def _analyze_market_context(self, state: OrchestratorState) -> OrchestratorState:
        """Analyze current market context and conditions."""
        logger.info("Analyzing market context")
        
        try:
            # Get recent market data from database
            market_data = await self._get_recent_market_data()
            
            # Use LLM to analyze market conditions
            system_prompt = """
            You are an expert real estate market analyst. Analyze the provided market data
            and provide insights about current market conditions, trends, and opportunities.
            Focus on identifying key patterns and anomalies that require attention.
            """
            
            human_prompt = f"""
            Analyze this recent market data and provide key insights:
            
            Market Data Summary:
            {json.dumps(market_data, indent=2)}
            
            Provide analysis in the following format:
            - Market Condition: [bullish/bearish/neutral]
            - Key Trends: [list 3-5 key trends]
            - Opportunities: [list potential opportunities]  
            - Risks: [list potential risks]
            - Priority Actions: [list recommended actions]
            """
            
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            response = await self.llm.ainvoke(messages)
            
            # Parse LLM response
            analysis = self._parse_market_analysis(response.content)
            
            state["market_context"] = market_data
            state["current_analysis"] = analysis
            
            logger.info(f"Market analysis completed: {analysis.get('market_condition', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Market context analysis failed: {e}")
            state["current_analysis"] = {"error": str(e)}
        
        return state
    
    async def _prioritize_tasks(self, state: OrchestratorState) -> OrchestratorState:
        """Prioritize tasks based on market analysis and agent capabilities."""
        logger.info("Prioritizing tasks")
        
        try:
            tasks = state["task_queue"]
            analysis = state["current_analysis"]
            
            # Use LLM to prioritize tasks
            system_prompt = """
            You are a task prioritization expert for real estate analysis.
            Given the current market analysis and available tasks, prioritize tasks
            based on market urgency, potential impact, and resource requirements.
            """
            
            human_prompt = f"""
            Current Market Analysis:
            {json.dumps(analysis, indent=2)}
            
            Available Tasks:
            {json.dumps(tasks, indent=2)}
            
            Available Agents: {list(self.agent_registry.keys())}
            
            Prioritize these tasks and suggest task allocation. Return a JSON response with:
            {{
                "prioritized_tasks": [list of tasks ordered by priority],
                "allocation_suggestions": {{
                    "task_id": "recommended_agent_name"
                }}
            }}
            """
            
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            response = await self.llm.ainvoke(messages)
            
            # Parse prioritization response
            prioritization = self._parse_json_response(response.content)
            
            state["task_queue"] = prioritization.get("prioritized_tasks", tasks)
            state["next_action"] = "allocate"
            
            logger.info(f"Prioritized {len(state['task_queue'])} tasks")
            
        except Exception as e:
            logger.error(f"Task prioritization failed: {e}")
        
        return state
    
    async def _allocate_agents(self, state: OrchestratorState) -> OrchestratorState:
        """Allocate tasks to appropriate specialized agents."""
        logger.info("Allocating agents to tasks")
        
        allocated_count = 0
        for task_dict in state["task_queue"]:
            if len(state["active_agents"]) >= settings.max_concurrent_agents:
                break
            
            task_id = task_dict["task_id"]
            task_type = task_dict["task_type"]
            
            # Find best agent for this task
            best_agent = self._find_best_agent_for_task(task_type)
            
            if best_agent and best_agent.status == AgentStatus.IDLE:
                state["active_agents"][task_id] = best_agent.agent_name
                allocated_count += 1
                
                # Start task execution asynchronously
                task = AgentTask(**task_dict)
                asyncio.create_task(self._execute_agent_task(best_agent, task, state))
        
        logger.info(f"Allocated {allocated_count} tasks to agents")
        state["next_action"] = "monitor"
        
        return state
    
    async def _monitor_execution(self, state: OrchestratorState) -> OrchestratorState:
        """Monitor agent execution and handle results."""
        logger.info("Monitoring agent execution")
        
        # Wait for a short period to allow agents to make progress
        await asyncio.sleep(2.0)
        
        # Check status of active agents
        active_tasks = list(state["active_agents"].keys())
        
        for task_id in active_tasks:
            agent_name = state["active_agents"][task_id]
            agent = self.agent_registry.get(agent_name)
            
            if agent and agent.status in [AgentStatus.COMPLETED, AgentStatus.FAILED]:
                # Task completed, move from active to completed/failed
                del state["active_agents"][task_id]
                
                if agent.status == AgentStatus.COMPLETED:
                    # Get result from agent memory
                    result = self._get_agent_result(agent, task_id)
                    state["completed_tasks"].append(result)
                    logger.info(f"Task {task_id} completed by {agent_name}")
                else:
                    state["failed_tasks"].append({
                        "task_id": task_id,
                        "agent_name": agent_name,
                        "error": "Agent execution failed"
                    })
                    logger.warning(f"Task {task_id} failed on {agent_name}")
        
        # Determine next action
        if not state["active_agents"] and not state["task_queue"]:
            state["next_action"] = "synthesize"
        elif not state["active_agents"] and state["task_queue"]:
            state["next_action"] = "reallocate"
        else:
            state["next_action"] = "continue"
        
        return state
    
    async def _synthesize_results(self, state: OrchestratorState) -> OrchestratorState:
        """Synthesize results from all agent executions."""
        logger.info("Synthesizing agent results")
        
        try:
            completed_tasks = state["completed_tasks"]
            
            if not completed_tasks:
                state["decisions"] = [{"type": "no_results", "message": "No tasks completed"}]
                return state
            
            # Use LLM to synthesize insights
            system_prompt = """
            You are an expert real estate intelligence synthesizer. 
            Analyze the results from multiple specialized agents and provide
            actionable insights and recommendations.
            """
            
            human_prompt = f"""
            Agent Execution Results:
            {json.dumps(completed_tasks, indent=2)}
            
            Market Context:
            {json.dumps(state["current_analysis"], indent=2)}
            
            Synthesize these results and provide:
            1. Key insights discovered
            2. Market opportunities identified
            3. Risk factors to consider
            4. Recommended actions
            5. Confidence level in recommendations
            
            Format as JSON with these sections.
            """
            
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            response = await self.llm.ainvoke(messages)
            
            synthesis = self._parse_json_response(response.content)
            state["current_analysis"]["synthesis"] = synthesis
            
            logger.info("Results synthesis completed")
            
        except Exception as e:
            logger.error(f"Results synthesis failed: {e}")
            state["current_analysis"]["synthesis_error"] = str(e)
        
        return state
    
    async def _make_decisions(self, state: OrchestratorState) -> OrchestratorState:
        """Make strategic decisions based on synthesized results."""
        logger.info("Making strategic decisions")
        
        try:
            synthesis = state["current_analysis"].get("synthesis", {})
            
            # Use LLM for decision making
            system_prompt = """
            You are a strategic decision-maker for real estate intelligence.
            Based on the analysis and synthesis, make concrete decisions about
            next steps, resource allocation, and strategic priorities.
            """
            
            human_prompt = f"""
            Analysis Synthesis:
            {json.dumps(synthesis, indent=2)}
            
            Make strategic decisions covering:
            1. Immediate actions to take
            2. Resources to allocate
            3. Follow-up analyses needed
            4. Risk mitigation strategies
            5. Performance metrics to track
            
            Return decisions as a JSON list with priority, action, and reasoning.
            """
            
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            response = await self.llm.ainvoke(messages)
            
            decisions = self._parse_json_response(response.content)
            state["decisions"] = decisions if isinstance(decisions, list) else [decisions]
            
            logger.info(f"Made {len(state['decisions'])} strategic decisions")
            
        except Exception as e:
            logger.error(f"Decision making failed: {e}")
            state["decisions"] = [{"error": str(e)}]
        
        return state
    
    async def _update_strategy(self, state: OrchestratorState) -> OrchestratorState:
        """Update orchestration strategy based on learnings."""
        logger.info("Updating orchestration strategy")
        
        # Update long-term memory with learnings
        decisions = state["decisions"]
        completed_tasks = state["completed_tasks"]
        
        # Extract patterns for future optimization
        patterns = {
            "successful_agent_allocations": [],
            "optimal_task_sequences": [],
            "market_response_patterns": []
        }
        
        for task in completed_tasks:
            if task.get("success"):
                patterns["successful_agent_allocations"].append({
                    "task_type": task.get("task_type"),
                    "agent_name": task.get("agent_name"),
                    "execution_time": task.get("execution_time")
                })
        
        # Store patterns in memory
        self.memory.long_term_memory["orchestration_patterns"] = patterns
        
        state["next_action"] = "complete"
        return state
    
    # Conditional edge functions
    def _should_continue_monitoring(self, state: OrchestratorState) -> str:
        """Determine if monitoring should continue."""
        if state["next_action"] == "continue":
            return "continue"
        elif state["next_action"] == "reallocate":
            return "reallocate"
        else:
            return "synthesize"
    
    def _should_update_strategy(self, state: OrchestratorState) -> str:
        """Determine if strategy should be updated."""
        # Check if there are learnings that warrant strategy update
        failed_tasks = state["failed_tasks"]
        if len(failed_tasks) > len(state["completed_tasks"]) * 0.3:  # >30% failure rate
            return "update"
        else:
            return "complete"
    
    # Helper methods
    async def _get_recent_market_data(self) -> Dict[str, Any]:
        """Get recent market data summary from database."""
        try:
            # Get record counts and recent data
            table_counts = await self.db_manager.get_table_record_counts()
            
            return {
                "data_sources_available": list(table_counts.keys()),
                "total_records": sum(table_counts.values()),
                "last_update": datetime.now().isoformat(),
                "data_freshness": "recent"  # Could calculate actual freshness
            }
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {"error": str(e)}
    
    def _parse_market_analysis(self, content: str) -> Dict[str, Any]:
        """Parse market analysis from LLM response."""
        try:
            # Try to parse as JSON first
            return json.loads(content)
        except json.JSONDecodeError:
            # Parse structured text format
            lines = content.strip().split('\n')
            analysis = {}
            
            current_section = None
            for line in lines:
                line = line.strip()
                if line.startswith('- Market Condition:'):
                    analysis['market_condition'] = line.split(':', 1)[1].strip()
                elif line.startswith('- Key Trends:'):
                    analysis['key_trends'] = line.split(':', 1)[1].strip()
                elif line.startswith('- Opportunities:'):
                    analysis['opportunities'] = line.split(':', 1)[1].strip()
                elif line.startswith('- Risks:'):
                    analysis['risks'] = line.split(':', 1)[1].strip()
                elif line.startswith('- Priority Actions:'):
                    analysis['priority_actions'] = line.split(':', 1)[1].strip()
            
            return analysis
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON response from LLM."""
        try:
            # Clean up common JSON formatting issues
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return {"error": "Failed to parse response", "raw_content": content}
    
    def _find_best_agent_for_task(self, task_type: str) -> Optional[BaseAgent]:
        """Find the best agent for a given task type."""
        # Simple mapping - could be made more sophisticated
        agent_mapping = {
            "market_discovery": "MarketDiscoveryAgent",
            "data_analysis": "KnowledgeSynthesisAgent", 
            "risk_assessment": "ValidationRiskAgent",
            "property_analysis": "KnowledgeSynthesisAgent"
        }
        
        preferred_agent = agent_mapping.get(task_type)
        if preferred_agent and preferred_agent in self.agent_registry:
            return self.agent_registry[preferred_agent]
        
        # Return any available agent as fallback
        for agent in self.agent_registry.values():
            if agent.status == AgentStatus.IDLE:
                return agent
        
        return None
    
    async def _execute_agent_task(self, agent: BaseAgent, task: AgentTask, state: OrchestratorState):
        """Execute a task on a specific agent asynchronously."""
        try:
            result = await agent.run(task)
            # Result handling is done in monitoring phase
        except Exception as e:
            logger.error(f"Agent task execution failed: {e}")
    
    def _get_agent_result(self, agent: BaseAgent, task_id: str) -> Dict[str, Any]:
        """Get result from agent's short-term memory."""
        if task_id in agent.memory.short_term_memory:
            return agent.memory.short_term_memory[task_id]['result']
        else:
            return {
                "task_id": task_id,
                "agent_name": agent.agent_name,
                "success": False,
                "error": "Result not found in agent memory"
            }
    
    def _calculate_confidence(self, state: OrchestratorState) -> float:
        """Calculate confidence score based on execution results."""
        completed = len(state["completed_tasks"])
        failed = len(state["failed_tasks"])
        total = completed + failed
        
        if total == 0:
            return 0.0
        
        success_rate = completed / total
        
        # Factor in market analysis quality
        analysis_quality = 1.0 if "synthesis" in state["current_analysis"] else 0.5
        
        return min(success_rate * analysis_quality, 1.0)