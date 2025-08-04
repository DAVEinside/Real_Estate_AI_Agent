"""Base classes for AREIP agents."""

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from langfuse import Langfuse
from langfuse.decorators import observe

from ..config import settings
from ..utils.database import DatabaseManager

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class AgentMemory:
    """Agent memory structure for state persistence."""
    agent_id: str
    short_term_memory: Dict[str, Any] = field(default_factory=dict)
    long_term_memory: Dict[str, Any] = field(default_factory=dict)
    episodic_memory: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class AgentTask:
    """Task structure for agent execution."""
    task_id: str
    task_type: str
    description: str
    input_data: Dict[str, Any]
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result structure for agent execution."""
    task_id: str
    agent_id: str
    success: bool
    output_data: Dict[str, Any]
    error_message: Optional[str] = None
    execution_time_seconds: float = 0.0
    confidence_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class BaseAgent(ABC):
    """Base class for all AREIP agents."""
    
    def __init__(
        self,
        agent_name: str,
        db_manager: DatabaseManager,
        langfuse_client: Optional[Langfuse] = None
    ):
        self.agent_name = agent_name
        self.agent_id = f"{agent_name}_{uuid.uuid4().hex[:8]}"
        self.db_manager = db_manager
        self.langfuse = langfuse_client
        self.status = AgentStatus.IDLE
        self.memory = AgentMemory(agent_id=self.agent_id)
        self.current_task: Optional[AgentTask] = None
        
        logger.info(f"Initialized agent {self.agent_id}")
    
    @abstractmethod
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute a specific task."""
        pass
    
    @observe(name="agent_execution")
    async def run(self, task: AgentTask) -> AgentResult:
        """Main execution method with observability."""
        start_time = datetime.now()
        execution_id = f"{self.agent_id}_{task.task_id}"
        
        logger.info(f"Agent {self.agent_id} starting task {task.task_id}")
        
        try:
            self.status = AgentStatus.RUNNING
            self.current_task = task
            
            # Store execution start in database
            await self.db_manager.store_agent_execution(
                agent_name=self.agent_name,
                execution_id=execution_id,
                status="started",
                input_data=task.__dict__,
                start_time=start_time
            )
            
            # Execute the task
            result = await self.execute_task(task)
            
            # Update memory
            self._update_memory(task, result)
            
            # Calculate execution time
            end_time = datetime.now()
            result.execution_time_seconds = (end_time - start_time).total_seconds()
            
            self.status = AgentStatus.COMPLETED if result.success else AgentStatus.FAILED
            
            # Store execution result
            await self.db_manager.store_agent_execution(
                agent_name=self.agent_name,
                execution_id=execution_id,
                status="completed" if result.success else "failed",
                output_data=result.__dict__,
                error_message=result.error_message,
                end_time=end_time
            )
            
            logger.info(f"Agent {self.agent_id} completed task {task.task_id} in {result.execution_time_seconds:.2f}s")
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            self.status = AgentStatus.FAILED
            error_message = str(e)
            
            logger.error(f"Agent {self.agent_id} failed task {task.task_id}: {error_message}")
            
            result = AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=False,
                output_data={},
                error_message=error_message,
                execution_time_seconds=execution_time
            )
            
            # Store failure
            await self.db_manager.store_agent_execution(
                agent_name=self.agent_name,
                execution_id=execution_id,
                status="failed",
                error_message=error_message,
                end_time=end_time
            )
            
            return result
        
        finally:
            self.current_task = None
    
    def _update_memory(self, task: AgentTask, result: AgentResult):
        """Update agent memory with task execution information."""
        # Update short-term memory (recent tasks)
        self.memory.short_term_memory[task.task_id] = {
            'task': task.__dict__,
            'result': result.__dict__,
            'timestamp': datetime.now()
        }
        
        # Keep only last 10 tasks in short-term memory
        if len(self.memory.short_term_memory) > 10:
            oldest_key = min(
                self.memory.short_term_memory.keys(),
                key=lambda k: self.memory.short_term_memory[k]['timestamp']
            )
            del self.memory.short_term_memory[oldest_key]
        
        # Update episodic memory (learning experiences)
        episode = {
            'task_type': task.task_type,
            'success': result.success,
            'execution_time': result.execution_time_seconds,
            'confidence_score': result.confidence_score,
            'timestamp': datetime.now(),
            'patterns': self._extract_patterns(task, result)
        }
        
        self.memory.episodic_memory.append(episode)
        
        # Keep only last 100 episodes
        if len(self.memory.episodic_memory) > 100:
            self.memory.episodic_memory = self.memory.episodic_memory[-100:]
        
        # Update long-term memory (learned patterns and strategies)
        self._update_long_term_memory(task, result)
        
        self.memory.last_updated = datetime.now()
    
    def _extract_patterns(self, task: AgentTask, result: AgentResult) -> Dict[str, Any]:
        """Extract patterns from task execution for learning."""
        patterns = {
            'task_complexity': len(str(task.input_data)),
            'success_rate': float(result.success),
            'efficiency': 1.0 / max(result.execution_time_seconds, 0.1)
        }
        
        if result.confidence_score:
            patterns['confidence'] = result.confidence_score
            
        return patterns
    
    def _update_long_term_memory(self, task: AgentTask, result: AgentResult):
        """Update long-term memory with learning insights."""
        task_type = task.task_type
        
        if task_type not in self.memory.long_term_memory:
            self.memory.long_term_memory[task_type] = {
                'execution_count': 0,
                'success_count': 0,
                'average_execution_time': 0.0,
                'learned_strategies': [],
                'common_errors': []
            }
        
        ltm = self.memory.long_term_memory[task_type]
        ltm['execution_count'] += 1
        
        if result.success:
            ltm['success_count'] += 1
        else:
            if result.error_message and result.error_message not in ltm['common_errors']:
                ltm['common_errors'].append(result.error_message)
                # Keep only top 5 errors
                ltm['common_errors'] = ltm['common_errors'][-5:]
        
        # Update average execution time
        ltm['average_execution_time'] = (
            (ltm['average_execution_time'] * (ltm['execution_count'] - 1) + 
             result.execution_time_seconds) / ltm['execution_count']
        )
    
    async def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of agent memory and learning."""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'status': self.status.value,
            'short_term_tasks': len(self.memory.short_term_memory),
            'total_episodes': len(self.memory.episodic_memory),
            'long_term_patterns': len(self.memory.long_term_memory),
            'last_updated': self.memory.last_updated,
            'performance_summary': self._get_performance_summary()
        }
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from memory."""
        if not self.memory.episodic_memory:
            return {}
        
        recent_episodes = self.memory.episodic_memory[-20:]  # Last 20 episodes
        
        success_rate = sum(1 for ep in recent_episodes if ep['success']) / len(recent_episodes)
        avg_execution_time = sum(ep['execution_time'] for ep in recent_episodes) / len(recent_episodes)
        avg_confidence = sum(
            ep.get('confidence_score', 0) for ep in recent_episodes 
            if ep.get('confidence_score')
        ) / max(1, sum(1 for ep in recent_episodes if ep.get('confidence_score')))
        
        return {
            'recent_success_rate': success_rate,
            'average_execution_time': avg_execution_time,
            'average_confidence': avg_confidence,
            'total_task_types': len(self.memory.long_term_memory)
        }
    
    async def save_memory(self):
        """Persist agent memory to database."""
        # This could be implemented to save memory state to database
        # For now, memory is kept in-process
        pass
    
    async def load_memory(self):
        """Load agent memory from database."""
        # This could be implemented to restore memory state from database
        # For now, memory starts fresh each session
        pass