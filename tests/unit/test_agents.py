"""Unit tests for AREIP agents."""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from areip.agents.base import BaseAgent, AgentTask, AgentResult, AgentStatus, AgentMemory
from areip.agents.discovery import MarketDiscoveryAgent
from areip.agents.validation import ValidationRiskAgent
from areip.utils.database import DatabaseManager


class MockAgent(BaseAgent):
    """Mock agent for testing base functionality."""
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Mock task execution."""
        if task.task_type == "failing_task":
            raise Exception("Mock task failure")
        
        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            success=True,
            output_data={"mock_result": "success"},
            confidence_score=0.8
        )


@pytest.fixture
def mock_db_manager():
    """Mock database manager."""
    db_manager = Mock(spec=DatabaseManager)
    db_manager.store_agent_execution = AsyncMock()
    return db_manager


@pytest.fixture
def mock_agent(mock_db_manager):
    """Create mock agent for testing."""
    return MockAgent("TestAgent", mock_db_manager)


@pytest.fixture
def sample_task():
    """Create sample task for testing."""
    return AgentTask(
        task_id="test_task_001",
        task_type="test_task",
        description="Test task for unit testing",
        input_data={"test_param": "test_value"}
    )


@pytest.mark.asyncio
class TestBaseAgent:
    """Test base agent functionality."""
    
    async def test_agent_initialization(self, mock_agent):
        """Test agent initialization."""
        assert mock_agent.agent_name == "TestAgent"
        assert mock_agent.agent_id.startswith("TestAgent_")
        assert mock_agent.status == AgentStatus.IDLE
        assert isinstance(mock_agent.memory, AgentMemory)
    
    async def test_successful_task_execution(self, mock_agent, sample_task):
        """Test successful task execution."""
        result = await mock_agent.run(sample_task)
        
        assert result.success is True
        assert result.task_id == sample_task.task_id
        assert result.agent_id == mock_agent.agent_id
        assert result.output_data == {"mock_result": "success"}
        assert result.confidence_score == 0.8
        assert result.execution_time_seconds > 0
    
    async def test_failed_task_execution(self, mock_agent):
        """Test failed task execution."""
        failing_task = AgentTask(
            task_id="fail_task_001",
            task_type="failing_task",
            description="Task that should fail",
            input_data={}
        )
        
        result = await mock_agent.run(failing_task)
        
        assert result.success is False
        assert result.error_message == "Mock task failure"
        assert result.output_data == {}
    
    async def test_memory_update(self, mock_agent, sample_task):
        """Test agent memory updates."""
        initial_memory_size = len(mock_agent.memory.short_term_memory)
        
        result = await mock_agent.run(sample_task)
        
        # Check that memory was updated
        assert len(mock_agent.memory.short_term_memory) == initial_memory_size + 1
        assert sample_task.task_id in mock_agent.memory.short_term_memory
        
        # Check episodic memory
        assert len(mock_agent.memory.episodic_memory) > 0
        
        # Check long-term memory
        assert "test_task" in mock_agent.memory.long_term_memory
        ltm = mock_agent.memory.long_term_memory["test_task"]
        assert ltm["execution_count"] == 1
        assert ltm["success_count"] == 1
    
    async def test_memory_summary(self, mock_agent, sample_task):
        """Test memory summary generation."""
        await mock_agent.run(sample_task)
        
        summary = await mock_agent.get_memory_summary()
        
        assert "agent_id" in summary
        assert "agent_name" in summary
        assert "status" in summary
        assert "performance_summary" in summary
        assert summary["short_term_tasks"] == 1
        assert summary["total_episodes"] == 1


@pytest.mark.asyncio
class TestMarketDiscoveryAgent:
    """Test Market Discovery Agent."""
    
    @pytest.fixture
    def discovery_agent(self, mock_db_manager):
        """Create discovery agent for testing."""
        with patch('areip.agents.discovery.ChatOpenAI'):
            agent = MarketDiscoveryAgent(mock_db_manager)
            return agent
    
    async def test_market_discovery_task(self, discovery_agent):
        """Test market discovery task execution."""
        task = AgentTask(
            task_id="market_discovery_001",
            task_type="market_discovery",
            description="Test market discovery",
            input_data={
                "focus_areas": ["market trends"],
                "time_period": "6m"
            }
        )
        
        with patch.object(discovery_agent, '_run_market_discovery') as mock_run:
            mock_run.return_value = {
                "analysis_type": "market_discovery",
                "crew_result": "Mock discovery results",
                "timestamp": datetime.now().isoformat()
            }
            
            result = await discovery_agent.execute_task(task)
            
            assert result.success is True
            assert result.output_data["analysis_type"] == "market_discovery"
            mock_run.assert_called_once()


@pytest.mark.asyncio
class TestValidationRiskAgent:
    """Test Validation & Risk Agent."""
    
    @pytest.fixture
    def validation_agent(self, mock_db_manager):
        """Create validation agent for testing."""
        with patch('areip.agents.validation.ChatOpenAI'):
            agent = ValidationRiskAgent(mock_db_manager)
            return agent
    
    async def test_risk_assessment_task(self, validation_agent):
        """Test risk assessment task execution."""
        task = AgentTask(
            task_id="risk_assessment_001",
            task_type="assess_risk",
            description="Test risk assessment",
            input_data={
                "risk_type": "market",
                "target_data": {}
            }
        )
        
        with patch.object(validation_agent, '_assess_risk') as mock_assess:
            mock_assess.return_value = {
                "risk_type": "market",
                "assessments": {},
                "overall_risk": {"score": 0.5, "level": "medium"}
            }
            
            result = await validation_agent.execute_task(task)
            
            assert result.success is True
            assert result.output_data["risk_type"] == "market"
            mock_assess.assert_called_once()
    
    async def test_trend_validation(self, validation_agent):
        """Test trend validation functionality."""
        # Mock database to return empty data
        validation_agent.db_manager.get_raw_data = AsyncMock(return_value=MockDataFrame())
        
        task = AgentTask(
            task_id="validate_trends_001",
            task_type="validate_trends",
            description="Test trend validation",
            input_data={}
        )
        
        result = await validation_agent.execute_task(task)
        
        assert result.success is True
        assert "validation_type" in result.output_data


class MockDataFrame:
    """Mock pandas DataFrame for testing."""
    
    def __init__(self):
        self.empty = True
        self.columns = []
    
    def __len__(self):
        return 0


@pytest.mark.asyncio
class TestAgentCoordination:
    """Test agent coordination and workflow management."""
    
    async def test_agent_registration(self, mock_db_manager):
        """Test agent registration with coordinator."""
        from areip.agents.orchestrator import IntelligenceOrchestratorAgent
        
        with patch('areip.agents.orchestrator.ChatOpenAI'):
            orchestrator = IntelligenceOrchestratorAgent(mock_db_manager)
            mock_agent = MockAgent("TestAgent", mock_db_manager)
            
            orchestrator.register_agent(mock_agent)
            
            assert mock_agent.agent_name in orchestrator.agent_registry
            assert orchestrator.agent_registry[mock_agent.agent_name] == mock_agent


if __name__ == "__main__":
    pytest.main([__file__])