"""End-to-end tests for complete AREIP workflows."""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd

from areip.agents.coordinator import AgentCoordinator
from areip.data.ingestion import DataIngestionPipeline
from areip.utils.database import DatabaseManager


@pytest.fixture
def mock_db_manager():
    """Mock database manager for E2E testing."""
    db_manager = Mock(spec=DatabaseManager)
    db_manager.initialize_database = AsyncMock()
    db_manager.store_raw_data = AsyncMock()
    db_manager.store_agent_execution = AsyncMock()
    db_manager.get_raw_data = AsyncMock(return_value=pd.DataFrame({
        'RegionName': ['San Francisco', 'New York'],
        'date': ['2023-01-01', '2023-01-01'],
        'home_values_value': [1000000, 800000]
    }))
    db_manager.get_table_record_counts = AsyncMock(return_value={
        'zillow_data_home_values': 1000,
        'fred_data_mortgage_rates': 500
    })
    db_manager.close = AsyncMock()
    return db_manager


@pytest.fixture
def coordinator(mock_db_manager):
    """Create agent coordinator for testing."""
    with patch('areip.agents.orchestrator.ChatOpenAI'), \
         patch('areip.agents.discovery.ChatOpenAI'), \
         patch('areip.agents.synthesis.ChatOpenAI'), \
         patch('areip.agents.validation.ChatOpenAI'), \
         patch('areip.agents.synthesis.GraphRAGKnowledgeBase'):
        
        coord = AgentCoordinator(mock_db_manager)
        return coord


@pytest.mark.asyncio
class TestCompleteWorkflows:
    """Test complete end-to-end workflows."""
    
    async def test_market_analysis_workflow_creation(self, coordinator):
        """Test market analysis workflow creation."""
        workflow_id = await coordinator.create_market_analysis_workflow({
            "regions": ["San Francisco", "New York"], 
            "time_period": "6m",
            "include_properties": True,
            "include_market_data": True
        })
        
        assert workflow_id.startswith("market_analysis_")
        assert workflow_id in coordinator.active_workflows
        
        workflow = coordinator.active_workflows[workflow_id]
        assert workflow.name == "Market Analysis Workflow"
        assert len(workflow.tasks) == 5  # Expected number of tasks
        assert len(workflow.dependencies) > 0
    
    async def test_property_evaluation_workflow_creation(self, coordinator):
        """Test property evaluation workflow creation."""
        property_data = {
            "id": "prop_001",
            "address": "123 Test St",
            "price": 500000,
            "bedrooms": 3,
            "bathrooms": 2,
            "square_feet": 1500
        }
        
        workflow_id = await coordinator.create_property_evaluation_workflow(property_data)
        
        assert workflow_id.startswith("property_eval_")
        assert workflow_id in coordinator.active_workflows
        
        workflow = coordinator.active_workflows[workflow_id]
        assert workflow.name == "Property Evaluation Workflow"
        assert len(workflow.tasks) == 4  # Expected number of tasks
    
    async def test_workflow_execution_success(self, coordinator):
        """Test successful workflow execution."""
        # Create workflow
        workflow_id = await coordinator.create_market_analysis_workflow({
            "regions": ["San Francisco"],
            "time_period": "3m"
        })
        
        # Mock agent execution to return successful results
        for agent in coordinator.agents.values():
            agent.run = AsyncMock(return_value=Mock(
                success=True,
                task_id="mock_task",
                agent_id=agent.agent_id,
                output_data={"mock_analysis": "completed"},
                confidence_score=0.8,
                execution_time_seconds=1.0,
                error_message=None
            ))
        
        # Execute workflow
        result = await coordinator.execute_workflow(workflow_id)
        
        assert result["status"] == "completed"
        assert result["tasks_completed"] > 0
        assert "execution_time" in result
        assert "results" in result
    
    async def test_workflow_execution_failure(self, coordinator):
        """Test workflow execution with failures."""
        workflow_id = await coordinator.create_market_analysis_workflow({
            "regions": ["San Francisco"]
        })
        
        # Mock agent execution to fail
        for agent in coordinator.agents.values():
            agent.run = AsyncMock(side_effect=Exception("Mock agent failure"))
        
        result = await coordinator.execute_workflow(workflow_id)
        
        assert result["status"] == "failed"
        assert "error" in result
    
    async def test_workflow_status_tracking(self, coordinator):
        """Test workflow status tracking."""
        workflow_id = await coordinator.create_market_analysis_workflow({
            "regions": ["Test Region"]
        })
        
        # Check initial status
        status = await coordinator.get_workflow_status(workflow_id)
        assert status["status"] == "pending"
        assert status["progress"] == 0.0
        assert status["tasks_completed"] == 0
        
        # Simulate partial completion
        workflow = coordinator.active_workflows[workflow_id]
        workflow.results["task_1"] = Mock()
        
        status = await coordinator.get_workflow_status(workflow_id)
        assert status["progress"] > 0.0
        assert status["tasks_completed"] == 1
    
    async def test_agent_coordination(self, coordinator):
        """Test agent coordination and communication."""
        # Verify all agents are registered
        expected_agents = ["orchestrator", "discovery", "synthesis", "validation"]
        for agent_name in expected_agents:
            assert agent_name in coordinator.agents
        
        # Test agent status retrieval
        agent_status = await coordinator.get_agent_status()
        assert len(agent_status) == len(expected_agents)
        
        for agent_name, status in agent_status.items():
            assert "status" in status
            assert "memory_summary" in status


@pytest.mark.asyncio
class TestDataIntegration:
    """Test data integration with workflows."""
    
    async def test_data_ingestion_integration(self, mock_db_manager):
        """Test data ingestion integration with workflow execution."""
        pipeline = DataIngestionPipeline(mock_db_manager)
        
        # Mock data sources
        mock_data = pd.DataFrame({
            'RegionName': ['San Francisco', 'New York'],
            'date': ['2023-01-01', '2023-01-01'], 
            'home_values_value': [1000000, 800000]
        })
        
        for source in pipeline.sources.values():
            source.fetch_data = AsyncMock(return_value=mock_data)
            source.validate_data = Mock(return_value=True)
        
        # Run ingestion
        results = await pipeline.run_incremental_ingestion()
        
        # Verify successful ingestion
        successful_results = [r for r in results if r.success]
        assert len(successful_results) > 0
        
        # Verify data was stored
        assert mock_db_manager.store_raw_data.call_count > 0
    
    async def test_workflow_with_real_data_access(self, coordinator):
        """Test workflow execution with data access."""
        workflow_id = await coordinator.create_market_analysis_workflow({
            "regions": ["San Francisco"],
            "include_market_data": True
        })
        
        # Mock agents to simulate data access
        coordinator.synthesis_agent.execute_task = AsyncMock(return_value=Mock(
            success=True,
            task_id="synthesis_task",
            agent_id=coordinator.synthesis_agent.agent_id,
            output_data={
                "analysis_type": "data_ingestion",
                "entities_ingested": 100,
                "status": "completed"
            },
            confidence_score=0.9,
            execution_time_seconds=2.0
        ))
        
        coordinator.discovery_agent.execute_task = AsyncMock(return_value=Mock(
            success=True,
            task_id="discovery_task", 
            agent_id=coordinator.discovery_agent.agent_id,
            output_data={
                "analysis_type": "market_discovery",
                "crew_result": "Market analysis completed",
                "timestamp": "2023-01-01T00:00:00"
            },
            confidence_score=0.8,
            execution_time_seconds=3.0
        ))
        
        coordinator.validation_agent.execute_task = AsyncMock(return_value=Mock(
            success=True,
            task_id="validation_task",
            agent_id=coordinator.validation_agent.agent_id,
            output_data={
                "risk_type": "comprehensive",
                "overall_risk": {"score": 0.3, "level": "low"},
                "assessments": {}
            },
            confidence_score=0.85,
            execution_time_seconds=1.5
        ))
        
        coordinator.orchestrator.execute_task = AsyncMock(return_value=Mock(
            success=True,
            task_id="orchestration_task",
            agent_id=coordinator.orchestrator.agent_id,
            output_data={
                "completed_tasks": [],
                "decisions": [{"action": "invest", "confidence": 0.8}],
                "market_analysis": {"condition": "favorable"}
            },
            confidence_score=0.9,
            execution_time_seconds=2.5
        ))
        
        result = await coordinator.execute_workflow(workflow_id)
        
        assert result["status"] == "completed"
        assert result["tasks_completed"] == 5


@pytest.mark.asyncio
class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms."""
    
    async def test_partial_workflow_failure_recovery(self, coordinator):
        """Test recovery from partial workflow failures."""
        workflow_id = await coordinator.create_market_analysis_workflow({
            "regions": ["Test Region"]
        })
        
        # Mock some agents to succeed and others to fail
        coordinator.synthesis_agent.run = AsyncMock(return_value=Mock(
            success=True,
            output_data={"status": "completed"}
        ))
        
        coordinator.discovery_agent.run = AsyncMock(side_effect=Exception("Discovery failed"))
        
        result = await coordinator.execute_workflow(workflow_id)
        
        # Should fail due to dependency issues
        assert result["status"] == "failed"
        assert "error" in result
    
    async def test_agent_failure_isolation(self, coordinator):
        """Test that individual agent failures don't crash the system."""
        # Create a task that should fail
        from areip.agents.base import AgentTask
        
        failing_task = AgentTask(
            task_id="failing_task",
            task_type="unknown_task_type",
            description="Task that should fail",
            input_data={}
        )
        
        # Execute on discovery agent
        result = await coordinator.discovery_agent.run(failing_task)
        
        # Agent should handle failure gracefully
        assert result.success is False
        assert result.error_message is not None
        
        # Agent should remain functional
        assert coordinator.discovery_agent.status.value in ["idle", "failed"]
    
    async def test_database_connection_failure_handling(self, coordinator):
        """Test handling of database connection failures."""
        # Mock database failure
        coordinator.db_manager.store_agent_execution = AsyncMock(
            side_effect=Exception("Database connection failed")
        )
        
        from areip.agents.base import AgentTask
        
        test_task = AgentTask(
            task_id="db_test_task",
            task_type="test_task",
            description="Test database handling",
            input_data={}
        )
        
        # Should handle database errors gracefully
        result = await coordinator.discovery_agent.run(test_task)
        
        # The task execution should still complete despite DB issues
        assert result is not None


@pytest.mark.asyncio
class TestPerformanceAndScaling:
    """Test performance and scaling characteristics."""
    
    async def test_concurrent_workflow_execution(self, coordinator):
        """Test concurrent execution of multiple workflows."""
        workflow_ids = []
        
        # Create multiple workflows
        for i in range(3):
            workflow_id = await coordinator.create_market_analysis_workflow({
                "regions": [f"Region_{i}"],
                "time_period": "3m"
            })
            workflow_ids.append(workflow_id)
        
        # Mock successful execution
        for agent in coordinator.agents.values():
            agent.run = AsyncMock(return_value=Mock(
                success=True,
                output_data={"result": "success"},
                execution_time_seconds=0.5
            ))
        
        # Execute workflows concurrently
        tasks = [coordinator.execute_workflow(wf_id) for wf_id in workflow_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete successfully
        for result in results:
            assert not isinstance(result, Exception)
            assert result["status"] == "completed"
    
    async def test_memory_management(self, coordinator):
        """Test agent memory management under load."""
        from areip.agents.base import AgentTask
        
        agent = coordinator.discovery_agent
        
        # Execute many tasks to test memory management
        for i in range(15):  # More than the short-term memory limit
            task = AgentTask(
                task_id=f"memory_test_{i}",
                task_type="test_task",
                description=f"Memory test task {i}",
                input_data={"iteration": i}
            )
            
            # Mock successful execution
            agent.execute_task = AsyncMock(return_value={
                "task_result": f"completed_{i}"
            })
            
            await agent.run(task)
        
        # Check memory limits are respected
        assert len(agent.memory.short_term_memory) <= 10
        assert len(agent.memory.episodic_memory) <= 100


if __name__ == "__main__":
    pytest.main([__file__])