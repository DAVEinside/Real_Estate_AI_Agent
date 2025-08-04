"""Main entry point for AREIP application."""

import asyncio
import logging
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from areip.agents.coordinator import AgentCoordinator
from areip.data.ingestion import DataIngestionPipeline
from areip.utils.database import DatabaseManager
from areip.utils.logging import setup_logging
from areip.config import settings

# Setup logging and console
setup_logging()
logger = logging.getLogger(__name__)
console = Console()


@click.group()
def cli():
    """Autonomous Real Estate Intelligence Platform (AREIP) CLI."""
    pass


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def serve(host: str, port: int, reload: bool):
    """Start the AREIP API server."""
    import uvicorn
    
    console.print(Panel.fit(
        "[bold green]Starting AREIP API Server[/bold green]\n"
        f"Host: {host}:{port}\n"
        f"Environment: {settings.environment}\n"
        f"Reload: {reload}",
        title="AREIP"
    ))
    
    uvicorn.run(
        "areip.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=settings.log_level.lower()
    )


@cli.command() 
@click.option('--full', is_flag=True, help='Run full data ingestion')
def ingest(full: bool):
    """Run data ingestion from all sources."""
    asyncio.run(_run_ingestion(full))


async def _run_ingestion(full: bool):
    """Run data ingestion pipeline."""
    console.print("[yellow]Initializing data ingestion...[/yellow]")
    
    db_manager = DatabaseManager()
    try:
        await db_manager.initialize_database()
        pipeline = DataIngestionPipeline(db_manager)
        
        console.print(f"[green]Starting {'full' if full else 'incremental'} data ingestion...[/green]")
        
        if full:
            results = await pipeline.run_full_ingestion()
        else:
            results = await pipeline.run_incremental_ingestion()
        
        # Display results
        table = Table(title="Data Ingestion Results")
        table.add_column("Source", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Records", justify="right", style="green")
        table.add_column("Time (s)", justify="right", style="blue")
        
        for result in results:
            status = "✓ Success" if result.success else "✗ Failed"
            table.add_row(
                result.source_name,
                status,
                str(result.records_processed),
                f"{result.execution_time_seconds:.2f}"
            )
        
        console.print(table)
        
        successful = sum(1 for r in results if r.success)
        total_records = sum(r.records_processed for r in results if r.success)
        
        console.print(f"\n[green]Completed: {successful}/{len(results)} sources successful, {total_records} total records[/green]")
        
    except Exception as e:
        console.print(f"[red]Ingestion failed: {e}[/red]")
        logger.error(f"Data ingestion failed: {e}")
    finally:
        await db_manager.close()


@cli.command()
@click.argument('analysis_type', type=click.Choice(['market', 'property']))
@click.option('--regions', multiple=True, help='Geographic regions to analyze')
@click.option('--time-period', default='6m', help='Time period for analysis')
def analyze(analysis_type: str, regions: tuple, time_period: str):
    """Run analysis using the multi-agent system."""
    asyncio.run(_run_analysis(analysis_type, list(regions), time_period))


async def _run_analysis(analysis_type: str, regions: list, time_period: str):
    """Run analysis workflow."""
    console.print(f"[yellow]Starting {analysis_type} analysis...[/yellow]")
    
    db_manager = DatabaseManager()
    try:
        await db_manager.initialize_database()
        coordinator = AgentCoordinator(db_manager)
        
        if analysis_type == 'market':
            # Create market analysis workflow
            workflow_id = await coordinator.create_market_analysis_workflow({
                "regions": regions,
                "time_period": time_period,
                "include_properties": True,
                "include_market_data": True,
                "focus_areas": ["market trends", "pricing patterns", "investment opportunities"]
            })
        else:
            # Property analysis requires property data
            console.print("[red]Property analysis requires property data. Use the API for property-specific analysis.[/red]")
            return
        
        console.print(f"[green]Created workflow: {workflow_id}[/green]")
        console.print("[yellow]Executing workflow...[/yellow]")
        
        # Execute workflow
        result = await coordinator.execute_workflow(workflow_id)
        
        if result["status"] == "completed":
            console.print(Panel.fit(
                f"[bold green]Analysis Completed Successfully[/bold green]\n"
                f"Workflow ID: {workflow_id}\n"
                f"Execution Time: {result.get('execution_time', 0):.2f}s\n"
                f"Tasks Completed: {result.get('tasks_completed', 0)}",
                title="Success"
            ))
            
            # Display summary if available
            if "summary" in result:
                console.print("\n[bold]Analysis Summary:[/bold]")
                console.print(result["summary"])
        else:
            console.print(f"[red]Analysis failed: {result.get('error', 'Unknown error')}[/red]")
        
    except Exception as e:
        console.print(f"[red]Analysis failed: {e}[/red]")
        logger.error(f"Analysis failed: {e}")
    finally:
        await coordinator.close()


@cli.command()
def status():
    """Show system status and health."""
    asyncio.run(_show_status())


async def _show_status():
    """Show system status."""
    console.print("[yellow]Checking system status...[/yellow]")
    
    db_manager = DatabaseManager()
    try:
        await db_manager.initialize_database()
        coordinator = AgentCoordinator(db_manager)
        
        # Get system status
        agent_status = await coordinator.get_agent_status()
        workflows = await coordinator.list_active_workflows()
        
        # Display agent status
        table = Table(title="Agent Status")
        table.add_column("Agent", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Current Task", style="green")
        table.add_column("Success Rate", justify="right", style="blue")
        
        for name, status in agent_status.items():
            success_rate = status["memory_summary"].get("performance_summary", {}).get("recent_success_rate", 0.0)
            table.add_row(
                name,
                status["status"],
                status["current_task"] or "None",
                f"{success_rate:.1%}"
            )
        
        console.print(table)
        
        # Display workflow status
        if workflows:
            workflow_table = Table(title="Active Workflows")
            workflow_table.add_column("Workflow ID", style="cyan")
            workflow_table.add_column("Name", style="magenta")
            workflow_table.add_column("Status", style="green")
            workflow_table.add_column("Progress", justify="right", style="blue")
            
            for workflow in workflows:
                workflow_table.add_row(
                    workflow["workflow_id"][:12] + "...",
                    workflow["name"],
                    workflow["status"],
                    f"{workflow['progress']:.1f}%"
                )
            
            console.print(workflow_table)
        else:
            console.print("[yellow]No active workflows[/yellow]")
        
        # Get data status
        pipeline = DataIngestionPipeline(db_manager)
        data_status = await pipeline.get_ingestion_status()
        
        console.print(f"\n[green]Total data records: {data_status.get('total_records', 0)}[/green]")
        console.print(f"[green]Available data sources: {len(data_status.get('sources', {}))}[/green]")
        
    except Exception as e:
        console.print(f"[red]Status check failed: {e}[/red]")
        logger.error(f"Status check failed: {e}")
    finally:
        await coordinator.close()


@cli.command()
def demo():
    """Run a demo analysis to showcase AREIP capabilities."""
    asyncio.run(_run_demo())


async def _run_demo():
    """Run demonstration workflow."""
    console.print(Panel.fit(
        "[bold blue]AREIP Demonstration[/bold blue]\n"
        "This demo will showcase the autonomous real estate intelligence platform\n"
        "by running a comprehensive market analysis workflow.",
        title="Demo"
    ))
    
    db_manager = DatabaseManager()
    try:
        await db_manager.initialize_database()
        
        # First, run data ingestion
        console.print("\n[yellow]Step 1: Data Ingestion[/yellow]")
        pipeline = DataIngestionPipeline(db_manager)
        ingestion_results = await pipeline.run_full_ingestion()
        
        successful_ingestion = sum(1 for r in ingestion_results if r.success)
        console.print(f"[green]Ingested data from {successful_ingestion} sources[/green]")
        
        # Create and run analysis
        console.print("\n[yellow]Step 2: Multi-Agent Analysis[/yellow]")
        coordinator = AgentCoordinator(db_manager)
        
        workflow_id = await coordinator.create_market_analysis_workflow({
            "regions": ["San Francisco", "New York", "Los Angeles"],
            "time_period": "12m",
            "include_properties": True,
            "include_market_data": True,
            "focus_areas": ["market trends", "investment opportunities", "risk assessment"],
            "analysis_depth": "comprehensive"
        })
        
        console.print(f"[green]Created workflow: {workflow_id}[/green]")
        
        # Execute with progress updates
        result = await coordinator.execute_workflow(workflow_id)
        
        if result["status"] == "completed":
            console.print(Panel.fit(
                "[bold green]Demo Completed Successfully![/bold green]\n"
                f"✓ Data ingested from {successful_ingestion} sources\n"
                f"✓ {result.get('tasks_completed', 0)} analysis tasks completed\n"
                f"✓ Execution time: {result.get('execution_time', 0):.2f} seconds\n"
                "✓ Multi-agent coordination successful",
                title="Demo Results"
            ))
            
            console.print("\n[bold]Key Capabilities Demonstrated:[/bold]")
            console.print("• [green]Real-time data ingestion from multiple sources[/green]")
            console.print("• [green]Multi-agent coordination and workflow management[/green]")
            console.print("• [green]Graph-RAG knowledge synthesis[/green]")
            console.print("• [green]Statistical validation and risk assessment[/green]")
            console.print("• [green]Autonomous market intelligence generation[/green]")
        else:
            console.print(f"[red]Demo failed: {result.get('error', 'Unknown error')}[/red]")
        
    except Exception as e:
        console.print(f"[red]Demo failed: {e}[/red]")
        logger.error(f"Demo failed: {e}")
    finally:
        await coordinator.close()


if __name__ == "__main__":
    cli()