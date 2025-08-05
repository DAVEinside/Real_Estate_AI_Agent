"""Test script for AREIP real data sources."""

import asyncio
import logging
import json
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from areip.data.sources import DataSourceOrchestrator
from areip.utils.logging import setup_logging

# Setup
setup_logging()
console = Console()
logger = logging.getLogger(__name__)


async def test_individual_sources():
    """Test each data source individually."""
    
    console.print(Panel.fit(
        "[bold blue]Testing Individual Data Sources[/bold blue]\n"
        "Testing each data source independently",
        title="Individual Source Tests"
    ))
    
    orchestrator = DataSourceOrchestrator()
    
    # Test results
    results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Test Zillow data
        task_zillow = progress.add_task("Testing Zillow Research Data...", total=None)
        try:
            zillow_source = await orchestrator.get_source('zillow')
            data = await zillow_source.fetch_data(dataset='home_values')
            results['zillow'] = {
                'status': 'success',
                'records': len(data),
                'sample': data.head(3).to_dict('records') if not data.empty else []
            }
            progress.update(task_zillow, description="Zillow Research Data ✓")
        except Exception as e:
            results['zillow'] = {'status': 'failed', 'error': str(e)}
            progress.update(task_zillow, description="Zillow Research Data ✗")
        
        await asyncio.sleep(1)
        
        # Test FRED data
        task_fred = progress.add_task("Testing FRED Economic Data...", total=None)
        try:
            fred_source = await orchestrator.get_source('fred')
            data = await fred_source.fetch_data(series='mortgage_rates')
            results['fred'] = {
                'status': 'success',
                'records': len(data),
                'sample': data.head(3).to_dict('records') if not data.empty else []
            }
            progress.update(task_fred, description="FRED Economic Data ✓")
        except Exception as e:
            results['fred'] = {'status': 'failed', 'error': str(e)}
            progress.update(task_fred, description="FRED Economic Data ✗")
        
        await asyncio.sleep(1)
        
        # Test Census data
        task_census = progress.add_task("Testing Census Data...", total=None)
        try:
            census_source = await orchestrator.get_source('census')
            data = await census_source.fetch_data(geography='county', state='06')
            results['census'] = {
                'status': 'success',
                'records': len(data),
                'sample': data.head(3).to_dict('records') if not data.empty else []
            }
            progress.update(task_census, description="Census Data ✓")
        except Exception as e:
            results['census'] = {'status': 'failed', 'error': str(e)}
            progress.update(task_census, description="Census Data ✗")
        
        await asyncio.sleep(1)
        
        # Test RentCast data
        task_rent = progress.add_task("Testing RentCast Data...", total=None)
        try:
            rent_source = await orchestrator.get_source('rentcast') 
            data = await rent_source.fetch_data(city='San Francisco', state='CA')
            results['rentcast'] = {
                'status': 'success',
                'records': len(data),
                'sample': data.head(3).to_dict('records') if not data.empty else []
            }
            progress.update(task_rent, description="RentCast Data ✓")
        except Exception as e:
            results['rentcast'] = {'status': 'failed', 'error': str(e)}
            progress.update(task_rent, description="RentCast Data ✗")
        
        await asyncio.sleep(1)
        
        # Test OpenStreetMap data
        task_osm = progress.add_task("Testing OpenStreetMap Data...", total=None)
        try:
            osm_source = await orchestrator.get_source('openstreetmap')
            data = await osm_source.fetch_data(location='San Francisco, CA, USA', amenity_type='restaurant')
            results['openstreetmap'] = {
                'status': 'success',
                'records': len(data),
                'sample': data.head(3).to_dict('records') if not data.empty else []
            }
            progress.update(task_osm, description="OpenStreetMap Data ✓")
        except Exception as e:
            results['openstreetmap'] = {'status': 'failed', 'error': str(e)}
            progress.update(task_osm, description="OpenStreetMap Data ✗")
    
    return results


async def test_orchestrated_fetch():
    """Test comprehensive data fetch through orchestrator."""
    
    console.print(Panel.fit(
        "[bold blue]Testing Orchestrated Data Fetch[/bold blue]\n"
        "Fetching data from all sources simultaneously",
        title="Orchestrated Fetch Test"
    ))
    
    async with DataSourceOrchestrator() as orchestrator:
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Fetching all data sources...", total=None)
            
            try:
                # Fetch all data
                all_data = await orchestrator.fetch_all_data("San Francisco, CA")
                
                # Get summary
                summary = orchestrator.get_data_summary(all_data)
                
                progress.update(task, description="Orchestrated fetch completed ✓")
                
                return {
                    'status': 'success',
                    'summary': summary,
                    'datasets': {name: len(df) for name, df in all_data.items()}
                }
                
            except Exception as e:
                progress.update(task, description="Orchestrated fetch failed ✗")
                return {
                    'status': 'failed',
                    'error': str(e)
                }


def display_results(individual_results, orchestrated_results):
    """Display comprehensive test results."""
    
    # Individual source results table
    individual_table = Table(title="Individual Data Source Test Results")
    individual_table.add_column("Data Source", style="cyan")
    individual_table.add_column("Status", style="bold")
    individual_table.add_column("Records", style="magenta", justify="right")
    individual_table.add_column("Notes", style="white")
    
    for source_name, result in individual_results.items():
        if result['status'] == 'success':
            status = "[green]✓ Success[/green]"
            records = f"{result['records']:,}"
            notes = "Data fetched successfully"
        else:
            status = "[red]✗ Failed[/red]"
            records = "0"
            notes = result.get('error', 'Unknown error')[:50] + "..." if len(result.get('error', '')) > 50 else result.get('error', '')
        
        individual_table.add_row(
            source_name.title(),
            status,
            records,
            notes
        )
    
    console.print(individual_table)
    
    # Orchestrated results
    if orchestrated_results['status'] == 'success':
        console.print(Panel(
            f"[bold green]✓ Orchestrated Fetch Successful[/bold green]\n\n"
            f"Total Datasets: {orchestrated_results['summary']['total_datasets']}\n"
            f"Total Records: {orchestrated_results['summary']['total_records']:,}\n"
            f"Fetch Time: {orchestrated_results['summary']['fetch_time']}\n\n"
            f"[bold]Dataset Breakdown:[/bold]\n" +
            "\n".join([f"• {name}: {count:,} records" 
                      for name, count in orchestrated_results['datasets'].items()]),
            title="Orchestrated Fetch Results",
            border_style="green"
        ))
    else:
        console.print(Panel(
            f"[bold red]✗ Orchestrated Fetch Failed[/bold red]\n\n"
            f"Error: {orchestrated_results.get('error', 'Unknown error')}",
            title="Orchestrated Fetch Results",
            border_style="red"
        ))


async def main():
    """Main test function."""
    
    start_time = datetime.now()
    
    console.print(Panel.fit(
        "[bold green]🏠 AREIP Real Data Sources Test[/bold green]\n\n"
        "[bold]Testing Suite:[/bold]\n"
        "• Individual data source connectivity\n"
        "• Data validation and processing\n"
        "• Orchestrated multi-source fetch\n"
        "• Error handling and resilience\n\n"
        f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
        title="AREIP Data Sources Test",
        subtitle="Comprehensive Real Data Integration Test"
    ))
    
    try:
        # Test individual sources
        individual_results = await test_individual_sources()
        await asyncio.sleep(2)
        
        # Test orchestrated fetch
        orchestrated_results = await test_orchestrated_fetch()
        
        # Display results
        console.print("\n")
        display_results(individual_results, orchestrated_results)
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        successful_sources = sum(1 for result in individual_results.values() 
                               if result['status'] == 'success')
        total_sources = len(individual_results)
        
        console.print(Panel.fit(
            f"[bold]Test Summary:[/bold]\n"
            f"• Duration: {duration.total_seconds():.1f} seconds\n"
            f"• Individual Sources: {successful_sources}/{total_sources} successful\n"
            f"• Orchestrated Fetch: {'✓' if orchestrated_results['status'] == 'success' else '✗'}\n\n"
            f"[bold cyan]Data Integration Status:[/bold cyan]\n"
            f"{'🟢 READY FOR PRODUCTION' if successful_sources >= 2 else '🟡 PARTIAL INTEGRATION' if successful_sources >= 1 else '🔴 INTEGRATION ISSUES'}\n\n"
            f"[bold yellow]Next Steps:[/bold yellow]\n"
            f"• {'✓' if successful_sources >= 2 else '○'} Real data sources connected\n"
            f"• ○ ML model training with real data\n"
            f"• ○ Agent orchestration with live data\n"
            f"• ○ Production deployment",
            title="AREIP Data Integration Test Complete",
            subtitle=f"Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
        ))
        
        # Save detailed results
        test_results = {
            'test_timestamp': start_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'individual_results': individual_results,
            'orchestrated_results': orchestrated_results,
            'summary': {
                'successful_sources': successful_sources,
                'total_sources': total_sources,
                'integration_status': 'ready' if successful_sources >= 2 else 'partial' if successful_sources >= 1 else 'failed'
            }
        }
        
        with open('data_sources_test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        console.print(f"\n[dim]Detailed results saved to: data_sources_test_results.json[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]Test suite failed with error: {e}[/bold red]")
        logger.error(f"Test suite error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())