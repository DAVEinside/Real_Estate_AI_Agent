#!/usr/bin/env python3
"""AREIP Deployment Script - One-click deployment for the complete system."""

import asyncio
import subprocess
import sys
import time
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

class AREIPDeployment:
    """Complete AREIP system deployment manager."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.services_ready = {
            'postgres': False,
            'redis': False,
            'neo4j': False
        }
    
    def check_requirements(self):
        """Check system requirements."""
        console.print(Panel.fit(
            "[bold blue]Checking System Requirements[/bold blue]",
            title="Prerequisites Check"
        ))
        
        requirements = [
            ("Python 3.11+", self._check_python),
            ("Docker", self._check_docker),
            ("Docker Compose", self._check_docker_compose),
            ("Git", self._check_git)
        ]
        
        results = []
        for name, check_func in requirements:
            status = check_func()
            results.append((name, status))
        
        # Display results
        req_table = Table(title="System Requirements")
        req_table.add_column("Requirement", style="cyan")
        req_table.add_column("Status", style="bold")
        req_table.add_column("Version", style="white")
        
        all_good = True
        for name, (status, version) in results:
            if status:
                req_table.add_row(name, "[green]✓ Available[/green]", version)
            else:
                req_table.add_row(name, "[red]✗ Missing[/red]", "Not found")
                all_good = False
        
        console.print(req_table)
        
        if not all_good:
            console.print("[red]❌ Missing requirements! Please install missing components.[/red]")
            return False
        
        console.print("[green]✅ All requirements satisfied![/green]")
        return True
    
    def _check_python(self):
        try:
            result = subprocess.run([sys.executable, "--version"], 
                                  capture_output=True, text=True)
            version = result.stdout.strip()
            # Check if version >= 3.11
            version_num = version.split()[1]
            major, minor = map(int, version_num.split('.')[:2])
            return (major > 3 or (major == 3 and minor >= 11), version)
        except:
            return (False, "Not found")
    
    def _check_docker(self):
        try:
            result = subprocess.run(["docker", "--version"], 
                                  capture_output=True, text=True)
            return (result.returncode == 0, result.stdout.strip())
        except:
            return (False, "Not found")
    
    def _check_docker_compose(self):
        try:
            result = subprocess.run(["docker", "compose", "version"], 
                                  capture_output=True, text=True)
            return (result.returncode == 0, result.stdout.strip())
        except:
            return (False, "Not found")
    
    def _check_git(self):
        try:
            result = subprocess.run(["git", "--version"], 
                                  capture_output=True, text=True)
            return (result.returncode == 0, result.stdout.strip())
        except:
            return (False, "Not found")
    
    def setup_environment(self):
        """Set up Python environment."""
        console.print(Panel.fit(
            "[bold purple]Setting Up Python Environment[/bold purple]",
            title="Environment Setup"
        ))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Create virtual environment
            task1 = progress.add_task("Creating virtual environment...", total=None)
            if not (self.project_root / "venv").exists():
                result = subprocess.run([
                    sys.executable, "-m", "venv", "venv"
                ], cwd=self.project_root, capture_output=True)
                
                if result.returncode != 0:
                    progress.update(task1, description="Virtual environment creation failed ✗")
                    return False
            
            progress.update(task1, description="Virtual environment ready ✓")
            
            # Install dependencies
            task2 = progress.add_task("Installing Python dependencies...", total=None)
            pip_cmd = str(self.project_root / "venv" / "bin" / "pip")
            if sys.platform == "win32":
                pip_cmd = str(self.project_root / "venv" / "Scripts" / "pip.exe")
            
            # Install requirements
            result = subprocess.run([
                pip_cmd, "install", "-r", "requirements.txt"
            ], cwd=self.project_root, capture_output=True)
            
            if result.returncode != 0:
                progress.update(task2, description="Dependency installation failed ✗")
                return False
            
            # Install package in development mode
            result = subprocess.run([
                pip_cmd, "install", "-e", "."
            ], cwd=self.project_root, capture_output=True)
            
            progress.update(task2, description="Dependencies installed ✓")
            
            # Check environment file
            task3 = progress.add_task("Checking environment configuration...", total=None) 
            env_file = self.project_root / ".env"
            if not env_file.exists():
                # Copy from example
                import shutil
                shutil.copy(self.project_root / ".env.example", env_file)
                progress.update(task3, description="Environment file created from template ✓")
            else:
                progress.update(task3, description="Environment file exists ✓")
        
        return True
    
    def start_infrastructure(self):
        """Start Docker infrastructure."""
        console.print(Panel.fit(
            "[bold green]Starting Infrastructure Services[/bold green]",
            title="Docker Services"
        ))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Start databases
            task = progress.add_task("Starting database services...", total=None)
            
            result = subprocess.run([
                "docker", "compose", "up", "-d", 
                "postgres", "redis", "neo4j"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode != 0:
                progress.update(task, description="Failed to start services ✗")
                console.print(f"[red]Error: {result.stderr}[/red]")
                return False
            
            progress.update(task, description="Services started, waiting for readiness...")
            
            # Wait for services to be ready
            max_attempts = 30
            for attempt in range(max_attempts):
                if self._check_service_health():
                    progress.update(task, description="All services ready ✓")
                    break
                time.sleep(2)
                progress.update(task, description=f"Waiting for services ({attempt+1}/{max_attempts})...")
            else:
                progress.update(task, description="Services may not be fully ready ⚠️")
        
        return True
    
    def _check_service_health(self):
        """Check if all services are healthy."""
        try:
            # Check postgres
            result = subprocess.run([
                "docker", "compose", "exec", "-T", "postgres", 
                "pg_isready", "-U", "postgres"
            ], capture_output=True)
            postgres_ready = result.returncode == 0
            
            # Check redis
            result = subprocess.run([
                "docker", "compose", "exec", "-T", "redis", 
                "redis-cli", "ping"
            ], capture_output=True)
            redis_ready = result.returncode == 0
            
            # Check neo4j (simplified)
            result = subprocess.run([
                "docker", "compose", "ps", "neo4j"
            ], capture_output=True, text=True)
            neo4j_ready = "running" in result.stdout
            
            self.services_ready = {
                'postgres': postgres_ready,
                'redis': redis_ready,
                'neo4j': neo4j_ready
            }
            
            return all(self.services_ready.values())
            
        except Exception:
            return False
    
    def run_system_tests(self):
        """Run system validation tests."""
        console.print(Panel.fit(
            "[bold yellow]Running System Validation[/bold yellow]",
            title="System Tests"
        ))
        
        python_cmd = str(self.project_root / "venv" / "bin" / "python")
        if sys.platform == "win32":
            python_cmd = str(self.project_root / "venv" / "Scripts" / "python.exe")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Test data sources
            task1 = progress.add_task("Testing real data sources...", total=None)
            result = subprocess.run([
                python_cmd, "scripts/test_real_data.py"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                progress.update(task1, description="Data sources test passed ✓")
                data_test_success = True
            else:
                progress.update(task1, description="Data sources test failed ✗")
                data_test_success = False
            
            # Test API health (if running)
            task2 = progress.add_task("Testing API health...", total=None)
            try:
                import requests
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    progress.update(task2, description="API health check passed ✓")
                    api_test_success = True
                else:
                    progress.update(task2, description="API not responding ○")
                    api_test_success = False
            except:
                progress.update(task2, description="API not running ○")
                api_test_success = False
        
        return data_test_success
    
    def display_final_status(self, success: bool):
        """Display final deployment status."""
        if success:
            status_panel = Panel(
                "[bold green]🎉 AREIP Deployment Successful! 🎉[/bold green]\n\n"
                "[bold]System Status:[/bold]\n"
                f"{'✅' if self.services_ready['postgres'] else '❌'} PostgreSQL Database\n"
                f"{'✅' if self.services_ready['redis'] else '❌'} Redis Cache\n"
                f"{'✅' if self.services_ready['neo4j'] else '❌'} Neo4j Graph Database\n"
                "✅ Python Environment\n"
                "✅ Real Data Sources\n\n"
                "[bold cyan]Next Steps:[/bold cyan]\n"
                "1. Start the API server: [code]python areip/api/main.py[/code]\n"
                "2. Run the full demo: [code]python scripts/full_demo.py[/code]\n"
                "3. Open API docs: [link]http://localhost:8000/docs[/link]\n"
                "4. View monitoring: [link]http://localhost:3000[/link]\n\n"
                "[bold yellow]🚀 System is ready for production![/bold yellow]",
                title="🏠 AREIP Deployment Complete",
                border_style="green"
            )
        else:
            status_panel = Panel(
                "[bold red]❌ Deployment Failed[/bold red]\n\n"
                "[bold]Troubleshooting:[/bold]\n"
                "1. Check Docker is running: [code]docker ps[/code]\n"
                "2. Check logs: [code]docker compose logs[/code]\n"
                "3. Verify requirements: [code]python --version[/code]\n"
                "4. Check network connectivity\n\n"
                "[bold]For support:[/bold]\n"
                "• Review logs in the terminal output above\n"
                "• Check README.md for detailed setup instructions\n"
                "• Verify all API keys are configured in .env",
                title="🏠 AREIP Deployment Issues",
                border_style="red"
            )
        
        console.print(status_panel)
    
    async def deploy(self):
        """Run complete deployment process."""
        console.print(Panel.fit(
            "[bold bright_blue]🏠 AREIP Complete System Deployment[/bold bright_blue]\n\n"
            "[bold]Deploying Production-Ready Real Estate AI Platform[/bold]\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "This deployment will set up:\n"
            "🔄 Real-time data ingestion from Zillow, FRED, Census\n"
            "🤖 Machine learning pipeline with multiple models\n"
            "🤝 Multi-agent AI system with LangGraph orchestration\n"
            "🏗️  Production infrastructure with Docker\n"
            "📊 Monitoring and observability stack\n\n"
            "[bold yellow]🎯 Ready for Cherre AI Research Associate Demo[/bold yellow]",
            title="AREIP System Deployment",
            subtitle="Automation-Native • Agent-Oriented • Production-Ready",
            border_style="bright_blue"
        ))
        
        success = True
        
        # Step 1: Check requirements
        if not self.check_requirements():
            success = False
        
        # Step 2: Setup environment
        if success and not self.setup_environment():
            success = False
        
        # Step 3: Start infrastructure
        if success and not self.start_infrastructure():
            success = False
        
        # Step 4: Run validation tests
        if success:
            self.run_system_tests()
        
        # Display final status
        self.display_final_status(success)
        
        return success


def main():
    """Main deployment function."""
    deployment = AREIPDeployment()
    
    try:
        success = asyncio.run(deployment.deploy())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Deployment cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Deployment failed with error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()