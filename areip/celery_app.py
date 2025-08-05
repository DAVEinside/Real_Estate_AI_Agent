"""Celery application configuration for AREIP."""

import logging
from celery import Celery
from celery.schedules import crontab

from .config import settings

logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    "areip",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "areip.workers.data_ingestion",
        "areip.workers.model_training", 
        "areip.workers.agent_tasks",
        "areip.workers.analytics"
    ]
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    result_expires=3600,  # Results expire after 1 hour
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
)

# Periodic tasks schedule
celery_app.conf.beat_schedule = {
    # Data ingestion tasks
    'ingest-zillow-data': {
        'task': 'areip.workers.data_ingestion.ingest_zillow_data',
        'schedule': crontab(hour=1, minute=0),  # Daily at 1 AM
    },
    'ingest-fred-data': {
        'task': 'areip.workers.data_ingestion.ingest_fred_data',
        'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
    },
    'ingest-census-data': {
        'task': 'areip.workers.data_ingestion.ingest_census_data',
        'schedule': crontab(hour=3, minute=0, day_of_week=1),  # Weekly on Monday at 3 AM
    },
    
    # Model training tasks
    'train-price-models': {
        'task': 'areip.workers.model_training.train_price_prediction_models',
        'schedule': crontab(hour=4, minute=0, day_of_week=0),  # Weekly on Sunday at 4 AM
    },
    'train-market-models': {
        'task': 'areip.workers.model_training.train_market_trend_models',
        'schedule': crontab(hour=5, minute=0, day_of_week=0),  # Weekly on Sunday at 5 AM
    },
    
    # Analytics and reporting
    'generate-market-insights': {
        'task': 'areip.workers.analytics.generate_market_insights',
        'schedule': crontab(hour=6, minute=0),  # Daily at 6 AM
    },
    'update-risk-assessments': {
        'task': 'areip.workers.analytics.update_risk_assessments',
        'schedule': crontab(hour=7, minute=0),  # Daily at 7 AM
    },
    
    # Agent orchestration
    'run-agent-workflows': {
        'task': 'areip.workers.agent_tasks.run_scheduled_workflows',
        'schedule': crontab(minute='*/30'),  # Every 30 minutes
    },
    
    # Cleanup tasks
    'cleanup-old-data': {
        'task': 'areip.workers.data_ingestion.cleanup_old_data',
        'schedule': crontab(hour=23, minute=0),  # Daily at 11 PM
    },
}

# Task routing
celery_app.conf.task_routes = {
    'areip.workers.data_ingestion.*': {'queue': 'data_ingestion'},
    'areip.workers.model_training.*': {'queue': 'model_training'},
    'areip.workers.agent_tasks.*': {'queue': 'agent_tasks'},
    'areip.workers.analytics.*': {'queue': 'analytics'},
}

# Error handling
@celery_app.task(bind=True)
def debug_task(self):
    """Debug task for testing Celery setup."""
    print(f'Request: {self.request!r}')
    return 'Celery is working!'


if __name__ == '__main__':
    celery_app.start()