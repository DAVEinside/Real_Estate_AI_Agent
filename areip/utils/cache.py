"""Caching utilities for AREIP."""

import asyncio
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union
import redis.asyncio as redis

from ..config import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Redis-based cache manager for AREIP."""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.default_ttl = settings.cache_ttl_seconds
        
    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8", 
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Cache manager initialized successfully")
            
        except Exception as e:
            logger.warning(f"Could not initialize cache: {e}")
            self.redis_client = None
    
    def _generate_key(self, prefix: str, data: Dict[str, Any]) -> str:
        """Generate cache key from data."""
        # Create deterministic hash from data
        data_str = json.dumps(data, sort_keys=True)
        data_hash = hashlib.md5(data_str.encode()).hexdigest()[:8]
        return f"areip:{prefix}:{data_hash}"
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from cache."""
        if not self.redis_client:
            return None
        
        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
            
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    async def set(
        self, 
        key: str, 
        value: Dict[str, Any], 
        ttl: Optional[int] = None
    ):
        """Set value in cache."""
        if not self.redis_client:
            return
        
        try:
            ttl = ttl or self.default_ttl
            await self.redis_client.setex(
                key, 
                ttl, 
                json.dumps(value, default=str)
            )
            
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    async def delete(self, key: str):
        """Delete value from cache."""
        if not self.redis_client:
            return
        
        try:
            await self.redis_client.delete(key)
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
    
    async def cache_analysis_result(
        self, 
        analysis_type: str, 
        parameters: Dict[str, Any], 
        result: Dict[str, Any],
        ttl: int = 3600
    ):
        """Cache analysis result."""
        key = self._generate_key(f"analysis:{analysis_type}", parameters)
        
        cache_value = {
            "result": result,
            "cached_at": datetime.now().isoformat(),
            "analysis_type": analysis_type,
            "parameters": parameters
        }
        
        await self.set(key, cache_value, ttl)
        logger.info(f"Cached analysis result: {key}")
    
    async def get_cached_analysis(
        self, 
        analysis_type: str, 
        parameters: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get cached analysis result."""
        key = self._generate_key(f"analysis:{analysis_type}", parameters)
        
        cached = await self.get(key)
        if cached:
            logger.info(f"Found cached analysis: {key}")
            return cached
        
        return None
    
    async def cache_data_fetch(
        self, 
        source: str, 
        query_params: Dict[str, Any], 
        data: Dict[str, Any],
        ttl: int = 1800
    ):
        """Cache data fetch result."""
        key = self._generate_key(f"data:{source}", query_params)
        
        cache_value = {
            "data": data,
            "cached_at": datetime.now().isoformat(),
            "source": source,
            "query_params": query_params
        }
        
        await self.set(key, cache_value, ttl)
    
    async def get_cached_data(
        self, 
        source: str, 
        query_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get cached data fetch result."""
        key = self._generate_key(f"data:{source}", query_params)
        return await self.get(key)
    
    async def cache_agent_memory(
        self, 
        agent_id: str, 
        memory_data: Dict[str, Any]
    ):
        """Cache agent memory state."""
        key = f"areip:agent_memory:{agent_id}"
        await self.set(key, memory_data, ttl=7200)  # 2 hours
    
    async def get_agent_memory(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get cached agent memory."""
        key = f"areip:agent_memory:{agent_id}"
        return await self.get(key)
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern."""
        if not self.redis_client:
            return
        
        try:
            keys = await self.redis_client.keys(f"areip:{pattern}*")
            if keys:
                await self.redis_client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache entries matching {pattern}")
                
        except Exception as e:
            logger.warning(f"Cache invalidation error: {e}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.redis_client:
            return {"status": "disabled"}
        
        try:
            info = await self.redis_client.info()
            keys = await self.redis_client.keys("areip:*")
            
            # Count by prefix
            key_counts = {}
            for key in keys:
                prefix = key.split(":")[1] if ":" in key else "unknown"
                key_counts[prefix] = key_counts.get(prefix, 0) + 1
            
            return {
                "status": "active",
                "total_keys": len(keys),
                "key_counts_by_type": key_counts,
                "memory_usage": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"status": "error", "error": str(e)}
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Cache manager closed")