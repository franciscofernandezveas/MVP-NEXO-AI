# utils/cache.py
import asyncio
from typing import Callable, Any
from functools import wraps
from cachetools import TTLCache

# Cache global con TTL de 5 minutos
cache = TTLCache(maxsize=100, ttl=300)

# Tracking de requests en vuelo para deduplicación
in_flight = {}

# Lock para operaciones thread-safe
lock = asyncio.Lock()

# Métricas básicas
hits = 0
misses = 0

def get_cache_stats():
    """Obtener estadísticas del cache"""
    total = hits + misses
    hit_rate = hits / total if total > 0 else 0
    return {
        "hits": hits,
        "misses": misses,
        "hit_rate": hit_rate,
        "cache_size": len(cache),
        "in_flight": len(in_flight)
    }

def cacheable(key_fn: Callable[..., str]):
    """
    Decorator para añadir caching y deduplicación a funciones async.
    
    Args:
        key_fn: Función que genera la key del cache basada en los argumentos
               Ej: lambda: "sales_today" 
               Ej: lambda fecha_inicio, fecha_fin: f"sales_producto:{fecha_inicio}:{fecha_fin}"
    """
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            # Generar key usando la función proporcionada
            key = key_fn(*args, **kwargs)
            
            global hits, misses
            
            # 1. CACHE HIT - retorno inmediato
            if key in cache:
                hits += 1
                return cache[key]
            
            misses += 1
            
            async with lock:
                # 2. DOBLE CHECK - otro request pudo haber poblado el cache
                if key in cache:
                    hits += 1
                    return cache[key]
                
                # 3. REQUEST EN VUELO - esperar el resultado existente
                if key in in_flight:
                    return await in_flight[key]
                
                # 4. CREAR NUEVA TASK
                task = asyncio.create_task(fn(*args, **kwargs))
                in_flight[key] = task
            
            try:
                # 5. EJECUTAR Y CACHEAR
                result = await task
                cache[key] = result
                return result
                
            finally:
                # 6. LIMPIEZA - siempre remover del in_flight
                async with lock:
                    in_flight.pop(key, None)
                    
        return wrapper
    return decorator

# Helper para invalidación manual
def invalidate_cache(key: str = None):
    """Invalidar cache específico o todo el cache"""
    if key:
        cache.pop(key, None)
    else:
        cache.clear()

def invalidate_cache_pattern(pattern: str):
    """Invalidar todas las keys que contienen un patrón"""
    keys_to_remove = [k for k in cache.keys() if pattern in k]
    for key in keys_to_remove:
        cache.pop(key, None)
