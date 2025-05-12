from typing import Dict, Any, Optional, List
import aiohttp
import json
import redis
from urllib.parse import urlparse
import validators
from datetime import datetime
import os
import asyncio

class FirecrawlClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        redis_url: Optional[str] = None,
        cache_ttl: int = 3600  # 1 hour cache
    ):
        self.base_url = base_url or os.getenv("FIRECRAWL_API_URL", "http://localhost:3002")
        self.cache_ttl = cache_ttl
        self.redis_client = redis.from_url(redis_url) if redis_url else None

    def _get_cache_key(self, url: str) -> str:
        """Generate a cache key for a URL."""
        return f"firecrawl:content:{url}"

    async def _get_from_cache(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached content for a URL."""
        if not self.redis_client:
            return None
        
        cached = self.redis_client.get(self._get_cache_key(url))
        if cached:
            return json.loads(cached)
        return None

    async def _save_to_cache(self, url: str, content: Dict[str, Any]) -> None:
        """Save content to cache."""
        if not self.redis_client:
            return
        
        self.redis_client.setex(
            self._get_cache_key(url),
            self.cache_ttl,
            json.dumps(content)
        )

    def validate_url(self, url: str) -> bool:
        """Validate if a URL is well-formed and allowed."""
        if not validators.url(url):
            return False
        
        parsed = urlparse(url)
        # Add any additional validation rules here
        # For example, only allow certain domains or protocols
        return parsed.scheme in ['http', 'https']

    async def scrape_url(self, url: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Scrape content from a URL using Firecrawl.
        
        Args:
            url: The URL to scrape
            force_refresh: If True, ignore cache and fetch fresh content
            
        Returns:
            Dict containing scraped content and metadata
        """
        if not self.validate_url(url):
            raise ValueError(f"Invalid URL: {url}")

        # Check cache first
        if not force_refresh:
            cached = await self._get_from_cache(url)
            if cached:
                return cached

        # Prepare headers
        headers = {"Content-Type": "application/json"}

        # Make request to Firecrawl
        scrape_endpoint = f"{self.base_url}/v1/scrape"
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    scrape_endpoint,
                    json={"url": url},
                    headers=headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        # Try to parse error if JSON, otherwise return text
                        try:
                            error_json = json.loads(error_text)
                            error_message = error_json.get("error", error_text)
                        except json.JSONDecodeError:
                            error_message = error_text
                        raise Exception(f"Firecrawl API error ({response.status}): {error_message}")
                    
                    result = await response.json()
                    
                    # Ensure 'data' key exists, which contains 'content'
                    if 'data' not in result or 'content' not in result['data']:
                        # If the expected structure isn't there, but it was a 200, 
                        # wrap the raw result in the expected structure or log/raise an error
                        # For now, let's assume it might be a simpler direct content response sometimes
                        # or an error structure we haven't accounted for.
                        # This part might need refinement based on actual Firecrawl responses.
                        print(f"Warning: Firecrawl response for {url} did not have expected data.content structure. Response: {result}")
                        # Fallback: if 'content' is top-level, use it
                        if 'content' in result:
                            result = {'data': {'content': result['content']}, 'metadata':{}}
                        else: # If no content at all, consider it an issue
                            result = {'data': {'content': ''}, 'error': 'Unexpected response structure', 'metadata':{}}

                    if 'metadata' not in result:
                        result['metadata'] = {}
                    result["metadata"].update({
                        "scraped_at": datetime.utcnow().isoformat(),
                        "url": url
                    })
                    
                    # Cache the result
                    await self._save_to_cache(url, result)
                    
                    return result
            except aiohttp.ClientError as e:
                raise Exception(f"Failed to connect to Firecrawl: {str(e)}")

    async def scrape_multiple_urls(
        self,
        urls: List[str],
        force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """Scrape multiple URLs in parallel."""
        if not urls:
            return []

        # Validate all URLs first
        invalid_urls = [url for url in urls if not self.validate_url(url)]
        if invalid_urls:
            raise ValueError(f"Invalid URLs: {', '.join(invalid_urls)}")

        # Running sequentially to make debugging easier for now if issues persist
        processed_results = []
        for url in urls:
            try:
                result = await self.scrape_url(url, force_refresh=force_refresh)
                processed_results.append({
                    **result, # result should now have 'data' and 'metadata'
                    "success": True,
                    "url": url # Ensure URL is part of the top-level result for scrape_multiple_urls
                })
            except Exception as e:
                processed_results.append({
                    "url": url,
                    "error": str(e),
                    "success": False,
                    "data": {"content": ""}, # Ensure data.content exists for error cases too
                    "metadata": {"url": url}
                })
        return processed_results 