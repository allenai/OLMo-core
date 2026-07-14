import asyncio
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from openai.types.shared.reasoning import Reasoning
from tqdm import tqdm
import sqlite3
import threading
from google import genai
from google.genai import types
from google.genai.types import HttpOptions

# Pricing per 1M tokens
PRICING = {
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gemini-2.5-flash": {"input": 0.3, "output": 2.5},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
    "gemini-2.5-flash-lite": {"input": 0.1, "output": 0.4},
    "gemini-3-pro-preview": {"input": 2, "output": 12},
}

class ParallelResponsesClient:
    """
    Fast, parallel client for OpenAI and Gemini APIs with SQLite caching and logging.
    Uses a persistent event loop to avoid loop closure issues.
    Includes retry logic for 429 rate limit errors.
    """

    def load_oai_key(self, keypath="/accounts/projects/sewonm/prasann/oaikey.sh"):
        with open(keypath, "r") as f:
            key = f.read().strip()
        return key

    def _is_gemini_model(self, model: str) -> bool:
        return model.startswith("gemini")
    
    def _is_openai_model(self, model: str) -> bool:
        return model.startswith("gpt")

    def _is_local_model(self, model: str) -> bool:
        """A model served by a local vLLM OpenAI-compatible endpoint. Any model
        that is not Gemini/OpenAI is treated as local IFF a local base URL was
        configured (constructor arg or LOCAL_LLM_BASE_URL env var)."""
        return bool(self.local_base_url) and not (
            self._is_gemini_model(model) or self._is_openai_model(model))
    
    def __init__(
        self,
        max_concurrent: int = 25,
        cache_db: str = "cache/response_cache.db",
        log_file: str = "cache/requests_log.jsonl",
        use_cache: bool = True,
        openai_key_path: Optional[str] = None,
        use_vertexai: bool = True,
        max_retries: int = 5,
        initial_retry_delay: float = 0.1,
        local_base_url: Optional[str] = None,
    ):
        # Local vLLM OpenAI-compatible endpoint (e.g. http://horton:8000/v1).
        # Enables serving an instruct model locally instead of a paid API; any
        # non-Gemini/OpenAI model name then routes here. Cost is treated as 0.
        self.local_base_url = local_base_url or os.environ.get("LOCAL_LLM_BASE_URL")
        self.max_concurrent = max_concurrent
        self.cache_db = Path(cache_db)
        self.log_file = Path(log_file)
        self.use_cache = use_cache
        self.total_cost = 0.0
        self.cache_hits = 0
        self.api_calls = 0
        self.openai_key_path = openai_key_path
        self.use_vertexai = use_vertexai
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        
        # Store API key for OpenAI
        if openai_key_path:
            self.openai_api_key = self.load_oai_key(openai_key_path)
        else:
            self.openai_api_key = None
        
        # Set Gemini environment
        os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = str(use_vertexai).lower()
        
        # Persistent event loop and clients
        self._loop = None
        self._loop_thread = None
        self.openai_client = None
        self.gemini_client = None
        self.local_client = None
        self.semaphore = None
        
        # Thread-local storage for database connections
        self._local = threading.local()
        
        # Initialize database
        if self.use_cache:
            self._init_db()
        
        # Start the event loop in a background thread
        self._start_event_loop()
    
    def _start_event_loop(self):
        """Start a persistent event loop in a background thread."""
        def run_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()
        
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=run_loop, args=(self._loop,), daemon=True)
        self._loop_thread.start()
        
        # Initialize clients in the event loop
        future = asyncio.run_coroutine_threadsafe(self._initialize_clients(), self._loop)
        future.result()  # Wait for initialization
    
    async def _initialize_clients(self):
        """Initialize API clients in the event loop."""
        if self.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = AsyncOpenAI()
        
        self.gemini_client = genai.Client(
            http_options=HttpOptions(api_version="v1"),
            vertexai=self.use_vertexai
        )

        # Local vLLM endpoint speaks the OpenAI API; api_key is ignored by vLLM.
        if self.local_base_url:
            self.local_client = AsyncOpenAI(base_url=self.local_base_url,
                                            api_key="EMPTY")

        self.semaphore = asyncio.Semaphore(self.max_concurrent)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(str(self.cache_db), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _init_db(self):
        """Initialize SQLite database with required tables."""
        self.cache_db.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(self.cache_db))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS response_cache (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                response TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                cost_usd REAL,
                success INTEGER,
                error TEXT,
                created_at TEXT,
                temperature REAL,
                max_output_tokens INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_cache_key ON response_cache(cache_key)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_created_at ON response_cache(created_at)
        ''')
        
        conn.commit()
        conn.close()
    
    def _get_cache_key(self, model: str, prompt: str, temperature: float, max_output_tokens: Optional[int], **kwargs) -> str:
        """Generate a cache key from request parameters."""
        cache_data = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            **kwargs
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached response from SQLite."""
        if not self.use_cache:
            return None
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT model, prompt_hash, response, input_tokens, output_tokens,
                       total_tokens, cost_usd, success, error
                FROM response_cache
                WHERE cache_key = ?
            ''', (cache_key,))

            row = cursor.fetchone()

            if row:
                return {
                    "model": row[0],
                    "prompt_hash": row[1],
                    "response": row[2],
                    "usage": {
                        "input_tokens": row[3],
                        "output_tokens": row[4],
                        "total_tokens": row[5]
                    } if row[3] is not None else None,
                    "cost_usd": row[6],
                    "success": bool(row[7]),
                    "error": row[8],
                    "cached": True
                }
            
            return None
            
        except Exception as e:
            print(f"Warning: Could not read from cache: {e}")
            return None
    
    def _hash_prompt(self, prompt: str) -> str:
        """Generate a SHA256 hash of the prompt for storage."""
        return hashlib.sha256(prompt.encode()).hexdigest()

    def _save_to_cache(self, cache_key: str, result: Dict[str, Any], temperature: float, max_output_tokens: Optional[int]):
        """Save a response to SQLite cache."""
        if not self.use_cache:
            return

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            usage = result.get("usage")
            prompt_hash = self._hash_prompt(result["prompt"])

            cursor.execute('''
                INSERT OR REPLACE INTO response_cache
                (cache_key, model, prompt_hash, response, input_tokens, output_tokens,
                 total_tokens, cost_usd, success, error, created_at, temperature, max_output_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                cache_key,
                result["model"],
                prompt_hash,
                result["response"],
                usage["input_tokens"] if usage else None,
                usage["output_tokens"] if usage else None,
                usage["total_tokens"] if usage else None,
                result["cost_usd"],
                int(result["success"]),
                result["error"],
                datetime.now().isoformat(),
                temperature,
                max_output_tokens
            ))
            
            conn.commit()
            
        except Exception as e:
            print(f"Warning: Could not save to cache: {e}")
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate the cost of a request in USD."""
        if model not in PRICING:  # local / self-hosted models are free
            return 0.0
        pricing = PRICING[model]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
    
    def _log_request(self, result: Dict[str, Any], cached: bool, model: str):
        """Log request details to JSONL file."""
        prompt = result.get("prompt", "")
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "prompt_hash": self._hash_prompt(prompt) if prompt else result.get("prompt_hash"),
            "success": result["success"],
            "cached": cached,
            "cost_usd": result.get("cost_usd", 0.0),
            "usage": result.get("usage"),
            "error": result.get("error")
        }
        
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if an error is a rate limit (429) error."""
        error_str = str(error).lower()
        return "429" in error_str or "rate limit" in error_str or "quota" in error_str
    
    async def get_completion(
        self,
        model: str,
        prompt: str,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        reasoning: Optional[str] = "minimal",
        thinking_budget: Optional[int] = 0,
        **kwargs
    ) -> Dict[str, Any]:
        assert model in PRICING or self._is_local_model(model), \
            f"Model {model} not found in pricing (and no local endpoint configured)"

        cache_key = self._get_cache_key(model, prompt, temperature, max_output_tokens, reasoning=reasoning, thinking_budget=thinking_budget, **kwargs)
        
        cached_result = self._get_cached_response(cache_key)
        if cached_result:
            self.cache_hits += 1
            self._log_request(cached_result, cached=True, model=model)
            return cached_result
        
        async with self.semaphore:
            retry_count = 0
            last_error = None
            
            while retry_count <= self.max_retries:
                try:
                    self.api_calls += 1
                    
                    if self._is_local_model(model):
                        result = await self._get_local_completion(
                            model, prompt, temperature, max_output_tokens, **kwargs
                        )
                    elif self._is_openai_model(model):
                        result = await self._get_openai_completion(
                            model, prompt, temperature, max_output_tokens, reasoning, **kwargs
                        )
                    elif self._is_gemini_model(model):
                        result = await self._get_gemini_completion(
                            model, prompt, temperature, max_output_tokens, thinking_budget, **kwargs
                        )
                    else:
                        raise ValueError(f"Unknown model type: {model}")
                    
                    self._save_to_cache(cache_key, result, temperature, max_output_tokens)
                    self._log_request(result, cached=False, model=model)
                    return result
                    
                except Exception as e:
                    last_error = e
                    
                    # Check if it's a rate limit error
                    if self._is_rate_limit_error(e):
                        retry_count += 1
                        if retry_count <= self.max_retries:
                            # Exponential backoff: 0.1, 0.2, 0.4, 0.8, 1.6 seconds
                            delay = self.initial_retry_delay * (2 ** (retry_count - 1))
                            print(f"Rate limit hit for {model}. Retrying in {delay:.2f}s (attempt {retry_count}/{self.max_retries})...")
                            await asyncio.sleep(delay)
                            continue
                    
                    # If not a rate limit error, or max retries exceeded, break
                    break
            
            # If we get here, all retries failed
            result = {
                "model": model,
                "prompt": prompt,
                "response": None,
                "usage": None,
                "cost_usd": 0.0,
                "success": False,
                "error": str(last_error),
                "cached": False
            }
            print(f"Error after {retry_count} retries: {last_error}")
            self._log_request(result, cached=False, model=model)
            return result
    
    async def _get_openai_completion(
        self,
        model: str,
        prompt: str,
        temperature: Optional[float],
        max_output_tokens: Optional[int],
        reasoning: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Get completion from OpenAI API."""
        response = await self.openai_client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning=Reasoning(effort=reasoning),
            **kwargs
        )
        
        cost = self._calculate_cost(
            response.usage.input_tokens,
            response.usage.output_tokens,
            model
        )
        self.total_cost += cost
        
        return {
            "model": model,
            "prompt": prompt,
            "response": response.output_text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "cost_usd": cost,
            "success": True,
            "error": None,
            "cached": False
        }
    
    async def _get_local_completion(
        self,
        model: str,
        prompt: str,
        temperature: Optional[float],
        max_output_tokens: Optional[int],
        **kwargs
    ) -> Dict[str, Any]:
        """Get completion from a local vLLM OpenAI-compatible endpoint.

        Uses the chat-completions API (instruct models). The single user prompt
        mirrors how the Gemini/OpenAI paths send `prompt` as the whole input.
        Cost is 0 (self-hosted). `reasoning`/`thinking_budget` don't apply.
        """
        if not self.local_client:
            raise ValueError("Local client not initialized (no base_url)")
        kwargs.pop("reasoning", None)
        kwargs.pop("thinking_budget", None)
        params = {"model": model,
                  "messages": [{"role": "user", "content": prompt}]}
        if temperature is not None:
            params["temperature"] = temperature
        if max_output_tokens is not None:
            params["max_tokens"] = max_output_tokens
        response = await self.local_client.chat.completions.create(**params, **kwargs)

        usage = getattr(response, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0
        return {
            "model": model,
            "prompt": prompt,
            "response": response.choices[0].message.content,
            "usage": {"input_tokens": in_tok, "output_tokens": out_tok,
                      "total_tokens": in_tok + out_tok},
            "cost_usd": 0.0,
            "success": True,
            "error": None,
            "cached": False,
        }

    async def _get_gemini_completion(
        self,
        model: str,
        prompt: str,
        temperature: Optional[float],
        max_output_tokens: Optional[int],
        thinking_budget: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Get completion from Gemini API."""
        if not self.gemini_client:
            raise ValueError("Gemini client not initialized")
        
        config_params = {}
        if temperature is not None:
            config_params["temperature"] = temperature
        if max_output_tokens is not None:
            config_params["max_output_tokens"] = max_output_tokens
        
        config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
        config_params["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
        config = types.GenerateContentConfig(**config_params)
        
        response = await self.gemini_client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )
        
        usage_metadata = response.usage_metadata
        input_tokens = usage_metadata.prompt_token_count
        output_tokens = usage_metadata.total_token_count - usage_metadata.prompt_token_count
        total_tokens = usage_metadata.total_token_count
        
        cost = self._calculate_cost(input_tokens, output_tokens, model)
        self.total_cost += cost
        
        return {
            "model": model,
            "prompt": prompt,
            "response": response.text,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            },
            "cost_usd": cost,
            "success": True,
            "error": None,
            "cached": False
        }
    
    async def get_completions(
        self,
        model: str,
        prompts: List[str],
        temperature: float = 1.0,
        max_output_tokens: Optional[int] = None,
        reasoning: Optional[str] = "minimal",
        thinking_budget: Optional[int] = 0,
        show_progress: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        
        tasks = [
            self.get_completion(model, prompt, temperature, max_output_tokens, reasoning, thinking_budget, **kwargs)
            for prompt in prompts
        ]

        if show_progress:
            pbar = tqdm(total=len(prompts), desc="Processing prompts", unit="prompt")

            async def _track(coro):
                result = await coro
                pbar.update(1)
                return result

            results = await asyncio.gather(*[_track(t) for t in tasks])
            pbar.close()
        else:
            results = await asyncio.gather(*tasks)

        return list(results)
    
    def run(
        self,
        model: str,
        prompts: List[str],
        temperature: float = 1.0,
        max_output_tokens: Optional[int] = None,
        reasoning: Optional[str] = "minimal",
        thinking_budget: Optional[int] = 0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for getting completions using persistent event loop."""
        if model == "gemini-2.5-pro" or model == "gemini-3-pro-preview":
            thinking_budget = max(thinking_budget, 128)
        
        # Submit to the persistent event loop
        future = asyncio.run_coroutine_threadsafe(
            self.get_completions(model, prompts, temperature, max_output_tokens, reasoning, thinking_budget, **kwargs),
            self._loop
        )
        
        return future.result()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about cache usage and costs."""
        cache_size = 0
        if self.use_cache:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM response_cache')
                cache_size = cursor.fetchone()[0]
            except Exception as e:
                print(f"Warning: Could not get cache size: {e}")
        
        return {
            "total_cost_usd": round(self.total_cost, 6),
            "api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(self.cache_hits / (self.api_calls + self.cache_hits), 2) if (self.api_calls + self.cache_hits) > 0 else 0,
            "cache_size": cache_size
        }
    
    def clear_cache(self):
        """Clear all entries from the cache."""
        if not self.use_cache:
            return
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('DELETE FROM response_cache')
            conn.commit()
            print("Cache cleared successfully")
        except Exception as e:
            print(f"Warning: Could not clear cache: {e}")
    
    def close(self):
        """Close database connections, API clients, and stop the event loop."""
        # Close clients in the event loop
        if self._loop and self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._close_clients_async(), self._loop)
            try:
                future.result(timeout=5)
            except Exception as e:
                print(f"Warning: Could not close clients: {e}")
        
        # Stop the event loop
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        
        # Close database connection
        if hasattr(self._local, 'conn'):
            try:
                self._local.conn.close()
            except Exception as e:
                print(f"Warning: Could not close database connection: {e}")
    
    async def _close_clients_async(self):
        """Close API clients asynchronously."""
        if self.openai_client:
            await self.openai_client.close()
        
        if self.gemini_client:
            self.gemini_client.close()

# Example usage
if __name__ == "__main__":
    # Initialize the client with retry settings
    client = ParallelResponsesClient(
        max_concurrent=50,
        use_cache=True,
        use_vertexai=True,
        max_retries=5,  # Maximum number of retries for 429 errors
        initial_retry_delay=0.1  # Initial delay in seconds
    )
    
    # Example with OpenAI models
    print("=" * 60)
    print("Testing OpenAI GPT models...")
    print("=" * 60)
    
    gpt_prompts = [
        "What is the capital of India?",
        "Explain quantum computing in one sentence.",
        "Write a haiku about programming.",
    ]
    
    gpt_results = client.run(
        model="gpt-5-nano", 
        prompts=gpt_prompts, 
        max_output_tokens=100,
        reasoning="minimal"
    )
    
    for i, result in enumerate(gpt_results, 1):
        cached_label = " [CACHED]" if result.get("cached") else ""
        if result["success"]:
            print(f"\nGPT Prompt {i}{cached_label}: {result['prompt']}")
            print(f"Response: {result['response']}")
            print(f"Cost: ${result['cost_usd']:.6f}")
        else:
            print(f"\nGPT Prompt {i} failed: {result['error']}")
    
    # Example with Gemini models
    print("\n" + "=" * 60)
    print("Testing Google Gemini models...")
    print("=" * 60)
    
    gemini_prompts = [
        "Answer yes or no, no  other text: Is india's capital delhi? ",
        "Answer yes or no, no  other text: Is france's capital paris? ",
        "Answer yes or no, no  other text: Is china's capital beijing? ",
    ]
    
    gemini_results = client.run(
        model="gemini-2.5-pro",
        prompts=gemini_prompts,
        max_output_tokens=100,
        thinking_budget=0
    )
    
    for i, result in enumerate(gemini_results, 1):
        cached_label = " [CACHED]" if result.get("cached") else ""
        if result["success"]:
            print(f"\nGemini Prompt {i}{cached_label}: {result['prompt']}")
            print(f"Response: {result['response']}")
            print(f"Usage: {result['usage']}")
            print(f"Cost: ${result['cost_usd']:.6f}")
        else:
            print(f"\nGemini Prompt {i} failed: {result['error']}")
    
    # Summary
    stats = client.get_stats()
    print(f"\n{'='*60}")
    print(f"Statistics:")
    print(f"  Total cost: ${stats['total_cost_usd']:.6f}")
    print(f"  API calls: {stats['api_calls']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.0%}")
    print(f"  Cache size: {stats['cache_size']} entries")
    print(f"\nLogs saved to: {client.log_file}")
    print(f"Cache saved to: {client.cache_db}")
    
    # Close connections
    client.close()