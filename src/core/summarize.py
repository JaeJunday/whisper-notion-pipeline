import time
import subprocess
import json
from typing import Dict, Optional

from ..utils.config import settings
from ..utils.logger import logger


class SummarizeError(Exception):
    """Raised when summarization fails"""
    pass


class Summarizer:
    def __init__(self, config=None):
        self.config = config or settings.summarization
        self.client = None
        
    def load_client(self):
        """Load the LLM client"""
        if not self.config.enabled:
            return
            
        if self.client is not None:
            return
            
        logger.info(f"Loading {self.config.engine} client")
        
        if self.config.engine == "openai":
            self._load_openai_client()
        elif self.config.engine == "litellm":
            self._load_litellm_client()
        elif self.config.engine == "gemini-cli":
            self._load_gemini_cli()
        elif self.config.engine == "claude-cli":
            self._load_claude_cli()
        else:
            raise ValueError(f"Unknown summarization engine: {self.config.engine}")
            
    def _load_openai_client(self):
        """Load OpenAI client"""
        if not settings.openai_api_key:
            raise SummarizeError("OpenAI API key not provided")
            
        try:
            import openai
            openai.api_key = settings.openai_api_key
            self.client = openai
        except ImportError as e:
            raise SummarizeError("OpenAI package not installed") from e
            
    def _load_litellm_client(self):
        """Load LiteLLM client"""
        try:
            import litellm
            self.client = litellm
            
            # Set API keys if available
            if settings.openai_api_key:
                litellm.openai_key = settings.openai_api_key
                
        except ImportError as e:
            raise SummarizeError("LiteLLM package not installed") from e
            
    def _load_gemini_cli(self):
        """Load Gemini CLI"""
        try:
            # Check if gemini CLI is available
            result = subprocess.run(["which", "gemini"], capture_output=True, text=True)
            if result.returncode != 0:
                raise SummarizeError("Gemini CLI not found. Please install gemini CLI.")
            self.client = "gemini-cli"
            logger.info("Gemini CLI loaded successfully")
        except Exception as e:
            raise SummarizeError(f"Failed to load Gemini CLI: {e}") from e
            
    def _load_claude_cli(self):
        """Load Claude CLI"""
        try:
            # Check if claude CLI is available
            result = subprocess.run(["which", "claude"], capture_output=True, text=True)
            if result.returncode != 0:
                raise SummarizeError("Claude CLI not found. Please install claude CLI.")
            self.client = "claude-cli"
            logger.info("Claude CLI loaded successfully")
        except Exception as e:
            raise SummarizeError(f"Failed to load Claude CLI: {e}") from e
            
    def summarize(self, text: str, context: Optional[Dict] = None) -> Dict:
        """Summarize the given text"""
        if not self.config.enabled:
            logger.info("Summarization is disabled")
            return {
                "summary": text[:500] + "..." if len(text) > 500 else text,
                "full_text": text,
                "summarized": False,
            }
            
        self.load_client()
        
        logger.info(f"Summarizing text ({len(text)} characters)")
        start_time = time.time()
        
        try:
            # Prepare prompt
            user_prompt = self.config.user_prompt_template.format(
                transcript=text,
                **(context or {})
            )
            
            # Call appropriate engine
            if self.config.engine == "openai":
                result = self._summarize_openai(user_prompt)
            elif self.config.engine == "litellm":
                result = self._summarize_litellm(user_prompt)
            elif self.config.engine == "gemini-cli":
                result = self._summarize_gemini_cli(user_prompt)
            elif self.config.engine == "claude-cli":
                result = self._summarize_claude_cli(user_prompt)
            else:
                raise ValueError(f"Unknown summarization engine: {self.config.engine}")
                
            duration = time.time() - start_time
            logger.info(f"Summarization completed in {duration:.2f}s")
            
            return {
                "summary": result,
                "full_text": text,
                "summarized": True,
                "engine": self.config.engine,
                "model": self.config.model,
                "duration": duration,
            }
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise SummarizeError(f"Summarization failed: {e}") from e
            
    def _summarize_openai(self, prompt: str) -> str:
        """Summarize using OpenAI"""
        response = self.client.ChatCompletion.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        
        return response.choices[0].message.content
        
    def _summarize_litellm(self, prompt: str) -> str:
        """Summarize using LiteLLM"""
        response = self.client.completion(
            model=self.config.model,
            messages=[
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        
        return response.choices[0].message.content
        
    def _summarize_gemini_cli(self, prompt: str) -> str:
        """Summarize using Gemini CLI"""
        # Combine system prompt and user prompt
        full_prompt = f"{self.config.system_prompt}\n\n{prompt}"
        
        # Call gemini CLI
        cmd = ["gemini"]
        
        try:
            logger.info(f"Running gemini CLI command...")
            result = subprocess.run(
                cmd,
                input=full_prompt,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise SummarizeError(f"Gemini CLI error: {e.stderr}") from e
            
    def _summarize_claude_cli(self, prompt: str) -> str:
        """Summarize using Claude CLI"""
        # Combine system prompt and user prompt
        full_prompt = f"{self.config.system_prompt}\n\n{prompt}"
        
        # Call claude CLI
        cmd = ["claude"]
        
        try:
            result = subprocess.run(
                cmd,
                input=full_prompt,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise SummarizeError(f"Claude CLI error: {e.stderr}") from e
        
    def chunk_and_summarize(self, text: str, chunk_size: int = 10000) -> Dict:
        """Summarize long text by chunking"""
        if len(text) <= chunk_size:
            return self.summarize(text)
            
        # Split text into chunks
        chunks = []
        words = text.split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        logger.info(f"Text split into {len(chunks)} chunks")
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Summarizing chunk {i}/{len(chunks)}")
            result = self.summarize(chunk)
            chunk_summaries.append(result["summary"])
            
        # Combine chunk summaries
        combined_summary = "\n\n".join(chunk_summaries)
        
        # Final summarization if needed
        if len(combined_summary) > chunk_size:
            logger.info("Performing final summarization of chunk summaries")
            final_result = self.summarize(combined_summary)
            return {
                "summary": final_result["summary"],
                "full_text": text,
                "summarized": True,
                "chunked": True,
                "num_chunks": len(chunks),
                "engine": self.config.engine,
                "model": self.config.model,
            }
        else:
            return {
                "summary": combined_summary,
                "full_text": text,
                "summarized": True,
                "chunked": True,
                "num_chunks": len(chunks),
                "engine": self.config.engine,
                "model": self.config.model,
            }