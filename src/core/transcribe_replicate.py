import os
import time
import requests
from pathlib import Path
from typing import Dict, Union, Optional
import replicate

from ..utils.config import settings
from ..utils.logger import logger


class ReplicateTranscriber:
    """Transcribe using Replicate API (GPU in cloud)"""
    
    def __init__(self):
        self.api_token = os.getenv("REPLICATE_API_TOKEN")
        if not self.api_token:
            raise ValueError("REPLICATE_API_TOKEN environment variable not set")
        
        # Initialize client
        self.client = replicate.Client(api_token=self.api_token)
        
    def transcribe(self, audio_path: Union[str, Path], model: str = "large-v3") -> Dict:
        """Transcribe audio using Replicate"""
        audio_path = Path(audio_path)
        
        logger.info(f"Uploading audio to Replicate...")
        
        # Upload file
        with open(audio_path, "rb") as f:
            # Run the model
            output = self.client.run(
                "openai/whisper:4d50797290df275329f202e48c76360b3f22b08d28c196cbc54600319435f8d2",
                input={
                    "audio": f,
                    "model": model,
                    "transcription": "plain text",
                    "temperature": 0,
                    "condition_on_previous_text": True,
                    "temperature_increment_on_fallback": 0.2,
                    "compression_ratio_threshold": 2.4,
                    "logprob_threshold": -1.0,
                    "no_speech_threshold": 0.6
                }
            )
        
        logger.info("Transcription completed via Replicate")
        
        # Parse output
        if isinstance(output, dict):
            return {
                "text": output.get("transcription", ""),
                "language": output.get("detected_language", "unknown"),
                "segments": output.get("segments", []),
                "task": "transcribe"
            }
        else:
            # Simple text output
            return {
                "text": str(output),
                "language": "unknown",
                "segments": [],
                "task": "transcribe"
            }