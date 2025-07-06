import os
import time
from pathlib import Path
from typing import Dict, Optional, Union

import torch

from ..utils.config import settings
from ..utils.logger import logger


class TranscribeError(Exception):
    """Raised when transcription fails"""
    pass


class Transcriber:
    def __init__(self, config=None):
        self.config = config or settings.whisper
        self.model = None
        self.device = settings.get_device()
        
    def load_model(self):
        """Load the Whisper model"""
        if self.model is not None:
            return
            
        logger.info(f"Loading {self.config.engine} model: {self.config.model} on {self.device}")
        
        if self.config.engine == "openai-whisper":
            self._load_openai_whisper()
        elif self.config.engine == "faster-whisper":
            self._load_faster_whisper()
        else:
            raise ValueError(f"Unknown engine: {self.config.engine}")
            
    def _load_openai_whisper(self):
        """Load OpenAI Whisper model"""
        import whisper
        
        model_path = settings.whisper_model_path or self.config.model
        self.model = whisper.load_model(
            model_path,
            device=self.device if self.device != "mps" else "cpu"
        )
        
        if self.device == "mps":
            # Move to MPS after loading
            self.model = self.model.to("mps")
            
    def _load_faster_whisper(self):
        """Load Faster Whisper model"""
        from faster_whisper import WhisperModel
        
        compute_type = "float16" if self.device == "cuda" else "int8"
        self.model = WhisperModel(
            self.config.model,
            device=self.device if self.device != "mps" else "cpu",
            compute_type=compute_type,
            num_workers=settings.performance.num_workers,
        )
        
    def transcribe(self, audio_path: Union[str, Path]) -> Dict:
        """Transcribe audio file"""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        self.load_model()
        
        logger.info(f"Transcribing: {audio_path}")
        start_time = time.time()
        
        try:
            if self.config.engine == "openai-whisper":
                result = self._transcribe_openai(audio_path)
            else:
                result = self._transcribe_faster(audio_path)
                
            duration = time.time() - start_time
            logger.info(f"Transcription completed in {duration:.2f}s")
            
            # Add metadata
            result["duration"] = duration
            result["audio_path"] = str(audio_path)
            result["engine"] = self.config.engine
            result["model"] = self.config.model
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise TranscribeError(f"Transcription failed: {e}") from e
            
    def _transcribe_openai(self, audio_path: Path) -> Dict:
        """Transcribe using OpenAI Whisper"""
        import whisper
        
        options = {
            "language": self.config.language,
            "task": self.config.task,
            "temperature": self.config.temperature,
            "beam_size": self.config.beam_size,
            "best_of": self.config.best_of,
            "patience": self.config.patience,
            "compression_ratio_threshold": self.config.compression_ratio_threshold,
            "logprob_threshold": self.config.logprob_threshold,
            "no_speech_threshold": self.config.no_speech_threshold,
        }
        
        # Remove None values
        options = {k: v for k, v in options.items() if v is not None}
        
        result = self.model.transcribe(str(audio_path), **options)
        
        return {
            "text": result["text"],
            "language": result["language"],
            "segments": result.get("segments", []),
            "task": self.config.task,
        }
        
    def _transcribe_faster(self, audio_path: Path) -> Dict:
        """Transcribe using Faster Whisper"""
        segments, info = self.model.transcribe(
            str(audio_path),
            language=self.config.language,
            task=self.config.task,
            temperature=self.config.temperature,
            beam_size=self.config.beam_size,
            best_of=self.config.best_of,
            patience=self.config.patience,
            compression_ratio_threshold=self.config.compression_ratio_threshold,
            log_prob_threshold=self.config.logprob_threshold,
            no_speech_threshold=self.config.no_speech_threshold,
        )
        
        # Convert generator to list and build full text
        segment_list = []
        full_text = []
        
        for segment in segments:
            segment_dict = {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "tokens": segment.tokens,
                "temperature": segment.temperature,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "no_speech_prob": segment.no_speech_prob,
            }
            segment_list.append(segment_dict)
            full_text.append(segment.text)
            
        return {
            "text": "".join(full_text),
            "language": info.language,
            "segments": segment_list,
            "task": self.config.task,
            "duration": info.duration,
            "duration_after_vad": info.duration_after_vad,
        }
        
    def save_result(self, result: Dict, output_path: Union[str, Path]) -> Path:
        """Save transcription result to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save main text
        text_path = output_path.with_suffix(".txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
            
        # Save segments as SRT if available
        if result.get("segments"):
            srt_path = output_path.with_suffix(".srt")
            self._save_srt(result["segments"], srt_path)
            
        # Save full result as JSON
        import json
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Results saved to: {output_path.parent}")
        return text_path
        
    def _save_srt(self, segments: list, srt_path: Path):
        """Save segments as SRT subtitle file"""
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, 1):
                start = self._seconds_to_srt_time(segment["start"])
                end = self._seconds_to_srt_time(segment["end"])
                text = segment["text"].strip()
                
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
                
    @staticmethod
    def _seconds_to_srt_time(seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"