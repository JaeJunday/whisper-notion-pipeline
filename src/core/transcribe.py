import json
import time
import threading
from pathlib import Path
from typing import Dict, Union, List, Optional

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
        self.checkpoint_config = settings.checkpoint
        self._checkpoint_lock = threading.Lock()
        self._last_checkpoint_time = 0
        
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
        # Force CPU for now to avoid MPS issues
        logger.info(f"Loading model on CPU (MPS support disabled temporarily)")
        self.model = whisper.load_model(
            model_path,
            device="cpu"
        )
            
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
        
        logger.info(f"Starting OpenAI Whisper transcription...")
        logger.info(f"Audio file: {audio_path}")
        logger.info(f"File size: {audio_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Check for existing checkpoint
        checkpoint = self._load_checkpoint(audio_path)
        if checkpoint:
            logger.info(f"Resuming from checkpoint: {len(checkpoint['segments'])} segments already processed")
            # For now, we'll still process the whole file but can use this info for validation
        
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
        
        logger.info(f"Transcribing with options: {options}")
        file_size_mb = audio_path.stat().st_size / 1024 / 1024
        logger.info(f"Processing {file_size_mb:.1f}MB audio file...")
        
        # Estimate processing time (roughly 1-2 minutes per 10MB on base model)
        estimated_minutes = file_size_mb / 10 * 1.5
        if estimated_minutes > 1:
            logger.info(f"Estimated processing time: {estimated_minutes:.1f} minutes")
        
        # Add verbose=True to see progress
        # Disable FP16 for CPU to avoid warning
        if self.device == "cpu" or self.device == "mps":
            options['fp16'] = False
        
        # Create a custom progress callback
        segments_collected = []
        text_parts = []
        
        def progress_callback(seek, num_frames):
            """Called during transcription to track progress"""
            # This is called periodically but doesn't give us segments directly
            # We'll need to save checkpoints after transcription completes
            pass
        
        # Unfortunately, OpenAI Whisper doesn't support real-time segment collection
        # We'll save checkpoint after transcription completes
        result = self.model.transcribe(str(audio_path), verbose=True, **options)
        
        # Save final result as checkpoint (in case post-processing fails)
        if self.checkpoint_config.enabled and result.get("segments"):
            self._save_checkpoint(
                audio_path,
                result.get("segments", []),
                result["text"],
                result["language"]
            )
        
        logger.info(f"Transcription complete. Detected language: {result['language']}")
        logger.info(f"Text length: {len(result['text'])} characters")
        
        # Clear checkpoint on success
        self._clear_checkpoint(audio_path)
        
        return {
            "text": result["text"],
            "language": result["language"],
            "segments": result.get("segments", []),
            "task": self.config.task,
        }
        
    def _transcribe_faster(self, audio_path: Path) -> Dict:
        """Transcribe using Faster Whisper"""
        # Check for existing checkpoint
        checkpoint = self._load_checkpoint(audio_path)
        start_time_offset = 0
        if checkpoint:
            logger.info(f"Resuming from checkpoint: {len(checkpoint['segments'])} segments already processed")
            start_time_offset = checkpoint.get('last_segment_time', 0)
        
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
        
        # If we have a checkpoint, start with existing segments
        if checkpoint:
            segment_list = checkpoint['segments']
            full_text = [checkpoint['text']]
        
        last_checkpoint_save = time.time()
        
        for segment in segments:
            # Skip segments we've already processed
            if segment.end <= start_time_offset:
                continue
                
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
            
            # Save checkpoint periodically
            current_time = time.time()
            if (current_time - last_checkpoint_save) >= self.checkpoint_config.interval:
                self._save_checkpoint(
                    audio_path,
                    segment_list,
                    "".join(full_text),
                    info.language
                )
                last_checkpoint_save = current_time
        
        # Save final checkpoint
        if self.checkpoint_config.enabled:
            self._save_checkpoint(
                audio_path,
                segment_list,
                "".join(full_text),
                info.language
            )
        
        # Clear checkpoint on success
        self._clear_checkpoint(audio_path)
            
        return {
            "text": "".join(full_text),
            "language": info.language,
            "segments": segment_list,
            "task": self.config.task,
            "duration": info.duration,
            "duration_after_vad": info.duration_after_vad,
        }
        
    def save_result(self, result: Dict, output_path: Union[str, Path], is_partial: bool = False) -> Path:
        """Save transcription result to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add partial suffix if this is an intermediate save
        if is_partial:
            output_path = output_path.with_name(f"{output_path.stem}_partial")
        
        # Save main text
        text_path = output_path.with_suffix(".txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
            
        # Save segments as SRT if available
        if result.get("segments"):
            srt_path = output_path.with_suffix(".srt")
            self._save_srt(result["segments"], srt_path)
            
        # Save full result as JSON
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        if is_partial:
            logger.info(f"Partial results saved to: {output_path.parent}")
        else:
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
    
    def _save_checkpoint(self, audio_path: Path, segments: List[Dict], text: str, language: str):
        """Save checkpoint with current transcription progress"""
        if not self.checkpoint_config.enabled:
            return
            
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_file = checkpoint_dir / f"{audio_path.stem}_checkpoint.json"
        checkpoint_data = {
            "audio_path": str(audio_path),
            "language": language,
            "segments": segments,
            "text": text,
            "last_segment_time": segments[-1]["end"] if segments else 0,
            "timestamp": time.time()
        }
        
        with self._checkpoint_lock:
            # Save checkpoint
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            
            # Remove old checkpoints if keep_all is False
            if not self.checkpoint_config.keep_all:
                for old_checkpoint in checkpoint_dir.glob(f"{audio_path.stem}_checkpoint_*.json"):
                    if old_checkpoint != checkpoint_file:
                        old_checkpoint.unlink()
                        
            logger.info(f"Checkpoint saved: {len(segments)} segments, {len(text)} characters")
    
    def _load_checkpoint(self, audio_path: Path) -> Optional[Dict]:
        """Load checkpoint if exists"""
        checkpoint_file = Path("checkpoints") / f"{audio_path.stem}_checkpoint.json"
        
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    checkpoint_data = json.load(f)
                    
                logger.info(f"Found checkpoint with {len(checkpoint_data['segments'])} segments")
                return checkpoint_data
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                
        return None
    
    def _clear_checkpoint(self, audio_path: Path):
        """Clear checkpoint after successful completion"""
        checkpoint_file = Path("checkpoints") / f"{audio_path.stem}_checkpoint.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info("Checkpoint cleared after successful completion")