import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class WhisperConfig(BaseModel):
    model: str = "base"
    device: str = "auto"
    engine: str = "openai-whisper"
    language: Optional[str] = None
    task: str = "transcribe"
    temperature: float = 0
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6


class TranslationConfig(BaseModel):
    enabled: bool = False
    target_language: str = "en"
    engine: str = "whisper"


class SummarizationConfig(BaseModel):
    enabled: bool = True
    engine: str = "ollama"
    model: str = "gemma:2b"
    max_tokens: int = 4000
    temperature: float = 0.7
    system_prompt: str
    user_prompt_template: str


class NotionProperty(BaseModel):
    name: str
    type: str
    value: Optional[Any] = None


class NotionConfig(BaseModel):
    enabled: bool = True
    database_id: Optional[str] = None
    page_title_template: str = "[{date}] {filename} - Transcript"
    properties: List[NotionProperty] = []
    markdown_to_blocks: bool = True
    max_block_length: int = 2000


class QueueConfig(BaseModel):
    max_workers: int = 3
    worker_type: str = "thread"
    retry_attempts: int = 3
    retry_delay: int = 5


class MonitorConfig(BaseModel):
    enabled: bool = True
    input_dir: str = "input"
    output_dir: str = "output"
    watch_extensions: List[str] = Field(default_factory=lambda: [
        ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma", ".aac"
    ])
    poll_interval: int = 5


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str
    file: str
    rotation: str = "1 day"
    retention: str = "30 days"
    compression: str = "zip"


class HooksConfig(BaseModel):
    post_transcribe: Optional[str] = None
    post_translate: Optional[str] = None
    post_summarize: Optional[str] = None
    post_upload: Optional[str] = None


class PerformanceConfig(BaseModel):
    chunk_length: int = 30
    batch_size: int = 16
    num_workers: int = 4
    prefetch_factor: int = 2


class CheckpointConfig(BaseModel):
    enabled: bool = True
    interval: int = 300  # 5 minutes
    keep_all: bool = False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Environment variables
    notion_token: Optional[str] = None
    notion_parent_id: Optional[str] = None
    openai_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    whisper_model_path: Optional[str] = None
    max_workers: int = 3
    device: str = "auto"

    # Config from YAML
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    summarization: SummarizationConfig = Field(default_factory=lambda: SummarizationConfig(
        system_prompt="You are a professional meeting transcriptionist.",
        user_prompt_template="Summarize: {transcript}"
    ))
    notion: NotionConfig = Field(default_factory=NotionConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    logging: LoggingConfig = Field(default_factory=lambda: LoggingConfig(
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        file="logs/{time:YYYY-MM-DD}.log"
    ))
    hooks: HooksConfig = Field(default_factory=HooksConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "Settings":
        """Load settings from YAML file and environment variables"""
        settings = cls()
        
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            
            # Update settings with YAML data
            for key, value in config_data.items():
                if hasattr(settings, key) and value is not None:
                    if isinstance(getattr(settings, key), BaseModel):
                        # Update nested config
                        setattr(settings, key, type(getattr(settings, key))(**value))
                    else:
                        setattr(settings, key, value)
        
        # Override with environment variables
        if settings.notion_parent_id and settings.notion.database_id is None:
            settings.notion.database_id = settings.notion_parent_id
        
        return settings

    def get_device(self) -> str:
        """Determine the device to use for computation"""
        if self.device != "auto":
            return self.device
        
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        
        return "cpu"

    def validate_paths(self) -> None:
        """Create necessary directories if they don't exist"""
        dirs = [
            self.monitor.input_dir,
            self.monitor.output_dir,
            Path(self.logging.file).parent,
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings.from_yaml()