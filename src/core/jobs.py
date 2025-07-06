import concurrent.futures
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Union

from ..utils.config import settings
from ..utils.logger import logger
from .notion_uploader import NotionUploader
from .summarize import Summarizer
from .transcribe import Transcriber
from .translate import Translator


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class Job:
    id: str
    audio_path: Path
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "audio_path": str(self.audio_path),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count,
        }


class JobQueue:
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or settings.queue.max_workers
        self.worker_type = settings.queue.worker_type
        self.jobs: Dict[str, Job] = {}
        self.queue: Queue = Queue()
        self.lock = Lock()
        self.executor = None
        self.running = False
        
        # Components
        self.transcriber = Transcriber()
        self.translator = Translator()
        self.summarizer = Summarizer()
        self.uploader = NotionUploader()
        
    def start(self):
        """Start the job queue"""
        if self.running:
            return
            
        logger.info(f"Starting job queue with {self.max_workers} workers")
        
        if self.worker_type == "thread":
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            )
        else:
            self.executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers
            )
            
        self.running = True
        
        # Start worker threads
        for _ in range(self.max_workers):
            self.executor.submit(self._worker)
            
    def stop(self):
        """Stop the job queue"""
        if not self.running:
            return
            
        logger.info("Stopping job queue")
        self.running = False
        
        # Add stop signals to queue
        for _ in range(self.max_workers):
            self.queue.put(None)
            
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
            
    def add_job(self, audio_path: Union[str, Path]) -> str:
        """Add a new job to the queue"""
        audio_path = Path(audio_path)
        
        # Generate job ID
        job_id = f"{audio_path.stem}_{int(time.time() * 1000)}"
        
        # Create job
        job = Job(id=job_id, audio_path=audio_path)
        
        with self.lock:
            self.jobs[job_id] = job
            
        # Add to queue
        self.queue.put(job_id)
        
        logger.info(f"Added job {job_id} for {audio_path}")
        return job_id
        
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get the status of a job"""
        with self.lock:
            job = self.jobs.get(job_id)
            return job.to_dict() if job else None
            
    def get_all_jobs(self) -> List[Dict]:
        """Get all jobs"""
        with self.lock:
            return [job.to_dict() for job in self.jobs.values()]
            
    def _worker(self):
        """Worker thread/process"""
        while self.running:
            try:
                # Get job from queue
                job_id = self.queue.get(timeout=1)
                
                if job_id is None:
                    break
                    
                # Process job
                self._process_job(job_id)
                
            except Exception as e:
                logger.error(f"Worker error: {e}")
                
    def _process_job(self, job_id: str):
        """Process a single job"""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return
                
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            
        logger.info(f"Processing job {job_id}")
        
        try:
            # Create output directory
            output_dir = Path(settings.monitor.output_dir) / job.audio_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Transcribe
            result = self._step_transcribe(job, output_dir)
            
            # Step 2: Translate (optional)
            if settings.translation.enabled:
                result = self._step_translate(job, result, output_dir)
                
            # Step 3: Summarize (optional)
            if settings.summarization.enabled:
                result = self._step_summarize(job, result, output_dir)
                
            # Step 4: Upload to Notion (optional)
            if settings.notion.enabled:
                result = self._step_upload(job, result, output_dir)
                
            # Update job status
            with self.lock:
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now()
                job.result = result
                
            # Save job result
            self._save_job_result(job, output_dir)
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            
            with self.lock:
                job.error = str(e)
                job.retry_count += 1
                
                if job.retry_count < settings.queue.retry_attempts:
                    job.status = JobStatus.RETRYING
                    # Re-queue the job
                    self.queue.put(job_id)
                    logger.info(f"Retrying job {job_id} (attempt {job.retry_count})")
                else:
                    job.status = JobStatus.FAILED
                    job.completed_at = datetime.now()
                    
    def _step_transcribe(self, job: Job, output_dir: Path) -> Dict:
        """Transcription step"""
        logger.info(f"Step 1/4: Transcribing {job.audio_path}")
        
        result = self.transcriber.transcribe(job.audio_path)
        
        # Save transcription
        output_path = output_dir / "transcription"
        self.transcriber.save_result(result, output_path)
        
        # Run hook if configured
        self._run_hook("post_transcribe", result)
        
        return result
        
    def _step_translate(self, job: Job, result: Dict, output_dir: Path) -> Dict:
        """Translation step"""
        logger.info(f"Step 2/4: Translating from {result['language']}")
        
        translation = self.translator.translate(
            result["text"],
            source_language=result["language"]
        )
        
        # Update result
        result["translation"] = translation
        
        # Save translation
        if translation["translated"]:
            translation_path = output_dir / f"translation_{settings.translation.target_language}.txt"
            with open(translation_path, "w", encoding="utf-8") as f:
                f.write(translation["text"])
                
        # Run hook if configured
        self._run_hook("post_translate", result)
        
        return result
        
    def _step_summarize(self, job: Job, result: Dict, output_dir: Path) -> Dict:
        """Summarization step"""
        logger.info("Step 3/4: Summarizing")
        
        # Use translated text if available
        text_to_summarize = result.get("translation", {}).get("text", result["text"])
        
        # Context for summarization
        context = {
            "filename": job.audio_path.name,
            "duration": result.get("duration", 0),
            "language": result["language"],
        }
        
        summary = self.summarizer.summarize(text_to_summarize, context)
        
        # Update result
        result["summary"] = summary
        
        # Save summary
        if summary["summarized"]:
            summary_path = output_dir / "summary.md"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary["summary"])
                
        # Run hook if configured
        self._run_hook("post_summarize", result)
        
        return result
        
    def _step_upload(self, job: Job, result: Dict, output_dir: Path) -> Dict:
        """Notion upload step"""
        logger.info("Step 4/4: Uploading to Notion")
        
        # Prepare title
        title = settings.notion.page_title_template.format(
            date=datetime.now().strftime("%Y-%m-%d"),
            filename=job.audio_path.stem,
        )
        
        # Prepare content
        content_parts = []
        
        # Add metadata
        content_parts.append(f"# {title}\n")
        content_parts.append(f"**Audio File**: {job.audio_path.name}")
        content_parts.append(f"**Duration**: {result.get('duration', 0):.2f} seconds")
        content_parts.append(f"**Language**: {result['language']}")
        content_parts.append(f"**Processed**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Add summary if available
        if result.get("summary", {}).get("summarized"):
            content_parts.append("## Summary\n")
            content_parts.append(result["summary"]["summary"])
            content_parts.append("\n")
            
        # Add full transcript
        content_parts.append("## Full Transcript\n")
        text = result.get("translation", {}).get("text", result["text"])
        content_parts.append(text)
        
        content = "\n".join(content_parts)
        
        # Metadata for properties
        metadata = {
            "duration": result.get("duration", 0),
            "language": result["language"],
        }
        
        # Upload
        upload_result = self.uploader.upload(title, content, metadata)
        
        # Update result
        result["notion"] = upload_result
        
        # Run hook if configured
        self._run_hook("post_upload", result)
        
        return result
        
    def _save_job_result(self, job: Job, output_dir: Path):
        """Save job result to file"""
        result_path = output_dir / "job_result.json"
        
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(job.to_dict(), f, ensure_ascii=False, indent=2)
            
    def _run_hook(self, hook_name: str, data: Dict):
        """Run a configured hook"""
        hook_path = getattr(settings.hooks, hook_name)
        
        if not hook_path:
            return
            
        try:
            # Import and run hook
            import importlib.util
            
            spec = importlib.util.spec_from_file_location("hook", hook_path)
            hook_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hook_module)
            
            if hasattr(hook_module, "run"):
                hook_module.run(data)
                logger.debug(f"Ran hook: {hook_name}")
                
        except Exception as e:
            logger.warning(f"Failed to run hook {hook_name}: {e}")