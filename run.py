#!/usr/bin/env python3

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.core.jobs import JobQueue, JobStatus
from src.core.notion_uploader import NotionUploader
from src.core.summarize import Summarizer
from src.core.transcribe import Transcriber
from src.core.translate import Translator
from src.utils.config import settings
from src.utils.logger import logger

app = typer.Typer(
    name="whisper-notion-pipeline",
    help="Automatic speech transcription and Notion upload pipeline",
)
console = Console()


class AudioFileHandler(FileSystemEventHandler):
    """Handler for monitoring audio files"""
    
    def __init__(self, job_queue: JobQueue):
        self.job_queue = job_queue
        self.processed_files = set()
        
    def on_created(self, event):
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        # Check if it's an audio file
        if file_path.suffix.lower() in settings.monitor.watch_extensions:
            # Wait a bit to ensure file is fully written
            time.sleep(1)
            
            # Check if already processed
            if file_path in self.processed_files:
                return
                
            self.processed_files.add(file_path)
            console.print(f"[green]New audio file detected:[/green] {file_path}")
            
            # Add to job queue
            job_id = self.job_queue.add_job(file_path)
            console.print(f"[blue]Added job:[/blue] {job_id}")


@app.command()
def process(
    audio_files: List[Path] = typer.Argument(
        ...,
        help="Audio files to process",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Whisper model to use (tiny, base, small, medium, large)",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        "-d",
        help="Device to use (auto, cuda, cpu)",
    ),
    task: Optional[str] = typer.Option(
        None,
        "--task",
        "-t",
        help="Task to perform (transcribe, translate)",
    ),
    translate: Optional[str] = typer.Option(
        None,
        "--translate",
        help="Target language for translation",
    ),
    summary: bool = typer.Option(
        True,
        "--summary/--no-summary",
        help="Enable/disable summarization",
    ),
    upload: bool = typer.Option(
        True,
        "--upload/--no-upload",
        help="Enable/disable Notion upload",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run without actually uploading to Notion",
    ),
    workers: Optional[int] = typer.Option(
        None,
        "--workers",
        "-w",
        help="Number of parallel workers",
    ),
):
    """Process audio files through the pipeline"""
    
    # Override settings if provided
    if model:
        settings.whisper.model = model
    if device:
        settings.device = device
        settings.whisper.device = device
    if task:
        settings.whisper.task = task
    if translate:
        settings.translation.enabled = True
        settings.translation.target_language = translate
    if not summary:
        settings.summarization.enabled = False
    if not upload:
        settings.notion.enabled = False
        
    # Validate paths
    settings.validate_paths()
    
    # Create job queue
    job_queue = JobQueue(max_workers=workers)
    
    # Add jobs
    console.print(f"[bold]Processing {len(audio_files)} file(s)[/bold]")
    job_ids = []
    
    for audio_file in audio_files:
        job_id = job_queue.add_job(audio_file)
        job_ids.append(job_id)
        console.print(f"  " {audio_file.name} ’ Job {job_id}")
        
    # Start processing
    job_queue.start()
    
    # Monitor progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=len(job_ids))
        
        completed = set()
        while len(completed) < len(job_ids):
            time.sleep(1)
            
            for job_id in job_ids:
                if job_id in completed:
                    continue
                    
                status = job_queue.get_job_status(job_id)
                if status and status["status"] in ["completed", "failed"]:
                    completed.add(job_id)
                    progress.advance(task)
                    
                    if status["status"] == "completed":
                        console.print(f"[green][/green] Job {job_id} completed")
                        
                        # Show results
                        result = status["result"]
                        if result.get("notion", {}).get("uploaded"):
                            console.print(f"  [blue]Notion URL:[/blue] {result['notion']['url']}")
                    else:
                        console.print(f"[red][/red] Job {job_id} failed: {status['error']}")
                        
    # Stop job queue
    job_queue.stop()
    
    # Summary
    console.print("\n[bold]Summary:[/bold]")
    all_jobs = job_queue.get_all_jobs()
    
    table = Table(title="Processing Results")
    table.add_column("File", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Duration", style="yellow")
    table.add_column("Language", style="blue")
    table.add_column("Words", style="magenta")
    
    for job in all_jobs:
        file_name = Path(job["audio_path"]).name
        status = "" if job["status"] == "completed" else ""
        
        duration = "-"
        language = "-"
        words = "-"
        
        if job["result"]:
            duration = f"{job['result'].get('duration', 0):.1f}s"
            language = job["result"].get("language", "-")
            words = str(len(job["result"].get("text", "").split()))
            
        table.add_row(file_name, status, duration, language, words)
        
    console.print(table)


@app.command()
def monitor(
    input_dir: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        help="Directory to monitor for audio files",
    ),
    workers: Optional[int] = typer.Option(
        None,
        "--workers",
        "-w",
        help="Number of parallel workers",
    ),
):
    """Monitor a directory for new audio files"""
    
    # Use provided directory or default
    monitor_dir = input_dir or Path(settings.monitor.input_dir)
    monitor_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate paths
    settings.validate_paths()
    
    console.print(f"[bold]Monitoring directory:[/bold] {monitor_dir}")
    console.print(f"[bold]Watching for:[/bold] {', '.join(settings.monitor.watch_extensions)}")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")
    
    # Create job queue
    job_queue = JobQueue(max_workers=workers)
    job_queue.start()
    
    # Set up file monitoring
    event_handler = AudioFileHandler(job_queue)
    observer = Observer()
    observer.schedule(event_handler, str(monitor_dir), recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(settings.monitor.poll_interval)
            
            # Show active jobs
            jobs = job_queue.get_all_jobs()
            active_jobs = [
                j for j in jobs
                if j["status"] in ["pending", "running", "retrying"]
            ]
            
            if active_jobs:
                console.print(f"[dim]Active jobs: {len(active_jobs)}[/dim]", end="\r")
                
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping monitor...[/yellow]")
        observer.stop()
        job_queue.stop()
        
    observer.join()
    console.print("[green]Monitor stopped[/green]")


@app.command()
def status(
    job_id: Optional[str] = typer.Argument(
        None,
        help="Job ID to check status for",
    ),
):
    """Check the status of jobs"""
    
    # Create job queue to access job data
    job_queue = JobQueue()
    
    if job_id:
        # Show specific job
        status = job_queue.get_job_status(job_id)
        
        if not status:
            console.print(f"[red]Job not found:[/red] {job_id}")
            return
            
        console.print(f"[bold]Job {job_id}[/bold]")
        console.print(f"Status: {status['status']}")
        console.print(f"Audio: {status['audio_path']}")
        console.print(f"Created: {status['created_at']}")
        
        if status['started_at']:
            console.print(f"Started: {status['started_at']}")
        if status['completed_at']:
            console.print(f"Completed: {status['completed_at']}")
            
        if status['error']:
            console.print(f"[red]Error:[/red] {status['error']}")
            
        if status['result']:
            result = status['result']
            console.print(f"\n[bold]Results:[/bold]")
            console.print(f"Language: {result.get('language', '-')}")
            console.print(f"Duration: {result.get('duration', 0):.1f} seconds")
            console.print(f"Words: {len(result.get('text', '').split())}")
            
            if result.get('notion', {}).get('uploaded'):
                console.print(f"Notion URL: {result['notion']['url']}")
                
    else:
        # Show all jobs
        jobs = job_queue.get_all_jobs()
        
        if not jobs:
            console.print("[yellow]No jobs found[/yellow]")
            return
            
        table = Table(title="All Jobs")
        table.add_column("Job ID", style="cyan")
        table.add_column("File", style="blue")
        table.add_column("Status", style="green")
        table.add_column("Created", style="yellow")
        
        for job in jobs:
            file_name = Path(job["audio_path"]).name
            table.add_row(
                job["id"],
                file_name,
                job["status"],
                job["created_at"],
            )
            
        console.print(table)


@app.command()
def test(
    component: str = typer.Argument(
        help="Component to test (whisper, translate, summarize, notion)",
    ),
    text: Optional[str] = typer.Option(
        None,
        "--text",
        help="Text to use for testing (for translate/summarize/notion)",
    ),
    audio: Optional[Path] = typer.Option(
        None,
        "--audio",
        help="Audio file to use for testing (for whisper)",
    ),
):
    """Test individual components"""
    
    console.print(f"[bold]Testing {component}...[/bold]")
    
    if component == "whisper":
        if not audio:
            console.print("[red]Please provide an audio file with --audio[/red]")
            return
            
        transcriber = Transcriber()
        
        with console.status("Transcribing..."):
            result = transcriber.transcribe(audio)
            
        console.print(f"[green]Success![/green]")
        console.print(f"Language: {result['language']}")
        console.print(f"Text preview: {result['text'][:200]}...")
        
    elif component == "translate":
        if not text:
            text = "Hello, this is a test message for translation."
            
        translator = Translator()
        settings.translation.enabled = True
        
        with console.status("Translating..."):
            result = translator.translate(text, source_language="en")
            
        console.print(f"[green]Success![/green]")
        console.print(f"Original: {text}")
        console.print(f"Translated: {result['text']}")
        
    elif component == "summarize":
        if not text:
            text = "This is a test message. " * 50
            
        summarizer = Summarizer()
        settings.summarization.enabled = True
        
        with console.status("Summarizing..."):
            result = summarizer.summarize(text)
            
        console.print(f"[green]Success![/green]")
        console.print(f"Summary: {result['summary']}")
        
    elif component == "notion":
        if not text:
            text = "# Test Page\n\nThis is a test upload to Notion."
            
        uploader = NotionUploader()
        settings.notion.enabled = True
        
        with console.status("Uploading..."):
            result = uploader.upload(
                "Test Page",
                text,
                dry_run=True,  # Always dry run for tests
            )
            
        console.print(f"[green]Success![/green]")
        console.print(f"Would create page with {result.get('total_blocks', 0)} blocks")
        
    else:
        console.print(f"[red]Unknown component:[/red] {component}")


@app.command()
def config():
    """Show current configuration"""
    
    console.print("[bold]Current Configuration:[/bold]\n")
    
    # Whisper settings
    console.print("[cyan]Whisper:[/cyan]")
    console.print(f"  Model: {settings.whisper.model}")
    console.print(f"  Device: {settings.get_device()}")
    console.print(f"  Engine: {settings.whisper.engine}")
    console.print(f"  Task: {settings.whisper.task}")
    
    # Translation settings
    console.print("\n[cyan]Translation:[/cyan]")
    console.print(f"  Enabled: {settings.translation.enabled}")
    console.print(f"  Target Language: {settings.translation.target_language}")
    console.print(f"  Engine: {settings.translation.engine}")
    
    # Summarization settings
    console.print("\n[cyan]Summarization:[/cyan]")
    console.print(f"  Enabled: {settings.summarization.enabled}")
    console.print(f"  Engine: {settings.summarization.engine}")
    console.print(f"  Model: {settings.summarization.model}")
    
    # Notion settings
    console.print("\n[cyan]Notion:[/cyan]")
    console.print(f"  Enabled: {settings.notion.enabled}")
    console.print(f"  Database ID: {'*' * 10 if settings.notion.database_id else 'Not set'}")
    console.print(f"  Token: {'*' * 10 if settings.notion_token else 'Not set'}")
    
    # Queue settings
    console.print("\n[cyan]Queue:[/cyan]")
    console.print(f"  Max Workers: {settings.queue.max_workers}")
    console.print(f"  Worker Type: {settings.queue.worker_type}")
    console.print(f"  Retry Attempts: {settings.queue.retry_attempts}")
    
    # Monitor settings
    console.print("\n[cyan]Monitor:[/cyan]")
    console.print(f"  Input Directory: {settings.monitor.input_dir}")
    console.print(f"  Output Directory: {settings.monitor.output_dir}")
    console.print(f"  Extensions: {', '.join(settings.monitor.watch_extensions)}")


if __name__ == "__main__":
    app()