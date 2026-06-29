#!/usr/bin/env python3
"""
CLI for AI Candidate Recommendation Engine

Provides command-line interface for common operations.
"""

import typer
import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.services.recommendation_pipeline import get_recommendation_pipeline
from app.schemas.job import JobDescriptionInput
from app.schemas.candidate import CandidateProfile

app = typer.Typer(help="AI Candidate Recommendation Engine CLI")
console = Console()


@app.command()
def init():
    """
    Initialize the system and load models.
    """
    setup_logging()
    console.print("[bold blue]Initializing AI Candidate Recommendation Engine...[/bold blue]")

    pipeline = get_recommendation_pipeline()

    with Progress() as progress:
        task = progress.add_task("[cyan]Loading models...", total=100)

        if pipeline.initialize():
            progress.update(task, completed=100)
            console.print("[green]✓ System initialized successfully[/green]")
            console.print(f"  - Embedding model: {pipeline.embedding_service.model_name}")
            console.print(f"  - Cross-encoder: {pipeline.ranking_engine.model_name}")
            console.print(f"  - Vector store: {settings.chroma_db_path}")
        else:
            progress.update(task, completed=100)
            console.print("[red]✗ Failed to initialize system[/red]")


@app.command()
def process_job(
    job_file: str = typer.Argument(..., help="Path to job description file"),
    job_id: str = typer.Option("job_001", help="Job identifier"),
    title: str = typer.Option("Job Title", help="Job title"),
):
    """
    Process a job description from a text file.
    """
    setup_logging()
    console.print(f"[bold blue]Processing job: {job_id}[/bold blue]")

    # Read job description
    job_path = Path(job_file)
    if not job_path.exists():
        console.print(f"[red]File not found: {job_file}[/red]")
        raise typer.Exit(1)

    description = job_path.read_text()

    # Initialize pipeline
    pipeline = get_recommendation_pipeline()
    if not pipeline.is_initialized:
        pipeline.initialize()

    # Create job input
    job_input = JobDescriptionInput(
        job_id=job_id,
        title=title,
        description=description,
    )

    # Process job
    job_description = pipeline.process_job(job_input)

    console.print("[green]✓ Job processed successfully[/green]")
    console.print(f"  - Required skills: {len(job_description.parsed_job.required_skills)}")
    console.print(f"  - Preferred skills: {len(job_description.parsed_job.preferred_skills)}")
    console.print(f"  - Seniority: {job_description.parsed_job.seniority}")
    console.print(f"  - Industry: {job_description.parsed_job.industry}")


@app.command()
def index_candidates(
    candidates_file: str = typer.Argument(..., help="Path to candidates JSONL file"),
):
    """
    Index candidate profiles from a JSONL file.
    """
    setup_logging()
    console.print(f"[bold blue]Indexing candidates from: {candidates_file}[/bold blue]")

    # Read candidates
    candidates_path = Path(candidates_file)
    if not candidates_path.exists():
        console.print(f"[red]File not found: {candidates_file}[/red]")
        raise typer.Exit(1)

    candidates = []
    with open(candidates_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    profile_data = json.loads(line)
                    profile = CandidateProfile(**profile_data)
                    candidates.append(profile)
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to parse line: {e}[/yellow]")

    if not candidates:
        console.print("[red]No valid candidates found[/red]")
        raise typer.Exit(1)

    console.print(f"Found {len(candidates)} candidates")

    # Initialize pipeline
    pipeline = get_recommendation_pipeline()
    if not pipeline.is_initialized:
        pipeline.initialize()

    # Index candidates
    with Progress() as progress:
        task = progress.add_task("[cyan]Indexing candidates...", total=len(candidates))

        indexed, failed = pipeline.index_candidates(candidates)

        progress.update(task, completed=len(candidates))

    console.print("[green]✓ Indexing completed[/green]")
    console.print(f"  - Indexed: {indexed}")
    console.print(f"  - Failed: {failed}")
    console.print(f"  - Vector store size: {pipeline.vector_store.collection_size}")


@app.command()
def recommend(
    job_id: str = typer.Argument(..., help="Job identifier"),
    top_k: int = typer.Option(100, help="Number of recommendations"),
    output: str = typer.Option(None, help="Output CSV file path"),
    no_rerank: bool = typer.Option(False, help="Disable cross-encoder re-ranking"),
):
    """
    Generate recommendations for a job.
    """
    setup_logging()
    console.print(f"[bold blue]Generating recommendations for: {job_id}[/bold blue]")

    # Initialize pipeline
    pipeline = get_recommendation_pipeline()
    if not pipeline.is_initialized:
        pipeline.initialize()

    from app.schemas.recommendation import RecommendationRequest

    request = RecommendationRequest(
        job_id=job_id,
        top_k=top_k,
        include_reasoning=True,
        use_reranking=not no_rerank,
    )

    # Generate recommendations
    with Progress() as progress:
        task = progress.add_task("[cyan]Generating recommendations...", total=100)

        result = pipeline.recommend(request)

        progress.update(task, completed=100)

    # Export to CSV
    output_path = output or pipeline.export_to_csv(result)

    console.print("[green]✓ Recommendations generated[/green]")
    console.print(f"  - Total candidates: {result.total_candidates_processed}")
    console.print(f"  - Retrieved: {result.candidates_retrieved}")
    console.print(f"  - Re-ranked: {result.candidates_reranked}")
    console.print(f"  - Final recommendations: {result.final_recommendations}")
    console.print(f"  - Execution time: {result.execution_time_ms:.2f}ms")
    console.print(f"  - Output file: {output_path}")

    # Show top 5 recommendations
    if result.recommendations:
        console.print("\n[bold]Top 5 Recommendations:[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="dim", width=4)
        table.add_column("Candidate ID", width=20)
        table.add_column("Name", width=25)
        table.add_column("Score", justify="right", width=8)
        table.add_column("Key Skills", width=40)

        for rec in result.recommendations[:5]:
            table.add_row(
                str(rec.rank),
                rec.candidate_id,
                rec.candidate_name or "N/A",
                f"{rec.final_score:.3f}",
                ", ".join(rec.top_skills[:3]),
            )

        console.print(table)


@app.command()
def export(
    job_id: str = typer.Argument(..., help="Job identifier"),
    output: str = typer.Option("./output/recommendations.csv", help="Output CSV file"),
    top_k: int = typer.Option(100, help="Number of recommendations"),
):
    """
    Export recommendations to CSV.
    """
    setup_logging()

    # Initialize pipeline
    pipeline = get_recommendation_pipeline()
    if not pipeline.is_initialized:
        pipeline.initialize()

    from app.schemas.recommendation import RecommendationRequest

    request = RecommendationRequest(
        job_id=job_id,
        top_k=top_k,
        include_reasoning=True,
        use_reranking=True,
    )

    result = pipeline.recommend(request)
    output_path = pipeline.export_to_csv(result, output)

    console.print(f"[green]✓ Exported {len(result.recommendations)} recommendations to {output_path}[/green]")


@app.command()
def status():
    """
    Show system status and metrics.
    """
    setup_logging()

    pipeline = get_recommendation_pipeline()
    if not pipeline.is_initialized:
        pipeline.initialize()

    metrics = pipeline.get_metrics()

    console.print("[bold blue]System Status[/bold blue]\n")

    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Embedding Model", metrics["embedding_model"] or "Not loaded")
    table.add_row("Cross-Encoder", metrics["cross_encoder_model"] or "Not loaded")
    table.add_row("Models Loaded", "Yes" if metrics["models_loaded"] else "No")
    table.add_row("Total Candidates", str(metrics["total_candidates"]))
    table.add_row("Total Jobs", str(metrics["total_jobs"]))
    table.add_row("Vector Store Size", str(metrics["vector_store_size"]))
    table.add_row("ChromaDB Path", settings.chroma_db_path)

    console.print(table)

    console.print("\n[bold]Scoring Weights:[/bold]")
    weights_table = Table(show_header=False, box=None)
    weights_table.add_column("Component", style="cyan")
    weights_table.add_column("Weight", style="white")

    for name, weight in settings.weights.items():
        weights_table.add_row(name.capitalize(), f"{weight:.0%}")

    console.print(weights_table)


@app.command()
def reset():
    """
    Reset the system (clear all data).
    """
    from rich.prompt import Confirm

    setup_logging()

    if not Confirm.ask("[red]This will delete all indexed data. Continue?[/red]"):
        console.print("Cancelled")
        raise typer.Exit(0)

    pipeline = get_recommendation_pipeline()
    if not pipeline.is_initialized:
        pipeline.initialize()

    pipeline.vector_store.reset()
    pipeline._jobs.clear()
    pipeline._candidates.clear()

    console.print("[green]✓ System reset successfully[/green]")


if __name__ == "__main__":
    app()
