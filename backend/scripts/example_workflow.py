#!/usr/bin/env python3
"""
Example Workflow Script

Demonstrates the complete workflow of the AI Candidate Recommendation Engine.
"""

import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from app.core.logging_config import setup_logging
from app.services.recommendation_pipeline import get_recommendation_pipeline
from app.schemas.job import JobDescriptionInput
from app.schemas.candidate import CandidateProfile
from app.schemas.recommendation import RecommendationRequest

console = Console()


def main():
    """Run the complete recommendation workflow."""
    setup_logging()

    console.print(Panel.fit(
        "[bold blue]AI Candidate Recommendation Engine - Example Workflow[/bold blue]",
        title="Workflow Demo",
    ))

    # Initialize pipeline
    console.print("\n[bold]Step 1: Initializing System...[/bold]")
    pipeline = get_recommendation_pipeline()

    if not pipeline.initialize():
        console.print("[red]Failed to initialize pipeline[/red]")
        return

    console.print("[green]✓ System initialized[/green]")
    console.print(f"  - Embedding model: {pipeline.embedding_service.model_name}")
    console.print(f"  - Cross-encoder: {pipeline.ranking_engine.model_name}")

    # Load job description
    console.print("\n[bold]Step 2: Processing Job Description...[/bold]")
    job_file = Path("data/job_description.txt")

    if not job_file.exists():
        console.print(f"[red]Job file not found: {job_file}[/red]")
        return

    job_text = job_file.read_text()

    job_input = JobDescriptionInput(
        job_id="ml_engineer_001",
        title="Senior Machine Learning Engineer",
        description=job_text,
        company="TechCorp",
    )

    job_description = pipeline.process_job(job_input)
    console.print("[green]✓ Job processed[/green]")
    console.print(f"  - Title: {job_description.parsed_job.title}")
    console.print(f"  - Required skills: {len(job_description.parsed_job.required_skills)}")
    console.print(f"  - Preferred skills: {len(job_description.parsed_job.preferred_skills)}")
    console.print(f"  - Seniority: {job_description.parsed_job.seniority}")

    # Load candidates
    console.print("\n[bold]Step 3: Loading Candidates...[/bold]")
    candidates_file = Path("data/candidates.jsonl")

    if not candidates_file.exists():
        console.print(f"[red]Candidates file not found: {candidates_file}[/red]")
        return

    candidates = []
    with open(candidates_file, "r") as f:
        for line in f:
            if line.strip():
                profile_data = json.loads(line)
                profile = CandidateProfile(**profile_data)
                candidates.append(profile)

    console.print(f"  - Loaded {len(candidates)} candidates")

    # Index candidates
    console.print("\n[bold]Step 4: Indexing Candidates...[/bold]")
    indexed, failed = pipeline.index_candidates(candidates)
    console.print(f"[green]✓ Indexed {indexed} candidates[/green]")
    if failed > 0:
        console.print(f"[yellow]  - {failed} failed[/yellow]")

    # Generate recommendations
    console.print("\n[bold]Step 5: Generating Recommendations...[/bold]")
    request = RecommendationRequest(
        job_id="ml_engineer_001",
        top_k=10,
        include_reasoning=True,
        use_reranking=True,
    )

    result = pipeline.recommend(request)
    console.print(f"[green]✓ Generated {len(result.recommendations)} recommendations[/green]")
    console.print(f"  - Execution time: {result.execution_time_ms:.2f}ms")
    console.print(f"  - Candidates retrieved: {result.candidates_retrieved}")
    console.print(f"  - Candidates re-ranked: {result.candidates_reranked}")

    # Display top recommendations
    console.print("\n[bold]Top Recommendations:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Candidate", width=25)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Key Skills", width=35)
    table.add_column("Reasoning", width=50)

    for rec in result.recommendations[:5]:
        table.add_row(
            str(rec.rank),
            f"{rec.candidate_name}\n[dim]{rec.candidate_headline}[/dim]",
            f"{rec.final_score:.3f}",
            ", ".join(rec.top_skills[:3]),
            rec.reasoning[:100] + "..." if rec.reasoning and len(rec.reasoning) > 100 else (rec.reasoning or ""),
        )

    console.print(table)

    # Export to CSV
    console.print("\n[bold]Step 6: Exporting Results...[/bold]")
    output_path = pipeline.export_to_csv(result)
    console.print(f"[green]✓ Exported to: {output_path}[/green]")

    # Show metrics
    console.print("\n[bold]System Metrics:[/bold]")
    metrics = pipeline.get_metrics()
    metrics_table = Table(show_header=False, box=None)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="white")

    metrics_table.add_row("Total Candidates", str(metrics["total_candidates"]))
    metrics_table.add_row("Total Jobs", str(metrics["total_jobs"]))
    metrics_table.add_row("Vector Store Size", str(metrics["vector_store_size"]))

    console.print(metrics_table)

    console.print("\n[bold green]✓ Workflow completed successfully![/bold green]")


if __name__ == "__main__":
    main()
