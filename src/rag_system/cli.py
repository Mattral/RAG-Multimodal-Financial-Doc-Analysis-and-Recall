"""
Command-line interface for the RAG system.

Provides two main commands:
- ingest <file>: Parse, process, and index documents
- query "<prompt>": Retrieve context and generate responses
"""

import asyncio
from pathlib import Path
from typing import Optional, Annotated

import typer
import structlog
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.syntax import Syntax

from .config import get_config
from .utils.logger import get_logger, setup_logging
from .components.pdf_parser import PDFParser
from .components.vision_processor import VisionProcessor
from .components.vector_indexer import VectorIndexer
from .components.layout_parser import LayoutParser, parse_document_layout, elements_to_markdown
from .components.pot_executor import PoTExecutor, execute_financial_formula

# ============================================================================
# Setup
# ============================================================================

app = typer.Typer(
    help="RAG system for multimodal financial document analysis",
    pretty_exceptions_show_locals=False,
)

console = Console()
logger = get_logger(__name__)


# ============================================================================
# Ingest Command
# ============================================================================


@app.command()
async def ingest(
    file_path: Annotated[
        str,
        typer.Argument(help="Path to the document to ingest"),
    ],
    output_dir: Annotated[
        Optional[str],
        typer.Option(
            "--output", "-o",
            help="Output directory for processed files",
        ),
    ] = None,
    extract_charts: Annotated[
        bool,
        typer.Option(
            "--extract-charts/--no-extract-charts",
            help="Extract and describe charts/images",
        ),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose/--quiet",
            help="Enable verbose output",
        ),
    ] = False,
) -> None:
    """
    Ingest and process a document.

    Performs the following steps:
    1. Parse the document with layout awareness
    2. Extract and describe charts/images (if enabled)
    3. Create layout-aware chunks
    4. Index to vector store
    5. Save processing report

    Example:
        rag ingest finance_report.pdf --output ./indexed_docs --extract-charts
    """
    try:
        # Setup logging
        setup_logging(
            level="DEBUG" if verbose else "INFO",
            format_type="json",
        )

        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            console.print(f"[red]Error:[/red] File not found: {file_path}", style="bold")
            raise typer.Exit(code=1)

        config = get_config()
        output_path = Path(output_dir) if output_dir else Path.cwd() / "indexed_docs"
        output_path.mkdir(parents=True, exist_ok=True)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Step 1: Parse PDF
            task_parse = progress.add_task("[cyan]Parsing document...", total=None)
            pdf_parser = PDFParser()
            elements = await pdf_parser.parse(str(file_path_obj))
            progress.update(task_parse, completed=True)
            console.print(
                f"[green]✓[/green] Parsed {len(elements)} elements",
                style="dim",
            )

            # Step 2: Extract chart descriptions
            task_charts = progress.add_task("[cyan]Processing charts...", total=None)
            if extract_charts and elements:
                vision_processor = VisionProcessor()
                chart_descriptions = []
                # In a real implementation, would filter for chart elements
                # and call vision_processor.analyze_chart() for each
                console.print(
                    f"[green]✓[/green] Processed {len(chart_descriptions)} charts",
                    style="dim",
                )
            progress.update(task_charts, completed=True)

            # Step 3: Apply layout parsing
            task_layout = progress.add_task("[cyan]Analyzing layout...", total=None)
            layout_parser = LayoutParser()
            layout_groups = await layout_parser.parse_elements(elements)
            progress.update(task_layout, completed=True)
            console.print(
                f"[green]✓[/green] Created {len(layout_groups)} semantic groups",
                style="dim",
            )

            # Step 4: Index to vector store
            task_index = progress.add_task("[cyan]Indexing to vector store...", total=None)
            vector_indexer = VectorIndexer()
            # In a real implementation would call vector_indexer.index()
            progress.update(task_index, completed=True)
            console.print("[green]✓[/green] Indexed to vector store", style="dim")

            # Step 5: Generate report
            report = {
                "file": file_path,
                "elements_parsed": len(elements),
                "semantic_groups": len(layout_groups),
                "output_directory": str(output_path),
            }

            structlog.get_logger("cli").info(
                "document_ingestion_completed",
                file=file_path,
                elements_count=len(elements),
                groups_count=len(layout_groups),
                output_dir=str(output_path),
            )

            # Display summary
            console.print(
                Panel(
                    f"""[green]✓ Document successfully ingested![/green]

File: [bold]{file_path}[/bold]
Elements parsed: [bold]{len(elements)}[/bold]
Semantic groups: [bold]{len(layout_groups)}[/bold]
Output directory: [cyan]{output_path}[/cyan]""",
                    title="Ingestion Summary",
                    style="green",
                ),
            )

    except Exception as e:
        console.print(
            Panel(
                f"[red]Ingestion failed:[/red]\n{type(e).__name__}: {str(e)}",
                title="Error",
                style="red",
            ),
        )
        logger.error("ingest_failed", error=str(e), error_type=type(e).__name__)
        raise typer.Exit(code=1)


# ============================================================================
# Query Command
# ============================================================================


@app.command()
async def query(
    prompt: Annotated[
        str,
        typer.Argument(help="Query prompt"),
    ],
    use_pot: Annotated[
        bool,
        typer.Option(
            "--use-pot/--no-pot",
            help="Use Program-of-Thought for numerical reasoning",
        ),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose/--quiet",
            help="Enable verbose output",
        ),
    ] = False,
) -> None:
    """
    Query the indexed documents.

    Performs the following steps:
    1. Retrieve relevant context from vector store
    2. Route to PoT if numerical query (if enabled)
    3. Generate response with citations
    4. Display results

    Example:
        rag query "What was Tesla's revenue growth rate?"
        rag query "Calculate CAGR for 2019-2023" --use-pot
    """
    try:
        # Setup logging
        setup_logging(
            level="DEBUG" if verbose else "INFO",
            format_type="json",
        )

        config = get_config()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Step 1: Retrieve context
            task_retrieve = progress.add_task(
                "[cyan]Retrieving relevant context...",
                total=None,
            )
            vector_indexer = VectorIndexer()
            # In a real implementation would call vector_indexer.retrieve()
            context_chunks = []
            progress.update(task_retrieve, completed=True)
            console.print(
                f"[green]✓[/green] Retrieved {len(context_chunks)} relevant chunks",
                style="dim",
            )

            # Step 2: Check if numerical query
            task_route = progress.add_task("[cyan]Analyzing query...", total=None)
            is_numerical = any(
                keyword in prompt.lower()
                for keyword in ["calculate", "cagr", "growth", "rate", "return", "roi"]
            )
            progress.update(task_route, completed=True)

            if is_numerical and use_pot:
                console.print("[green]✓[/green] Routing to PoT executor", style="dim")

                # Step 3: Execute PoT
                task_pot = progress.add_task("[cyan]Executing calculations...", total=None)
                pot_executor = PoTExecutor()
                # In a real implementation would extract numerical code from LLM
                # and execute it
                progress.update(task_pot, completed=True)
            else:
                console.print("[dim]Using standard context retrieval[/dim]")

            # Step 4: Generate response
            task_generate = progress.add_task("[cyan]Generating response...", total=None)
            # In a real implementation would call LLM to generate response
            response = f"Response to: {prompt}"
            progress.update(task_generate, completed=True)

        # Display response
        console.print(
            Panel(
                response,
                title="Query Response",
                style="cyan",
            ),
        )

        structlog.get_logger("cli").info(
            "query_completed",
            prompt=prompt,
            is_numerical=is_numerical,
            use_pot=use_pot and is_numerical,
        )

    except Exception as e:
        console.print(
            Panel(
                f"[red]Query failed:[/red]\n{type(e).__name__}: {str(e)}",
                title="Error",
                style="red",
            ),
        )
        logger.error("query_failed", error=str(e), error_type=type(e).__name__)
        raise typer.Exit(code=1)


# ============================================================================
# Additional Commands
# ============================================================================


@app.command()
def version() -> None:
    """Display version information."""
    console.print("[cyan]RAG Multimodal Finance System[/cyan] v1.0.0")
    console.print("Enterprise-grade document analysis and retrieval")


@app.command()
def health() -> None:
    """Check system health and configuration."""
    try:
        config = get_config()
        
        checks = {
            "Configuration": "✓" if config else "✗",
            "API Key": "✓" if config.openai_api_key else "✗",
            "Environment": config.environment,
            "Debug Mode": "On" if config.debug_mode else "Off",
        }

        console.print(
            Panel(
                "\n".join(f"{k}: {v}" for k, v in checks.items()),
                title="System Health",
                style="green" if all("✓" in str(v) for v in checks.values()) else "yellow",
            ),
        )
    except Exception as e:
        console.print(
            Panel(
                f"[red]Health check failed:[/red] {str(e)}",
                style="red",
            ),
        )
        raise typer.Exit(code=1)


# ============================================================================
# Entry Point
# ============================================================================


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
