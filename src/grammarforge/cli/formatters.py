"""Rich terminal formatting helpers for GrammarForge CLI."""

from __future__ import annotations

from typing import Any, Dict, List
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

console = Console()


def print_banner() -> None:
    """Print GrammarForge CLI header banner."""
    console.print(
        Panel.fit(
            "[bold cyan]GrammarForge[/bold cyan] [dim]v0.1.0[/dim]\n"
            "[white]Constrained Decoding Engine & GBNF Compiler for Local LLMs[/white]",
            border_style="cyan",
        )
    )


def print_gbnf(gbnf_code: str, title: str = "Compiled GBNF Grammar") -> None:
    """Print syntax-highlighted GBNF code."""
    syntax = Syntax(gbnf_code, "ebnf", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title=f"[bold green]{title}[/bold green]", border_style="green"))


def print_benchmark_table(stats: Dict[str, Any]) -> None:
    """Render latency and throughput benchmark statistics in a table."""
    table = Table(title="Logit Masking Benchmark Results", border_style="cyan")
    table.add_column("Metric", style="bold white", justify="left")
    table.add_column("Value", style="bold green", justify="right")

    table.add_row("Vocabulary Size", f"{stats['vocab_size']:,} tokens")
    table.add_row("Iterations Tested", f"{stats['iterations']:,} steps")
    table.add_row("Average Latency", f"{stats['avg_latency_us']:.2f} µs")
    table.add_row("p50 Latency", f"{stats['p50_latency_us']:.2f} µs")
    table.add_row("p95 Latency", f"{stats['p95_latency_us']:.2f} µs")
    table.add_row("p99 Latency", f"{stats['p99_latency_us']:.2f} µs")
    table.add_row("Throughput", f"{stats['throughput_steps_per_sec']:,.0f} steps/sec")

    console.print(table)


def print_mask_result(
    prefix: str,
    allowed_count: int,
    total_vocab: int,
    latency_us: float,
    is_accepting: bool,
    sample_tokens: List[str],
) -> None:
    """Print token masking inspection results."""
    table = Table(title="Logit Masking Step", border_style="blue")
    table.add_column("Property", style="bold white")
    table.add_column("Details", style="cyan")

    table.add_row("Prefix Evaluated", repr(prefix))
    table.add_row("Allowed Tokens", f"[bold green]{allowed_count:,}[/bold green] / {total_vocab:,}")
    table.add_row("Step Latency", f"[bold yellow]{latency_us:.2f} µs[/bold yellow]")
    table.add_row("Is Accepting (EOS allowed)", "[green]Yes[/green]" if is_accepting else "[red]No[/red]")
    table.add_row("Sample Allowed Tokens", ", ".join(repr(t) for t in sample_tokens[:10]))

    console.print(table)
