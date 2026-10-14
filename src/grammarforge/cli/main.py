"""GrammarForge Typer Command Line Interface."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional
import typer
from rich.console import Console
from grammarforge.cli.formatters import (
    console,
    print_banner,
    print_benchmark_table,
    print_gbnf,
    print_mask_result,
)
from grammarforge.compiler.gbnf_ir import Grammar, Rule
from grammarforge.compiler.json_schema import compile_json_schema
from grammarforge.compiler.linter import GbnfLinter, format_gbnf
from grammarforge.compiler.pydantic_bridge import compile_pydantic_model, load_model_from_spec
from grammarforge.compiler.regex_to_gbnf import compile_regex_to_gbnf_node
from grammarforge.compiler.sql_ddl import compile_sql_ddl_to_gbnf
from grammarforge.core.config import CompilerConfig, EngineConfig, KeyOrdering, WhitespaceMode
from grammarforge.engine.masking import LogitMasker
from grammarforge.engine.repair import heal_partial_json
from grammarforge.engine.state_machine import JsonStateMachine, ParserStatus
from grammarforge.engine.trie import VocabularyTrie
from grammarforge.gateway.proxy import create_gateway_proxy

app = typer.Typer(
    name="grammarforge",
    help="High-throughput constrained decoding engine & JSON Schema to GBNF compiler for local LLMs.",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", "-v", help="Show GrammarForge version."),
) -> None:
    if version:
        console.print("[bold cyan]GrammarForge[/bold cyan] v0.1.0")
        raise typer.Exit()


@app.command("compile")
def compile_cmd(
    schema_path: str = typer.Argument(..., help="Path to JSON Schema file or '-' for stdin."),
    output: Optional[str] = typer.Option(None, "-o", "--output", help="Output path for compiled .gbnf file."),
    whitespace: WhitespaceMode = typer.Option(
        WhitespaceMode.LENIENT, "--whitespace", "-w", help="Whitespace handling mode."
    ),
    ordering: KeyOrdering = typer.Option(
        KeyOrdering.CANONICAL, "--ordering", help="Property key ordering mode."
    ),
    root_rule: str = typer.Option("root", "--root", help="Name of root rule."),
) -> None:
    """Compile a JSON Schema file to a GBNF grammar specification."""
    try:
        if schema_path == "-":
            raw_content = sys.stdin.read()
        else:
            if not os.path.exists(schema_path):
                console.print(f"[bold red]Error:[/bold red] Schema file not found: {schema_path}")
                raise typer.Exit(1)
            with open(schema_path, "r", encoding="utf-8") as f:
                raw_content = f.read()

        schema = json.loads(raw_content)
    except json.JSONDecodeError as err:
        console.print(f"[bold red]Error parsing JSON Schema:[/bold red] {err}")
        raise typer.Exit(1)

    config = CompilerConfig(
        whitespace_mode=whitespace,
        key_ordering=ordering,
        root_rule_name=root_rule,
    )

    try:
        gbnf = compile_json_schema(schema, config=config)
    except Exception as err:
        console.print(f"[bold red]Compilation error:[/bold red] {err}")
        raise typer.Exit(1)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(gbnf)
        console.print(f"[bold green]Saved GBNF grammar to:[/bold green] {output}")
    else:
        print_gbnf(gbnf, title=f"Compiled GBNF: {schema_path}")


@app.command("pydantic")
def pydantic_cmd(
    spec: str = typer.Argument(..., help="Model spec e.g. 'path/to/file.py:MyModel' or 'module:Model'."),
    output: Optional[str] = typer.Option(None, "-o", "--output", help="Output path for compiled .gbnf file."),
    whitespace: WhitespaceMode = typer.Option(
        WhitespaceMode.LENIENT, "--whitespace", "-w", help="Whitespace handling mode."
    ),
) -> None:
    """Compile a Pydantic model directly to GBNF grammar."""
    try:
        model_cls = load_model_from_spec(spec)
        config = CompilerConfig(whitespace_mode=whitespace)
        gbnf = compile_pydantic_model(model_cls, config=config)
    except Exception as err:
        console.print(f"[bold red]Error compiling Pydantic model:[/bold red] {err}")
        raise typer.Exit(1)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(gbnf)
        console.print(f"[bold green]Saved GBNF grammar to:[/bold green] {output}")
    else:
        print_gbnf(gbnf, title=f"Compiled GBNF from Pydantic: {spec}")


@app.command("regex")
def regex_cmd(
    pattern: str = typer.Argument(..., help="Regular expression pattern to compile."),
    output: Optional[str] = typer.Option(None, "-o", "--output", help="Output path for compiled .gbnf file."),
    rule_name: str = typer.Option("root", "--rule-name", help="Name of the root rule."),
) -> None:
    """Compile a regular expression pattern to GBNF."""
    try:
        node = compile_regex_to_gbnf_node(pattern)
        grammar = Grammar()
        grammar.add_rule(Rule(rule_name, node, comment=f"regex: {pattern}"))
        gbnf = grammar.serialize(root_name=rule_name)
    except Exception as err:
        console.print(f"[bold red]Error compiling regex:[/bold red] {err}")
        raise typer.Exit(1)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(gbnf)
        console.print(f"[bold green]Saved GBNF grammar to:[/bold green] {output}")
    else:
        print_gbnf(gbnf, title=f"Compiled Regex: {pattern}")


@app.command("validate")
def validate_cmd(
    schema_path: str = typer.Argument(..., help="Path to JSON Schema file."),
    target: str = typer.Argument(..., help="Path to JSON file or raw JSON string to validate."),
) -> None:
    """Validate a JSON string or file against a JSON Schema and grammar."""
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
    except Exception as err:
        console.print(f"[bold red]Error loading schema:[/bold red] {err}")
        raise typer.Exit(1)

    if os.path.exists(target):
        with open(target, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = target

    sm = JsonStateMachine(schema)
    status, err = sm.check_partial_json(text)

    if status == ParserStatus.ACCEPTING:
        console.print("[bold green]Validation SUCCESS:[/bold green] Payload is valid complete JSON conforming to schema.")
    elif status == ParserStatus.VALID:
        console.print("[bold yellow]Partial Match:[/bold yellow] Payload is a valid incomplete prefix of schema target.")
    else:
        console.print(f"[bold red]Validation FAILED:[/bold red] {err}")
        raise typer.Exit(1)


@app.command("mask")
def mask_cmd(
    schema_path: str = typer.Argument(..., help="Path to JSON Schema file."),
    prefix: str = typer.Option(..., "--prefix", "-p", help="Current generated text prefix."),
    vocab_file: Optional[str] = typer.Option(None, "--vocab-file", help="Path to tokenizer vocab.json."),
    vocab_size: int = typer.Option(32000, "--vocab-size", help="Synthetic vocabulary size if no file given."),
) -> None:
    """Evaluate allowed next tokens and measure logit masking latency."""
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    if vocab_file and os.path.exists(vocab_file):
        trie = VocabularyTrie.from_huggingface_json(vocab_file)
    else:
        trie = VocabularyTrie.create_synthetic_vocab(vocab_size)

    masker = LogitMasker(schema, trie)
    mask, result = masker.compute_mask(prefix)

    sample_tokens = [trie.get_token_str(tid) for tid in result.allowed_tokens[:10]]
    print_mask_result(
        prefix=prefix,
        allowed_count=len(result.allowed_tokens),
        total_vocab=trie.vocab_size,
        latency_us=result.latency_us,
        is_accepting=result.is_accepting,
        sample_tokens=sample_tokens,
    )


@app.command("bench")
def bench_cmd(
    schema_path: Optional[str] = typer.Option(None, "--schema", "-s", help="Path to JSON Schema file."),
    vocab_size: int = typer.Option(32000, "--vocab-size", help="Tokenizer vocabulary size."),
    iterations: int = typer.Option(500, "--iterations", "-n", help="Number of steps to benchmark."),
) -> None:
    """Benchmark token logit masking latency and throughput."""
    print_banner()

    if schema_path and os.path.exists(schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
    else:
        # Standard realistic schema for benchmarking
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string"},
                "email": {"type": "string", "format": "email"},
                "status": {"type": "string", "enum": ["active", "pending", "disabled"]},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["id", "name", "email", "status"],
        }

    console.print(f"[cyan]Initializing synthetic vocabulary with {vocab_size:,} tokens...[/cyan]")
    trie = VocabularyTrie.create_synthetic_vocab(vocab_size)
    masker = LogitMasker(schema, trie)

    # Sequence of generation prefixes
    prefixes = [
        "",
        '{"',
        '{"id":',
        '{"id": 42, ',
        '{"id": 42, "name": ',
        '{"id": 42, "name": "Alex", ',
        '{"id": 42, "name": "Alex", "email": ',
        '{"id": 42, "name": "Alex", "email": "alex@example.com", "status": ',
        '{"id": 42, "name": "Alex", "email": "alex@example.com", "status": "active"}',
    ]

    console.print(f"[cyan]Running {iterations:,} masking steps across {len(prefixes)} generation states...[/cyan]")

    latencies_us = []
    total_start = time.perf_counter()

    for i in range(iterations):
        prefix = prefixes[i % len(prefixes)]
        t0 = time.perf_counter_ns()
        masker.compute_mask(prefix)
        elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
        latencies_us.append(elapsed_us)

    total_time_s = time.perf_counter() - total_start
    latencies_us.sort()

    stats = {
        "vocab_size": vocab_size,
        "iterations": iterations,
        "avg_latency_us": sum(latencies_us) / len(latencies_us),
        "p50_latency_us": latencies_us[int(len(latencies_us) * 0.50)],
        "p95_latency_us": latencies_us[int(len(latencies_us) * 0.95)],
        "p99_latency_us": latencies_us[int(len(latencies_us) * 0.99)],
        "throughput_steps_per_sec": iterations / total_time_s if total_time_s > 0 else 0,
    }

    print_benchmark_table(stats)


@app.command("repair")
def repair_cmd(
    target: str = typer.Argument(..., help="Path to truncated JSON file or raw JSON text."),
    schema_path: Optional[str] = typer.Option(None, "--schema", "-s", help="Optional JSON Schema to enforce."),
    output: Optional[str] = typer.Option(None, "-o", "--output", help="Output file to save repaired JSON."),
) -> None:
    """Repair and heal an incomplete or truncated JSON string."""
    if os.path.exists(target):
        with open(target, "r", encoding="utf-8") as f:
            raw_text = f.read()
    else:
        raw_text = target

    schema_data = None
    if schema_path and os.path.exists(schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_data = json.load(f)

    result = heal_partial_json(raw_text, schema=schema_data)

    if result.is_valid_json:
        console.print(f"[bold green]Repair SUCCESS:[/bold green] Added suffix: [yellow]{repr(result.suffix_added)}[/yellow]")
        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(result.repaired_text)
            console.print(f"[bold green]Saved repaired JSON to:[/bold green] {output}")
        else:
            console.print(result.repaired_text)
    else:
        console.print(f"[bold red]Repair FAILED:[/bold red] {result.error}")
        raise typer.Exit(1)


@app.command("lint")
def lint_cmd(
    gbnf_path: str = typer.Argument(..., help="Path to .gbnf file to lint."),
) -> None:
    """Analyze and lint a GBNF grammar for syntax, unreachable, and left-recursion errors."""
    if not os.path.exists(gbnf_path):
        console.print(f"[bold red]Error:[/bold red] File not found: {gbnf_path}")
        raise typer.Exit(1)

    with open(gbnf_path, "r", encoding="utf-8") as f:
        content = f.read()

    linter = GbnfLinter(content)
    diagnostics = linter.lint()

    if not diagnostics:
        console.print(f"[bold green]No issues found![/bold green] '{gbnf_path}' is valid and well-formed.")
        return

    has_errors = False
    for diag in diagnostics:
        color = "red" if diag.severity == "error" else "yellow"
        if diag.severity == "error":
            has_errors = True
        console.print(f"[{color}][bold]{diag.severity.upper()}[/bold][/{color}] line {diag.line_number or '?'}: {diag.message}")

    if has_errors:
        raise typer.Exit(1)


@app.command("fmt")
def fmt_cmd(
    gbnf_path: str = typer.Argument(..., help="Path to .gbnf file to format."),
    write: bool = typer.Option(False, "--write", "-w", help="Overwrite file with formatted grammar."),
) -> None:
    """Format and align GBNF grammar syntax."""
    if not os.path.exists(gbnf_path):
        console.print(f"[bold red]Error:[/bold red] File not found: {gbnf_path}")
        raise typer.Exit(1)

    with open(gbnf_path, "r", encoding="utf-8") as f:
        content = f.read()

    formatted = format_gbnf(content)
    if write:
        with open(gbnf_path, "w", encoding="utf-8") as f:
            f.write(formatted)
        console.print(f"[bold green]Formatted and updated:[/bold green] {gbnf_path}")
    else:
        print_gbnf(formatted, title=f"Formatted GBNF: {gbnf_path}")


@app.command("sql")
def sql_cmd(
    target: str = typer.Argument(..., help="SQL CREATE TABLE statement or path to .sql file."),
    output: Optional[str] = typer.Option(None, "-o", "--output", help="Output .gbnf file path."),
    format_type: str = typer.Option("json", "--format", "-f", help="'json' row object or 'insert' SQL statement."),
) -> None:
    """Compile SQL DDL schema into GBNF grammar."""
    if os.path.exists(target):
        with open(target, "r", encoding="utf-8") as f:
            sql_text = f.read()
    else:
        sql_text = target

    try:
        gbnf = compile_sql_ddl_to_gbnf(sql_text, output_format=format_type)
    except Exception as err:
        console.print(f"[bold red]SQL Compilation Error:[/bold red] {err}")
        raise typer.Exit(1)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(gbnf)
        console.print(f"[bold green]Saved GBNF grammar to:[/bold green] {output}")
    else:
        print_gbnf(gbnf, title=f"Compiled SQL DDL ({format_type})")


@app.command("proxy")
def proxy_cmd(
    upstream: str = typer.Option("http://127.0.0.1:11434", "--upstream", "-u", help="Upstream LLM server URL."),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Proxy bind host address."),
    port: int = typer.Option(8080, "--port", "-p", help="Proxy bind port."),
) -> None:
    """Start the OpenAI-compatible reverse proxy gateway with GBNF enforcement."""
    import uvicorn
    print_banner()
    console.print(f"[bold cyan]Starting GrammarForge Gateway Proxy[/bold cyan]")
    console.print(f"  Upstream server: [green]{upstream}[/green]")
    console.print(f"  Listening on:    [bold green]http://{host}:{port}[/bold green]")
    app_instance = create_gateway_proxy(upstream_url=upstream)
    uvicorn.run(app_instance, host=host, port=port)



@app.command("studio")
def studio_cmd(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind host address."),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port number."),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development."),
) -> None:
    """Launch the interactive GrammarForge Web Studio."""
    import uvicorn
    print_banner()
    console.print(f"[bold green]Starting Web Studio on http://{host}:{port}[/bold green]")
    uvicorn.run("grammarforge.studio.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
