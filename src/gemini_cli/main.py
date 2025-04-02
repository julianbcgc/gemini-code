"""
Main entry point for the Gemini CLI application.
Targets Gemini 2.5 Pro Experimental. Includes ASCII Art welcome.
Passes console object to model.
"""

import os
import sys
import click
import json
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from pathlib import Path
import yaml
import google.generativeai as genai
import logging
import time

from .models.gemini import GeminiModel, list_available_models
from .models.vertex import VertexAIModel, VERTEX_SDK_AVAILABLE
from .config import Config
from .utils import count_tokens
from .tools import AVAILABLE_TOOLS
from .tools.code_analyzer import analyze_codebase

# Setup console and config
console = Console() # Create console instance HERE
try:
    config = Config()
except Exception as e:
    console.print(f"[bold red]Error loading configuration:[/bold red] {e}")
    config = None

# Setup logging - MORE EXPLICIT CONFIGURATION
log_level = os.environ.get("LOG_LEVEL", "WARNING").upper() # <-- Default back to WARNING
log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'

# Get root logger and set level
root_logger = logging.getLogger()
root_logger.setLevel(log_level)

# Remove existing handlers to avoid duplicates if basicConfig was called elsewhere
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Add a stream handler to output to console
stream_handler = logging.StreamHandler(sys.stdout) 
stream_handler.setLevel(log_level)
formatter = logging.Formatter(log_format)
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)

log = logging.getLogger(__name__) # Get logger for this module
log.info(f"Logging initialized with level: {log_level}") # Confirm level

# --- Default Model ---
DEFAULT_MODEL = "gemini-2.5-pro-exp-03-25"
# --- ---

# --- ASCII Art Definition ---
GEMINI_CODE_ART = r"""

[medium_purple]
  ██████╗ ███████╗███╗   ███╗██╗███╗   ██╗██╗        ██████╗  ██████╗ ██████╗ ███████╗
 ██╔════╝ ██╔════╝████╗ ████║██║████╗  ██║██║       ██╔════╝ ██╔═══██╗██╔══██╗██╔════╝
 ██║ ███╗███████╗██╔████╔██║██║██╔██╗ ██║██║       ██║      ██║   ██║██║  ██║███████╗
 ██║  ██║██╔════╝██║╚██╔╝██║██║██║╚██╗██║██║       ██║      ██║   ██║██║  ██║██╔════╝
 ╚██████╔╝███████╗██║ ╚═╝ ██║██║██║ ╚████║██║       ╚██████╗ ╚██████╔╝██████╔╝███████╗
  ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝        ╚═════╝  ╚═════╝ ╚═════╝ ╚══════╝
[/medium_purple]
"""
# --- End ASCII Art ---


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
@click.option(
    '--model', '-m',
    help=f'Model ID to use (e.g., gemini-2.5-pro-exp-03-25). Default: {DEFAULT_MODEL}',
    default=None
)
@click.option(
    '--provider', '-p',
    type=click.Choice(['google', 'vertex']),
    help='Provider to use (google or vertex). Defaults to configured provider.'
)
@click.option(
    '--init', is_flag=True,
    help='Generate project context before starting the interactive session'
)
@click.pass_context
def cli(ctx, model, provider, init):
    """Interactive CLI for Gemini models with coding assistance tools."""
    if not config:
        console.print("[bold red]Configuration could not be loaded. Cannot proceed.[/bold red]")
        sys.exit(1)

    # Set provider if specified
    if provider:
        try:
            config.set_provider(provider)
            console.print(f"[green]Provider set to: [bold]{provider}[/bold][/green]")
        except Exception as e:
            console.print(f"[bold red]Error setting provider:[/bold red] {e}")
            sys.exit(1)

    if ctx.invoked_subcommand is None:
        model_name_to_use = model or config.get_default_model() or DEFAULT_MODEL
        log.info(f"Attempting to start interactive session with model: {model_name_to_use}")
        
        # Generate context file if --init flag is specified
        if init:
            try:
                console.print("[yellow]Generating project context before starting...[/yellow]")
                output_file = analyze_codebase(
                    root_dir=".",
                    output_file="GEMINI.md",
                    max_files=300
                )
                console.print(f"[green]✓[/green] Context file generated: [bold]{output_file}[/bold]")
                time.sleep(1)  # Brief pause to show the message
            except Exception as e:
                console.print(f"[bold red]Error generating context:[/bold red] {e}")
                console.print("[yellow]Continuing without context...[/yellow]")
                time.sleep(1)
                
        # Pass the console object to start_interactive_session
        start_interactive_session(model_name_to_use, console)

# ... (setup, set_default_model, list_models functions remain the same) ...
@cli.command()
@click.argument('key', required=True)
@click.option('--provider', '-p', type=click.Choice(['google', 'vertex']), default='google', 
              help='Provider to use (google or vertex)')
def setup(key, provider):
    if not config: console.print("[bold red]Config error.[/bold red]"); return
    try: 
        config.set_api_key(provider, key)
        console.print(f"[green]✓[/green] {provider.title()} API key saved.")
        if provider == 'vertex':
            console.print("[yellow]Note:[/yellow] For Vertex AI, you also need to set the project ID.")
            console.print("Run [bold]gemini setup-vertex PROJECT_ID [LOCATION][/bold]")
    except Exception as e: console.print(f"[bold red]Error saving API key:[/bold red] {e}")

@cli.command()
@click.argument('project_id', required=True)
@click.argument('location', required=False, default="us-central1")
def setup_vertex(project_id, location):
    """Set up Vertex AI project and location."""
    if not config: console.print("[bold red]Config error.[/bold red]"); return
    
    # Check for Vertex SDK availability
    if not VERTEX_SDK_AVAILABLE:
        console.print("[bold red]Error:[/bold red] Vertex AI SDK not installed or incompatible version.")
        console.print("Run [bold]pip install google-cloud-aiplatform>=1.56.0[/bold] first.")
        console.print("If you installed it with pipx, run [bold]pipx inject gemini-code google-cloud-aiplatform>=1.56.0[/bold]")
        return
        
    try:
        config.set_vertex_config(project_id, location)
        config.set_provider("vertex")  # Switch to vertex provider
        console.print(f"[green]✓[/green] Vertex AI configured with:")
        console.print(f"  Project ID: [bold]{project_id}[/bold]")
        console.print(f"  Location: [bold]{location}[/bold]")
        console.print(f"  Provider set to: [bold]vertex[/bold]")
    except Exception as e: console.print(f"[bold red]Error setting up Vertex AI:[/bold red] {e}")

@cli.command()
@click.argument('model_name', required=True)
def set_default_model(model_name):
    if not config: console.print("[bold red]Config error.[/bold red]"); return
    try: config.set_default_model(model_name); console.print(f"[green]✓[/green] Default model set to [bold]{model_name}[/bold].")
    except Exception as e: console.print(f"[bold red]Error setting default model:[/bold red] {e}")

@cli.command()
@click.option('--provider', '-p', type=click.Choice(['google', 'vertex']), 
              help='Provider to use (google or vertex). Defaults to configured provider.')
def list_models(provider):
    if not config: console.print("[bold red]Config error.[/bold red]"); return
    
    # Determine provider
    provider = provider or config.get_provider()
    console.print(f"[yellow]Using provider: {provider}[/yellow]")

@cli.command()
@click.option('--output', '-o', default='GEMINI.md', help='Output file path')
@click.option('--max-files', default=200, help='Maximum number of files to analyze')
@click.option('--ignore', '-i', multiple=True, help='Additional patterns to ignore')
def generate_context(output, max_files, ignore):
    """Generate a GEMINI.md context file by analyzing the codebase."""
    if not config: console.print("[bold red]Config error.[/bold red]"); return
    
    console.print("[yellow]Analyzing codebase to generate context file...[/yellow]")
    
    try:
        # Run code analyzer
        ignore_patterns = list(ignore) if ignore else None
        output_file = analyze_codebase(
            root_dir=".",
            output_file=output,
            ignore_patterns=ignore_patterns,
            max_files=max_files
        )
        
        console.print(f"[green]✓[/green] Context file generated: [bold]{output_file}[/bold]")
        
        # Check if the context file was generated
        if os.path.exists(output_file):
            console.print("\n[bold cyan]Context file analysis:[/bold cyan]")
            
            # Show file size
            file_size = os.path.getsize(output_file) / 1024  # KB
            console.print(f"- Size: [bold]{file_size:.2f} KB[/bold]")
            
            # Count number of sections
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
                section_count = content.count('## ')
                console.print(f"- Sections: [bold]{section_count}[/bold]")
                
                # Show a preview of the sections
                console.print("\n[bold cyan]Sections in context file:[/bold cyan]")
                for line in content.splitlines():
                    if line.startswith('## '):
                        console.print(f"- {line[3:]}")
            
            console.print("\n[bold green]This context file will be automatically used when running Gemini Code.[/bold green]")
            console.print("You can edit the file to add or modify information as needed.")
        
    except Exception as e:
        console.print(f"[bold red]Error generating context file:[/bold red] {e}")
        log.error("Error in generate_context", exc_info=True)
    
    if provider == "google":
        api_key = config.get_api_key("google")
        if not api_key: console.print("[bold red]Error:[/bold red] Google API key not found. Run 'gemini setup KEY'."); return
        console.print("[yellow]Fetching Google AI models...[/yellow]")
        try:
            models_list = list_available_models(api_key)
            if not models_list or (isinstance(models_list, list) and len(models_list) > 0 and isinstance(models_list[0], dict) and "error" in models_list[0]):
                 console.print(f"[red]Error listing models:[/red] {models_list[0].get('error', 'Unknown error') if models_list else 'No models found or fetch error.'}"); return
            console.print("\n[bold cyan]Available Google AI Models (Access may vary):[/bold cyan]")
            for model_data in models_list: console.print(f"- [bold green]{model_data['name']}[/bold green] (Display: {model_data.get('display_name', 'N/A')})")
        except Exception as e: console.print(f"[bold red]Error listing Google AI models:[/bold red] {e}"); log.error("List models failed", exc_info=True)
    
    elif provider == "vertex":
        if not VERTEX_SDK_AVAILABLE:
            console.print("[bold red]Error:[/bold red] Vertex AI SDK not installed.")
            console.print("Run [bold]pip install google-cloud-aiplatform[/bold] first.")
            return
            
        # Check for Vertex configuration
        vertex_config = config.get_vertex_config()
        if not vertex_config or not vertex_config.get("project_id"):
            console.print("[bold red]Error:[/bold red] Vertex AI not configured. Run 'gemini setup-vertex PROJECT_ID'.")
            return
            
        console.print("[yellow]Fetching Vertex AI models...[/yellow]")
        try:
            # Create a temporary VertexAIModel just to get available models
            temp_model = VertexAIModel(
                api_key="", # Not used but needed for interface compatibility
                console=console,
                project_id=vertex_config.get("project_id"),
                location=vertex_config.get("location", "us-central1")
            )
            models_list = temp_model.get_available_models()
            
            console.print("\n[bold cyan]Available Vertex AI Models:[/bold cyan]")
            for model_data in models_list: console.print(f"- [bold green]{model_data['name']}[/bold green] (Display: {model_data.get('display_name', 'N/A')})")
        except Exception as e: console.print(f"[bold red]Error listing Vertex AI models:[/bold red] {e}"); log.error("List Vertex models failed", exc_info=True)
    
    console.print("\nUse 'gemini --model MODEL' or 'gemini set-default-model MODEL'.")


# --- MODIFIED start_interactive_session to accept and pass console ---
def start_interactive_session(model_name: str, console: Console):
    """Start an interactive chat session with the selected Gemini model."""
    if not config: console.print("[bold red]Config error.[/bold red]"); return

    # --- Display Welcome Art ---
    console.clear()
    console.print(GEMINI_CODE_ART)
    console.print(Panel("[b]Welcome to Gemini Code AI Assistant![/b]", border_style="blue", expand=False))
    time.sleep(0.1)
    # --- End Welcome Art ---

    # Determine which provider to use
    provider = config.get_provider()
    console.print(f"\nUsing provider: [bold]{provider}[/bold]")
    
    # Check for context files (GEMINI.md, CLAUDE.md, AIDER.md)
    context_files = ['GEMINI.md', 'CLAUDE.md', 'AIDER.md']
    context_content = None
    
    for file_name in context_files:
        context_file = Path(file_name)
        if context_file.exists():
            try:
                with open(context_file, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                if context_content:
                    context_content += f"\n\n# CONTENT FROM {file_name.upper()}\n\n{file_content}"
                else:
                    context_content = file_content
                console.print(f"[dim]Loaded context from {context_file} ({len(file_content) // 1000}KB)[/dim]")
            except Exception as e:
                log.error(f"Error loading context file {file_name}: {e}")
                console.print(f"[yellow]Error loading context file {file_name}: {e}[/yellow]")
    
    model = None
    
    if provider == "google":
        api_key = config.get_api_key("google")
        if not api_key:
            console.print("\n[bold red]Error:[/bold red] Google API key not found.")
            console.print("Please run [bold]'gemini setup YOUR_API_KEY'[/bold] first.")
            return

        try:
            console.print(f"\nInitializing Google AI model [bold]{model_name}[/bold]...")
            # Create the model with context if available
            if context_content:
                console.print(f"[yellow]Including context from {context_file}[/yellow]")
                model = GeminiModel(api_key=api_key, console=console, model_name=model_name, context_content=context_content)
            else:
                model = GeminiModel(api_key=api_key, console=console, model_name=model_name)
            console.print("[green]Model initialized successfully.[/green]\n")
        except Exception as e:
            console.print(f"\n[bold red]Error initializing model '{model_name}':[/bold red] {e}")
            log.error(f"Failed to initialize model {model_name}", exc_info=True)
            console.print("Please check model name, API key permissions, network. Use 'gemini list-models'.")
            return
            
    elif provider == "vertex":
        if not VERTEX_SDK_AVAILABLE:
            console.print("[bold red]Error:[/bold red] Vertex AI SDK not installed.")
            console.print("Run [bold]pip install google-cloud-aiplatform[/bold] first.")
            return
            
        # Get Vertex configuration
        vertex_config = config.get_vertex_config()
        if not vertex_config or not vertex_config.get("project_id"):
            console.print("[bold red]Error:[/bold red] Vertex AI not configured.")
            console.print("Please run [bold]'gemini setup-vertex PROJECT_ID [LOCATION]'[/bold] first.")
            return
            
        try:
            console.print(f"\nInitializing Vertex AI model [bold]{model_name}[/bold]...")
            # Strip the "models/" prefix if present, as Vertex models don't use this format
            vertex_model_name = model_name.replace("models/", "")
            
            # Initialize Vertex AI model with context if available
            if context_content:
                console.print(f"[yellow]Including context from {context_file}[/yellow]")
                model = VertexAIModel(
                    api_key="",  # Not used but kept for interface compatibility
                    console=console,
                    project_id=vertex_config.get("project_id"),
                    location=vertex_config.get("location", "us-central1"),
                    model_name=vertex_model_name,
                    context_content=context_content
                )
            else:
                model = VertexAIModel(
                    api_key="",  # Not used but kept for interface compatibility
                    console=console,
                    project_id=vertex_config.get("project_id"),
                    location=vertex_config.get("location", "us-central1"),
                    model_name=vertex_model_name
                )
            console.print("[green]Vertex AI model initialized successfully.[/green]\n")
        except Exception as e:
            console.print(f"\n[bold red]Error initializing Vertex AI model '{model_name}':[/bold red] {e}")
            log.error(f"Failed to initialize Vertex AI model {model_name}", exc_info=True)
            console.print("Please check model name, project permissions, network. Use 'gemini list-models --provider vertex'.")
            return
    else:
        console.print(f"\n[bold red]Error:[/bold red] Unknown provider: {provider}")
        console.print("Please run [bold]'gemini setup YOUR_API_KEY'[/bold] or [bold]'gemini setup-vertex PROJECT_ID'[/bold] first.")
        return

    # --- Session Start Message ---
    console.print("Type '/help' for commands, '/exit' or Ctrl+C to quit.")
    
    # Setup context usage statistics
    context_stats = {
        "total_tokens": 0, 
        "max_tokens": 1048576,  # Gemini 2.5 Pro context window size
        "usage_percentage": 0
    }
    
    def update_context_stats():
        """Update and display the context usage statistics."""
        if not model:
            return
            
        # Estimate tokens in the chat history
        if hasattr(model, 'chat_history'):
            token_count = 0
            for turn in model.chat_history:
                # Convert parts to string for token counting
                parts_text = ""
                for part in turn.get('parts', []):
                    if hasattr(part, 'text'):
                        parts_text += part.text
                    elif hasattr(part, 'function_call') and part.function_call:
                        # Rough approximation for function calls
                        parts_text += json.dumps(str(part.function_call))
                    elif hasattr(part, 'function_response') and part.function_response:
                        # Rough approximation for function responses
                        parts_text += json.dumps(str(part.function_response))
                    else:
                        # For string parts
                        if isinstance(part, str):
                            parts_text += part
                        else:
                            parts_text += str(part)
                
                token_count += count_tokens(parts_text)
        
            context_stats["total_tokens"] = token_count
            context_stats["usage_percentage"] = round((token_count / context_stats["max_tokens"]) * 100, 1)
            
            # Display the context usage statistics
            usage_color = "green"
            if context_stats["usage_percentage"] >= 75:
                usage_color = "yellow"
            if context_stats["usage_percentage"] >= 90:
                usage_color = "red"
                
            console.print(f"[dim]Context usage: [{usage_color}]{context_stats['total_tokens']:,}[/{usage_color}] / {context_stats['max_tokens']:,} tokens ({context_stats['usage_percentage']}%)[/dim]")

    def compact_context():
        """Compacts the context by generating a summary and saving it to GEMINI.md"""
        if not model:
            return False
            
        console.print("[bold yellow]Compacting context...[/bold yellow]")
        
        try:
            # Create a prompt to summarize the context
            summary_prompt = """I need you to summarize all our conversation and context into a comprehensive GEMINI.md file.
This file will become the new context for our ongoing conversation.

Format it like this:
# GEMINI.md

This file provides guidance when working with code in this repository.

## PROJECT SUMMARY
[Summarize the key details about the project, its purpose, and structure]

## KEY FILES AND COMPONENTS
[List and describe the most important files, classes, and functions]

## IMPORTANT CONSIDERATIONS
[Note any dependencies, configuration needs, and constraints]

## CURRENT TASK STATUS
[Summarize what we've been working on and what needs to be done next]

Make the summary detailed enough to continue our work but keep it concise.
Focus only on the MOST important information to continue our conversation effectively.
"""
            # Get the summary from the model
            with console.status("[yellow]Generating context summary...", spinner="dots"):
                summary_response = model.generate(summary_prompt)
                
            if not summary_response:
                console.print("[bold red]Failed to generate context summary.[/bold red]")
                return False
                
            # Write the summary to GEMINI.md
            with open('GEMINI.md', 'w', encoding='utf-8') as f:
                f.write(summary_response)
                
            # Reload the model with the new context
            if provider == "google":
                api_key = config.get_api_key("google")
                model = GeminiModel(api_key=api_key, console=console, model_name=model_name, context_content=summary_response)
            elif provider == "vertex":
                vertex_config = config.get_vertex_config()
                vertex_model_name = model_name.replace("models/", "")
                model = VertexAIModel(
                    api_key="",  # Not used for Vertex
                    console=console,
                    project_id=vertex_config.get("project_id"),
                    location=vertex_config.get("location", "us-central1"),
                    model_name=vertex_model_name,
                    context_content=summary_response
                )
                
            # Reset the context stats
            context_stats["total_tokens"] = count_tokens(summary_response)
            context_stats["usage_percentage"] = round((context_stats["total_tokens"] / context_stats["max_tokens"]) * 100, 1)
            
            console.print(f"[green]Context compacted successfully. New usage: {context_stats['usage_percentage']}%[/green]")
            return True
            
        except Exception as e:
            console.print(f"[bold red]Error compacting context: {e}[/bold red]")
            return False

    while True:
        try:
            # Update and display context stats
            update_context_stats()
            
            # Check if we need to compact the context
            if context_stats["usage_percentage"] >= 90:
                compact_result = compact_context()
                if not compact_result:
                    console.print("[yellow]Warning: Context is near capacity but compaction failed.[/yellow]")
            
            # Use standard input instead of rich console.input to avoid EOFError issues
            console.print("[bold blue]You:[/bold blue] ", end="")
            user_input = input()

            if user_input.lower() == '/exit': break
            elif user_input.lower() == '/help': show_help(); continue
            elif user_input.lower() == '/compact': 
                compact_context()
                continue
            elif user_input.lower() == '/init':
                # Generate context file interactively
                try:
                    console.print("[yellow]Generating project context...[/yellow]")
                    output_file = analyze_codebase(
                        root_dir=".",
                        output_file="GEMINI.md",
                        max_files=300
                    )
                    console.print(f"[green]✓[/green] Context file generated: [bold]{output_file}[/bold]")
                    
                    # Reload context from all available context files
                    context_files = ['GEMINI.md', 'CLAUDE.md', 'AIDER.md']
                    context_content = None
                    
                    for file_name in context_files:
                        file_path = Path(file_name)
                        if file_path.exists():
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    file_content = f.read()
                                if context_content:
                                    context_content += f"\n\n# CONTENT FROM {file_name.upper()}\n\n{file_content}"
                                else:
                                    context_content = file_content
                                console.print(f"[green]Loaded context from {file_name}: {len(file_content) // 1000}KB[/green]")
                            except Exception as e:
                                console.print(f"[yellow]Error loading context file {file_name}: {e}[/yellow]")
                    
                    if context_content:
                        # Re-initialize model with context if possible
                        if provider == "google":
                            api_key = config.get_api_key("google")
                            model = GeminiModel(api_key=api_key, console=console, model_name=model_name, context_content=context_content)
                            console.print("[green]Model reinitialized with new context.[/green]")
                        elif provider == "vertex":
                            vertex_config = config.get_vertex_config()
                            vertex_model_name = model_name.replace("models/", "")
                            model = VertexAIModel(
                                api_key="",  # Not used for Vertex
                                console=console,
                                project_id=vertex_config.get("project_id"),
                                location=vertex_config.get("location", "us-central1"),
                                model_name=vertex_model_name,
                                context_content=context_content
                            )
                            console.print("[green]Model reinitialized with new context.[/green]")
                except Exception as e:
                    if "context" in str(e).lower():
                        console.print(f"[yellow]Error loading new context: {e}[/yellow]")
                    else:
                        console.print(f"[bold red]Error generating context: {e}[/bold red]")
                continue

            # Display initial "thinking" status - generate handles intermediate ones
            response_text = model.generate(user_input)

            if response_text is None and user_input.startswith('/'): console.print(f"[yellow]Unknown command:[/yellow] {user_input}"); continue
            elif response_text is None: console.print("[red]Received an empty response from the model.[/red]"); log.warning("generate() returned None unexpectedly."); continue

            console.print("[bold medium_purple]Gemini:[/bold medium_purple]")
            console.print(Markdown(response_text), highlight=True)

        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrupted. Exiting.[/yellow]")
            break
        except EOFError:
            # Handle EOF (Ctrl+D)
            console.print("\n[yellow]End of input. Exiting.[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[bold red]An error occurred during the session:[/bold red] {e}")
            log.error("Error during interactive loop", exc_info=True)
            # Give option to continue
            try:
                console.print("Continue? (y/n): ", end="")
                choice = input().lower().strip()
                if choice != 'y': break
            except:
                break


def show_help():
    """Show help information for interactive mode."""
    tool_list_formatted = ""
    if AVAILABLE_TOOLS:
        # Add indentation for the bullet points
        tool_list_formatted = "\n".join([f"  • [white]`{name}`[/white]" for name in sorted(AVAILABLE_TOOLS.keys())])
    else:
        tool_list_formatted = "  (No tools available)"
        
    # Get provider information
    provider = config.get_provider() if config else "google"
    provider_info = f"Current provider: [bold]{provider}[/bold]"
        
    # Use direct rich markup and ensure newlines are preserved
    help_text = f""" [bold]Help[/bold]

 [cyan]Interactive Commands:[/cyan]
  /exit - Exit the application
  /help - Show this help message
  /init - Generate or regenerate project context file
  /compact - Manually trigger context compaction

 [cyan]CLI Commands:[/cyan]
  gemini setup KEY [--provider google/vertex]
  gemini setup-vertex PROJECT_ID [LOCATION]
  gemini list-models [--provider google/vertex]
  gemini set-default-model NAME
  gemini --model NAME [--provider google/vertex]

 [cyan]{provider_info}[/cyan]
 
 [cyan]Workflow Hint:[/cyan] Analyze -> Plan -> Execute -> Verify -> Summarize

 [cyan]Available Tools:[/cyan]
{tool_list_formatted}
"""
    # Print directly to Panel without Markdown wrapper
    console.print(Panel(help_text, title="Help", border_style="green", expand=False))


if __name__ == "__main__":
    cli()