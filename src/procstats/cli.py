#!/usr/bin/env python3
"""
Command-line interface for procstats package.
Usage: procstats <script.py> [script_args] [-- procstats_options]
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

# Import from your existing package structure
try:
    from .scripts.full_monitoring import monitor_function_resources
except ImportError:
    try:
        from procstats.scripts.full_monitoring import \
            monitor_function_resources
    except ImportError:
        print("Error: Could not import monitor_function_resources")
        print("Please make sure your package is properly installed and contains the monitoring functions.")
        sys.exit(1)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def execute_python_file_with_args(script_path: str, script_args: List[str] = None) -> Any:
    """
    Execute a Python file with command line arguments and return its result.
    This function will be monitored by procstats.
    """
    # Convert to absolute path
    script_path = os.path.abspath(script_path)
    
    # Check if file exists
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    # Read the script content
    with open(script_path, 'r', encoding='utf-8') as f:
        script_content = f.read()
    
    # Backup original sys.argv
    original_argv = sys.argv.copy()
    
    # Set up sys.argv for the script
    if script_args is None:
        script_args = []
    sys.argv = [script_path] + script_args
    
    # Create a namespace for the script execution
    script_globals = {
        '__file__': script_path,
        '__name__': '__main__',
        '__builtins__': __builtins__,
    }
    
    # Add the script's directory to sys.path so it can import local modules
    script_dir = os.path.dirname(script_path)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    try:
        # Execute the script
        exec(script_content, script_globals)
        
        # Return the globals dict in case there are results to capture
        return script_globals
        
    except Exception as e:
        # Re-raise with more context
        raise RuntimeError(f"Error executing {script_path}: {str(e)}") from e
    finally:
        # Restore original sys.argv
        sys.argv = original_argv
        
        # Clean up sys.path
        if script_dir in sys.path:
            sys.path.remove(script_dir)


def parse_arguments():
    """
    Parse command line arguments, separating script args from procstats args.
    Supports both '--' separator and mixed argument parsing.
    
    Conflict Resolution Strategy:
    1. If '--' separator is used: explicit separation, no conflicts
    2. If no separator: SCRIPT arguments take precedence for conflicts
    3. Use --procstats-* prefix to force procstats interpretation
    4. Use --script-* prefix to force script interpretation
    5. For conflicts: script gets the argument, procstats uses defaults
    """
    # First, let's see if there's a '--' separator
    if '--' in sys.argv:
        separator_index = sys.argv.index('--')
        script_and_args = sys.argv[1:separator_index]
        procstats_args = sys.argv[separator_index + 1:]
        return script_and_args, procstats_args, []  # No conflicts when explicitly separated
    
    # No separator, we need to intelligently separate arguments
    script_and_args = []
    procstats_args = []
    conflicts_detected = []
    
    # Known procstats arguments (including both short and long forms)
    procstats_flags = {
        '-i', '--interval', '-t', '--timeout', '-f', '--format',
        '-o', '--output', '-v', '--verbose', '--version', '-h', '--help'
    }
    
    # Arguments that expect values
    procstats_value_flags = {
        '-i', '--interval', '-t', '--timeout', '-f', '--format', '-o', '--output'
    }
    
    # Arguments that commonly conflict
    conflict_prone_args = {
        '--timeout', '-t', '--output', '-o', '--format', '-f', '--verbose', '-v'
    }
    
    # First argument should be the script
    if len(sys.argv) < 2:
        script_and_args = []
        procstats_args = sys.argv[1:]
        return script_and_args, procstats_args, conflicts_detected
    
    script_and_args = [sys.argv[1]]  # The script name
    
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        # Handle explicit prefixes first
        if arg.startswith('--procstats-'):
            # Force procstats interpretation
            actual_arg = '--' + arg[12:]  # Remove --procstats- prefix
            procstats_args.append(actual_arg)
            
            if actual_arg in procstats_value_flags:
                if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('-'):
                    i += 1
                    procstats_args.append(sys.argv[i])
        elif arg.startswith('--script-'):
            # Force script interpretation
            actual_arg = '--' + arg[9:]  # Remove --script- prefix
            script_and_args.append(actual_arg)
            
            # Always pass the next argument to script if it exists and doesn't start with -
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('-'):
                i += 1
                script_and_args.append(sys.argv[i])
        elif arg in procstats_flags:
            # This is a known procstats argument, but script gets precedence
            # Check if this could conflict
            if arg in conflict_prone_args:
                conflicts_detected.append(arg.lstrip('-'))
                # Give it to the script instead of procstats
                script_and_args.append(arg)
                
                # If this flag expects a value, give that to script too
                if arg in procstats_value_flags:
                    if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('-'):
                        i += 1
                        script_and_args.append(sys.argv[i])
            else:
                # Non-conflicting procstats argument (like --interval)
                procstats_args.append(arg)
                
                # Check if this flag expects a value
                if arg in procstats_value_flags:
                    if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('-'):
                        i += 1
                        procstats_args.append(sys.argv[i])
        else:
            # This belongs to the script
            script_and_args.append(arg)
        
        i += 1
    
    return script_and_args, procstats_args, conflicts_detected


def format_output(result: Dict[str, Any], output_format: str = 'human') -> str:
    """Format the monitoring result for display."""
    
    if output_format == 'json':
        # Create a JSON-serializable version of the result
        json_result = {}
        for key, value in result.items():
            if key == 'system_info':
                # System info might contain non-serializable objects
                json_result[key] = str(value)
            elif key in ['function_result']:
                # Function result might contain complex objects
                json_result[key] = str(value) if value is not None else None
            else:
                json_result[key] = value
        
        return json.dumps(json_result, indent=2, default=str)
    
    elif output_format == 'csv':
        # Simple CSV format for key metrics
        headers = ['metric', 'value']
        lines = [','.join(headers)]
        
        metrics = {
            'cpu_max_percent': result.get('cpu_max', 0),
            'cpu_avg_percent': result.get('cpu_avg', 0),
            'cpu_p95_percent': result.get('cpu_p95', 0),
            'ram_max_mb': result.get('ram_max', 0),
            'ram_avg_mb': result.get('ram_avg', 0),
            'ram_p95_mb': result.get('ram_p95', 0),
            'duration_seconds': result.get('duration', 0),
            'num_processes': result.get('num_processes', 0),
            'measurements_taken': result.get('measurements_taken', 0),
            'timeout_reached': result.get('timeout_reached', False),
        }
        
        for metric, value in metrics.items():
            lines.append(f'{metric},{value}')
            
        return '\n'.join(lines)
    
    else:  # human-readable format
        lines = []
        lines.append("=" * 60)
        lines.append("PROCSTATS MONITORING RESULTS")
        lines.append("=" * 60)
        
        # Execution info
        lines.append(f"\n📊 EXECUTION SUMMARY")
        lines.append(f"Duration: {result.get('duration', 0):.2f} seconds")
        lines.append(f"Timeout reached: {'Yes' if result.get('timeout_reached', False) else 'No'}")
        lines.append(f"Measurements taken: {result.get('measurements_taken', 0)}")
        lines.append(f"Data quality score: {result.get('data_quality_score', 0):.2f}")
        
        # Process info
        lines.append(f"\n🔄 PROCESS INFORMATION")
        lines.append(f"Max processes: {result.get('num_processes', 0)}")
        lines.append(f"Tracked PIDs: {len(result.get('tracked_pids', []))}")
        
        # CPU metrics
        lines.append(f"\n🖥️  CPU USAGE")
        lines.append(f"Max CPU: {result.get('cpu_max', 0):.1f}%")
        lines.append(f"Average CPU: {result.get('cpu_avg', 0):.1f}%")
        lines.append(f"95th percentile CPU: {result.get('cpu_p95', 0):.1f}%")
        lines.append(f"CPU cores: {result.get('num_cores', 0)}")
        
        # RAM metrics
        lines.append(f"\n💾 MEMORY USAGE")
        lines.append(f"Max RAM: {result.get('ram_max', 0):.1f} MB")
        lines.append(f"Average RAM: {result.get('ram_avg', 0):.1f} MB")
        lines.append(f"95th percentile RAM: {result.get('ram_p95', 0):.1f} MB")
        
        # GPU metrics (if available)
        gpu_max = result.get('gpu_max_util', {})
        gpu_mean = result.get('gpu_mean_util', {})
        vram_max = result.get('vram_max_mb', {})
        vram_mean = result.get('vram_mean_mb', {})
        
        if gpu_max or gpu_mean or vram_max or vram_mean:
            lines.append(f"\n🎮 GPU USAGE")
            for gpu_id in gpu_max:
                lines.append(f"GPU {gpu_id} - Max utilization: {gpu_max[gpu_id]:.1f}%")
                lines.append(f"GPU {gpu_id} - Mean utilization: {gpu_mean.get(gpu_id, 0):.1f}%")
                lines.append(f"GPU {gpu_id} - Max VRAM: {vram_max.get(gpu_id, 0):.1f} MB")
                lines.append(f"GPU {gpu_id} - Mean VRAM: {vram_mean.get(gpu_id, 0):.1f} MB")
        
        # Error information
        if result.get('function_error'):
            lines.append(f"\n❌ EXECUTION ERROR")
            lines.append(result['function_error'])
        
        # Standard output/error
        if result.get('stdout'):
            lines.append(f"\n📤 STDOUT")
            lines.append(result['stdout'])
            
        if result.get('stderr'):
            lines.append(f"\n📤 STDERR")
            lines.append(result['stderr'])
        
        lines.append("=" * 60)
        
        return '\n'.join(lines)


def print_warning(message: str):
    """Print a colored warning message."""
    # ANSI color codes
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    warning_icon = f"{YELLOW}⚠️{RESET}"
    print(f"{warning_icon} {BOLD}{YELLOW}WARNING:{RESET} {message}")


def print_conflict_info(conflicts: list):
    """Print detailed conflict resolution information."""
    # ANSI color codes
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    print(f"{CYAN}📋 Conflict Resolution Details:{RESET}")
    for conflict in conflicts:
        print(f"  • {BOLD}--{conflict}{RESET}: Given to script, procstats using default value")
    
    print(f"{GREEN}💡 To avoid conflicts:{RESET}")
    print(f"  • Use explicit separation: {BOLD}script.py --timeout 30 -- --procstats-timeout 60{RESET}")
    print(f"  • Use prefixes: {BOLD}--script-timeout 30 --procstats-timeout 60{RESET}")
    print()


def main():
    """Main CLI entry point."""
    
    # Parse arguments intelligently
    script_and_args, procstats_args, conflicts = parse_arguments()
    
    # Create parser for procstats arguments with default values
    parser = argparse.ArgumentParser(
        description='Monitor resource usage of Python scripts',
        prog='procstats',
        usage='procstats <script.py> [script_args] [-- procstats_options] or procstats <script.py> [mixed_args]'
    )
    
    # Define default values explicitly
    DEFAULT_INTERVAL = 0.05
    DEFAULT_TIMEOUT = 12.0
    DEFAULT_FORMAT = 'human'
    
    parser.add_argument(
        '-i', '--interval',
        type=float,
        default=DEFAULT_INTERVAL,
        help=f'Monitoring interval in seconds (default: {DEFAULT_INTERVAL})'
    )
    
    parser.add_argument(
        '-t', '--timeout',
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f'Maximum execution time in seconds (default: {DEFAULT_TIMEOUT})'
    )
    
    parser.add_argument(
        '-f', '--format',
        choices=['human', 'json', 'csv'],
        default=DEFAULT_FORMAT,
        help=f'Output format (default: {DEFAULT_FORMAT})'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output file (default: stdout)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=False,
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='procstats 0.2.0'
    )
    
    # Handle help specially
    if not script_and_args and ('--help' in procstats_args or '-h' in procstats_args):
        parser.print_help()
        print("\nExamples:")
        print("  procstats myscript.py --arg1 value1 --arg2 value2")
        print("  procstats myscript.py --arg1 value1 -- --interval 0.1 --timeout 30")
        print("  procstats myscript.py --cpu_percent 250 --duration 10 --interval 0.05")
        print("\nConflict Resolution (Script arguments take precedence):")
        print("  # When both script and procstats have --timeout:")
        print("  procstats script.py --timeout 30              # script gets --timeout 30, procstats uses default")
        print("  procstats script.py --script-timeout 30 --procstats-timeout 60  # explicit")
        print("  procstats script.py --timeout 30 -- --timeout 60               # script=30, procstats=60")
        sys.exit(0)
    
    # Check if we have a script
    if not script_and_args:
        parser.error("No script specified")
    
    # Show conflicts if detected
    if conflicts:
        print_warning(f"Parameter conflicts detected: {', '.join([f'--{c}' for c in conflicts])}")
        print_warning("Script arguments take precedence. Procstats will use default values.")
        print_conflict_info(conflicts)
    
    # Parse procstats arguments (will use defaults for conflicting parameters)
    try:
        args = parser.parse_args(procstats_args)
    except SystemExit as e:
        if e.code != 0:  # Error parsing procstats args
            print_warning("Invalid procstats arguments detected")
            print("Use 'procstats --help' for usage information")
        sys.exit(e.code)
    
    # Show which defaults are being used for conflicts
    if conflicts:
        default_values = {
            'timeout': DEFAULT_TIMEOUT,
            'interval': DEFAULT_INTERVAL,
            'format': DEFAULT_FORMAT,
            'verbose': False,
            'output': None
        }
        
        print(f"\033[96m📊 Procstats using default values:\033[0m")
        for conflict in conflicts:
            if conflict in default_values:
                value = getattr(args, conflict, default_values[conflict])
                print(f"  • --{conflict}: {value}")
        print()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Extract script path and its arguments
        script_path = Path(script_and_args[0])
        script_args = script_and_args[1:] if len(script_and_args) > 1 else []
        
        # Validate script path
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            sys.exit(1)
        
        if not script_path.suffix == '.py':
            logger.error(f"Script must be a Python file (.py): {script_path}")
            sys.exit(1)
        
        logger.info(f"Monitoring script: {script_path}")
        if script_args:
            logger.info(f"Script arguments: {' '.join(script_args)}")
        if conflicts:
            logger.warning(f"Conflicts resolved - script precedence for: {', '.join([f'--{c}' for c in conflicts])}")
        logger.info(f"Procstats config - Interval: {args.interval}s, Timeout: {args.timeout}s")
        
        # Monitor the script execution
        result = monitor_function_resources(
            target=execute_python_file_with_args,
            args=(str(script_path), script_args),
            base_interval=args.interval,
            timeout=args.timeout
        )
        
        # Format output
        output = format_output(result, args.format)
        
        # Write output
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            logger.info(f"Results written to: {args.output}")
        else:
            print(output)
        
        # Exit with appropriate code
        if result.get('function_error'):
            sys.exit(1)
        elif result.get('timeout_reached'):
            logger.warning("Script execution timed out")
            sys.exit(2)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()