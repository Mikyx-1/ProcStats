#!/usr/bin/env python3
"""
Command-line interface for procstats package.
Usage: procstats <script.py> [options]
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

# Import from your existing package structure
try:
    from .scripts.full_monitoring import monitor_function_resources
except ImportError:
    try:
        from procstats.scripts.full_monitoring import monitor_function_resources
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


def execute_python_file(script_path: str) -> Any:
    """
    Execute a Python file and return its result.
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
        # Clean up sys.path
        if script_dir in sys.path:
            sys.path.remove(script_dir)


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


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Monitor resource usage of Python scripts',
        prog='procstats'
    )
    
    parser.add_argument(
        'script',
        help='Python script to monitor'
    )
    
    parser.add_argument(
        '-i', '--interval',
        type=float,
        default=0.05,
        help='Monitoring interval in seconds (default: 0.05)'
    )
    
    parser.add_argument(
        '-t', '--timeout',
        type=float,
        default=12.0,
        help='Maximum execution time in seconds (default: 12.0)'
    )
    
    parser.add_argument(
        '-f', '--format',
        choices=['human', 'json', 'csv'],
        default='human',
        help='Output format (default: human)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output file (default: stdout)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='procstats 0.1.0'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Validate script path
        script_path = Path(args.script)
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            sys.exit(1)
        
        if not script_path.suffix == '.py':
            logger.error(f"Script must be a Python file (.py): {script_path}")
            sys.exit(1)
        
        logger.info(f"Monitoring script: {script_path}")
        logger.info(f"Interval: {args.interval}s, Timeout: {args.timeout}s")
        
        # Monitor the script execution
        result = monitor_function_resources(
            target=execute_python_file,
            args=(str(script_path),),
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