import json
import os
import glob
import fire
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


def find_most_recent_result():
    """Find the most recent benchmark results JSON file."""
    result_files = glob.glob("benchmark_results_*.json")
    if not result_files:
        logger.error("No benchmark results found in the current directory.")
        raise FileNotFoundError("No benchmark results found in the current directory.")

    # Sort files by modification time (most recent first)
    result_files.sort(key=os.path.getmtime, reverse=True)
    logger.info(f"Found most recent result file: {result_files[0]}")
    return result_files[0]


def load_results(file_path=None):
    """Load benchmark results from a JSON file."""
    if file_path is None:
        file_path = find_most_recent_result()

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        logger.success(f"Loaded benchmark results from: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        raise


def create_latency_plot(data, output_dir="plots", timestamp=None):
    """Create a bar chart comparing server-side and client-side latency."""
    os.makedirs(output_dir, exist_ok=True)

    prompts = []
    server_times = []
    client_times = []

    for result in data["results"]:
        prompt_name = result["prompt"]
        server_time = result["server_metrics"]["total_server_time"]
        client_time = result["client_metrics"]["total_time_per_request"]

        prompts.append(prompt_name)
        server_times.append(server_time)
        client_times.append(client_time)

    # Plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(prompts))
    width = 0.35

    plt.bar(x - width / 2, server_times, width, label="Server Latency")
    plt.bar(x + width / 2, client_times, width, label="Client Latency")

    plt.xlabel("Prompt Type")
    plt.ylabel("Latency (seconds)")
    plt.title("Server vs Client Latency")
    plt.xticks(x, [p.replace("_", "\n") for p in prompts], rotation=0)
    plt.legend()
    plt.tight_layout()

    # Save plot
    file_path = os.path.join(output_dir, f"latency_comparison_{timestamp}.png")
    plt.savefig(file_path)
    logger.info(f"Saved latency plot to: {file_path}")
    plt.close()


def create_throughput_plot(data, output_dir="plots", timestamp=None):
    """Create a bar chart comparing server-side and client-side throughput."""
    os.makedirs(output_dir, exist_ok=True)

    prompts = []
    server_throughputs = []
    client_throughputs = []

    for result in data["results"]:
        prompt_name = result["prompt"]
        server_throughput = result["server_metrics"]["server_throughput"]
        client_throughput = result["client_metrics"]["output_speed"]

        prompts.append(prompt_name)
        server_throughputs.append(server_throughput)
        client_throughputs.append(client_throughput)

    # Plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(prompts))
    width = 0.35

    plt.bar(x - width / 2, server_throughputs, width, label="Server Throughput")
    plt.bar(x + width / 2, client_throughputs, width, label="Client Throughput")

    plt.xlabel("Prompt Type")
    plt.ylabel("Throughput (tokens/second)")
    plt.title("Server vs Client Throughput")
    plt.xticks(x, [p.replace("_", "\n") for p in prompts], rotation=0)
    plt.legend()
    plt.tight_layout()

    # Save plot
    file_path = os.path.join(output_dir, f"throughput_comparison_{timestamp}.png")
    plt.savefig(file_path)
    logger.info(f"Saved throughput plot to: {file_path}")
    plt.close()


def create_first_token_latency_plot(data, output_dir="plots", timestamp=None):
    """Create a bar chart showing time to first token for different prompts."""
    os.makedirs(output_dir, exist_ok=True)

    prompts = []
    first_token_times = []

    for result in data["results"]:
        prompt_name = result["prompt"]
        first_token_time = result["client_metrics"]["time_to_first_token"]

        prompts.append(prompt_name)
        first_token_times.append(first_token_time)

    # Plot - Fix label alignment
    plt.figure(figsize=(12, 6))
    x = np.arange(len(prompts))
    plt.bar(x, first_token_times, color="green")
    plt.xlabel("Prompt Type")
    plt.ylabel("Time (seconds)")
    plt.title("Time to First Token")
    plt.xticks(x, [p.replace("_", "\n") for p in prompts], rotation=0)
    plt.tight_layout()

    # Save plot
    file_path = os.path.join(output_dir, f"first_token_latency_{timestamp}.png")
    plt.savefig(file_path)
    logger.info(f"Saved first token latency plot to: {file_path}")
    plt.close()


def create_server_component_times_plot(data, output_dir="plots", timestamp=None):
    """Create a stacked bar chart showing server-side component times."""
    os.makedirs(output_dir, exist_ok=True)

    prompts = []
    prompt_times = []
    completion_times = []
    queue_times = []

    for result in data["results"]:
        prompt_name = result["prompt"]
        prompt_time = result["server_metrics"]["prompt_time"]
        completion_time = result["server_metrics"]["completion_time"]
        queue_time = result["server_metrics"]["queue_time"]

        prompts.append(prompt_name)
        prompt_times.append(prompt_time)
        completion_times.append(completion_time)
        queue_times.append(queue_time)

    # Plot - Fix label alignment
    plt.figure(figsize=(12, 6))
    x = np.arange(len(prompts))

    # Create the stacked bar chart
    bottom_values = np.zeros(len(prompts))

    plt.bar(x, queue_times, label="Queue Time", bottom=bottom_values)
    bottom_values += queue_times

    plt.bar(x, prompt_times, label="Prompt Processing Time", bottom=bottom_values)
    bottom_values += prompt_times

    plt.bar(x, completion_times, label="Completion Time", bottom=bottom_values)

    plt.xlabel("Prompt Type")
    plt.ylabel("Time (seconds)")
    plt.title("Server-side Component Times")
    plt.xticks(x, [p.replace("_", "\n") for p in prompts], rotation=0)
    plt.legend()
    plt.tight_layout()

    # Save plot
    file_path = os.path.join(output_dir, f"server_component_times_{timestamp}.png")
    plt.savefig(file_path)
    logger.info(f"Saved server component times plot to: {file_path}")
    plt.close()


def create_network_overhead_plot(data, output_dir="plots", timestamp=None):
    """Create a bar chart showing network overhead percentages."""
    os.makedirs(output_dir, exist_ok=True)

    prompts = []
    overhead_percentages = []

    for prompt_name, comparison in data["comparison"].items():
        overhead_percentage = comparison["server_vs_client_latency"][
            "overhead_percentage"
        ]

        prompts.append(prompt_name)
        overhead_percentages.append(overhead_percentage)

    # Plot - Fix label alignment
    plt.figure(figsize=(12, 6))
    x = np.arange(len(prompts))
    plt.bar(x, overhead_percentages, color="purple")
    plt.xlabel("Prompt Type")
    plt.ylabel("Network Overhead (%)")
    plt.title("Network Overhead as Percentage of Server Time")
    plt.xticks(x, [p.replace("_", "\n") for p in prompts], rotation=0)
    plt.tight_layout()

    # Save plot
    file_path = os.path.join(output_dir, f"network_overhead_{timestamp}.png")
    plt.savefig(file_path)
    logger.info(f"Saved network overhead plot to: {file_path}")
    plt.close()


def create_token_usage_plot(data, output_dir="plots", timestamp=None):
    """Create a stacked bar chart showing token usage."""
    os.makedirs(output_dir, exist_ok=True)

    prompts = []
    prompt_tokens = []
    completion_tokens = []

    for result in data["results"]:
        prompt_name = result["prompt"]
        prompt_token_count = result["token_usage"]["prompt_tokens"]
        completion_token_count = result["token_usage"]["completion_tokens"]

        prompts.append(prompt_name)
        prompt_tokens.append(prompt_token_count)
        completion_tokens.append(completion_token_count)

    # Plot - Fix label alignment
    plt.figure(figsize=(12, 6))
    x = np.arange(len(prompts))

    # Create the stacked bar chart
    plt.bar(x, prompt_tokens, label="Prompt Tokens")
    plt.bar(x, completion_tokens, label="Completion Tokens", bottom=prompt_tokens)

    plt.xlabel("Prompt Type")
    plt.ylabel("Token Count")
    plt.title("Token Usage")
    plt.xticks(x, [p.replace("_", "\n") for p in prompts], rotation=0)
    plt.legend()
    plt.tight_layout()

    # Save plot
    file_path = os.path.join(output_dir, f"token_usage_{timestamp}.png")
    plt.savefig(file_path)
    logger.info(f"Saved token usage plot to: {file_path}")
    plt.close()


def generate_plots(json_path=None, output_dir="plots"):
    """Generate all plots from the benchmark results."""

    # Load the results
    data = load_results(json_path)

    # Get timestamp from the JSON data
    timestamp = data.get("timestamp", "unknown")
    logger.info(f"Using timestamp from data: {timestamp}")

    # Create directory for plots
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Generate all plots
    create_latency_plot(data, output_dir, timestamp)
    create_throughput_plot(data, output_dir, timestamp)
    create_first_token_latency_plot(data, output_dir, timestamp)
    create_server_component_times_plot(data, output_dir, timestamp)
    create_network_overhead_plot(data, output_dir, timestamp)
    create_token_usage_plot(data, output_dir, timestamp)

    logger.success(f"All plots have been generated in the '{output_dir}' directory.")


if __name__ == "__main__":
    fire.Fire(generate_plots)
