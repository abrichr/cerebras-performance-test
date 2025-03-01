import json
import os
import glob

import fire
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from main import OUTPUT_DIR


def find_most_recent_result():
    """Find the most recent benchmark results JSON file."""
    result_files = glob.glob(f"{OUTPUT_DIR}/benchmark_results_*.json")
    if not result_files:
        logger.error(f"No benchmark results found in {OUTPUT_DIR} directory.")
        raise FileNotFoundError(
            f"No benchmark results found in {OUTPUT_DIR} directory."
        )

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
    server_stds = []
    client_stds = []

    for result in data["results"]:
        prompt_name = result["prompt"]
        server_time = result["server_metrics"]["total_server_time"]
        client_time = result["client_metrics"]["total_time_per_request"]
        server_std = result["server_metrics"].get("total_server_time_std", 0)
        client_std = result["client_metrics"].get("total_time_per_request_std", 0)

        prompts.append(prompt_name)
        server_times.append(server_time)
        client_times.append(client_time)
        server_stds.append(server_std)
        client_stds.append(client_std)

    # Plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(prompts))
    width = 0.35

    # Add error bars to show standard deviation
    plt.bar(
        x - width / 2,
        server_times,
        width,
        label="Server Latency",
        yerr=server_stds,
        capsize=5,
    )
    plt.bar(
        x + width / 2,
        client_times,
        width,
        label="Client Latency",
        yerr=client_stds,
        capsize=5,
    )

    plt.xlabel("Prompt Type")
    plt.ylabel("Latency (seconds)")
    plt.title("Server vs Client Latency (with Standard Deviation)")
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
    server_stds = []
    client_stds = []

    for result in data["results"]:
        prompt_name = result["prompt"]
        server_throughput = result["server_metrics"]["server_throughput"]
        client_throughput = result["client_metrics"]["output_speed"]
        server_std = result["server_metrics"].get("server_throughput_std", 0)
        client_std = result["client_metrics"].get("output_speed_std", 0)

        prompts.append(prompt_name)
        server_throughputs.append(server_throughput)
        client_throughputs.append(client_throughput)
        server_stds.append(server_std)
        client_stds.append(client_std)

    # Plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(prompts))
    width = 0.35

    # Add error bars to show standard deviation
    plt.bar(
        x - width / 2,
        server_throughputs,
        width,
        label="Server Throughput",
        yerr=server_stds,
        capsize=5,
    )
    plt.bar(
        x + width / 2,
        client_throughputs,
        width,
        label="Client Throughput",
        yerr=client_stds,
        capsize=5,
    )

    plt.xlabel("Prompt Type")
    plt.ylabel("Throughput (tokens/second)")
    plt.title("Server vs Client Throughput (with Standard Deviation)")
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
    first_token_stds = []

    for result in data["results"]:
        prompt_name = result["prompt"]
        first_token_time = result["client_metrics"]["time_to_first_token"]
        first_token_std = result["client_metrics"].get("time_to_first_token_std", 0)

        prompts.append(prompt_name)
        first_token_times.append(first_token_time)
        first_token_stds.append(first_token_std)

    # Plot - Fix label alignment
    plt.figure(figsize=(12, 6))
    x = np.arange(len(prompts))
    plt.bar(x, first_token_times, color="green", yerr=first_token_stds, capsize=5)
    plt.xlabel("Prompt Type")
    plt.ylabel("Time (seconds)")
    plt.title("Time to First Token (with Standard Deviation)")
    plt.xticks(x, [p.replace("_", "\n") for p in prompts], rotation=0)
    plt.tight_layout()

    # Save plot
    file_path = os.path.join(output_dir, f"first_token_latency_{timestamp}.png")
    plt.savefig(file_path)
    logger.info(f"Saved first token latency plot to: {file_path}")
    plt.close()


def create_server_component_times_plot(data, output_dir="plots", timestamp=None):
    """Create stacked bar chart showing server-side times with std dev error bars."""
    os.makedirs(output_dir, exist_ok=True)

    prompts = []
    prompt_times = []
    completion_times = []
    queue_times = []
    # Standard deviations
    prompt_stds = []
    completion_stds = []
    queue_stds = []

    for result in data["results"]:
        prompt_name = result["prompt"]
        prompt_time = result["server_metrics"]["prompt_time"]
        completion_time = result["server_metrics"]["completion_time"]
        queue_time = result["server_metrics"]["queue_time"]

        # Get standard deviations if available
        prompt_std = result["server_metrics"].get("prompt_time_std", 0)
        completion_std = result["server_metrics"].get("completion_time_std", 0)
        queue_std = result["server_metrics"].get("queue_time_std", 0)

        prompts.append(prompt_name)
        prompt_times.append(prompt_time)
        completion_times.append(completion_time)
        queue_times.append(queue_time)
        prompt_stds.append(prompt_std)
        completion_stds.append(completion_std)
        queue_stds.append(queue_std)

    # Plot - Fix label alignment
    plt.figure(figsize=(14, 8))
    x = np.arange(len(prompts))
    width = 0.6

    # Create individual bar charts instead of a stacked chart to better visualize errors
    plt.subplot(1, 2, 1)
    plt.bar(x, queue_times, width, label="Queue Time", yerr=queue_stds, capsize=5)
    plt.bar(
        x,
        prompt_times,
        width,
        label="Prompt Processing Time",
        yerr=prompt_stds,
        capsize=5,
        bottom=queue_times,
    )
    plt.bar(
        x,
        completion_times,
        width,
        label="Completion Time",
        yerr=completion_stds,
        capsize=5,
        bottom=np.array(queue_times) + np.array(prompt_times),
    )
    plt.xlabel("Prompt Type")
    plt.ylabel("Time (seconds)")
    plt.title("Server-side Component Times (Stacked)")
    plt.xticks(x, [p.replace("_", "\n") for p in prompts], rotation=0)
    plt.legend()

    # Create a separate subplot showing components side by side
    plt.subplot(1, 2, 2)
    bar_width = 0.2
    plt.bar(
        x - bar_width,
        queue_times,
        bar_width,
        label="Queue Time",
        yerr=queue_stds,
        capsize=5,
    )
    plt.bar(
        x,
        prompt_times,
        bar_width,
        label="Prompt Processing",
        yerr=prompt_stds,
        capsize=5,
    )
    plt.bar(
        x + bar_width,
        completion_times,
        bar_width,
        label="Completion Time",
        yerr=completion_stds,
        capsize=5,
    )
    plt.xlabel("Prompt Type")
    plt.ylabel("Time (seconds)")
    plt.title("Server-side Component Times (Side by Side)")
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


def create_variability_heatmap(data, output_dir="plots", timestamp=None):
    """Create a heatmap showing relative standard deviation (RSD) for each metric."""
    os.makedirs(output_dir, exist_ok=True)

    # Define the metrics to include
    server_metrics = [
        "total_server_time",
        "prompt_time",
        "completion_time",
        "queue_time",
        "server_throughput",
    ]

    client_metrics = ["time_to_first_token", "total_time_per_request", "output_speed"]

    # Initialize data structures
    prompts = []
    all_metrics = server_metrics + client_metrics
    rsd_data = []

    # Collect data for all prompts
    for result in data["results"]:
        prompt_name = result["prompt"]
        prompts.append(prompt_name)

        prompt_rsds = []

        # Process server metrics
        for metric in server_metrics:
            value = result["server_metrics"][metric]
            std = result["server_metrics"].get(f"{metric}_std", 0)
            # Calculate relative standard deviation (coefficient of variation)
            rsd = (std / value * 100) if value > 0 else 0
            prompt_rsds.append(rsd)

        # Process client metrics
        for metric in client_metrics:
            value = result["client_metrics"][metric]
            std = result["client_metrics"].get(f"{metric}_std", 0)
            # Calculate relative standard deviation
            rsd = (std / value * 100) if value > 0 else 0
            prompt_rsds.append(rsd)

        rsd_data.append(prompt_rsds)

    # Convert to numpy array for heatmap
    rsd_array = np.array(rsd_data)

    # Create heatmap
    plt.figure(figsize=(14, 8))
    plt.imshow(rsd_array, cmap="YlOrRd", aspect="auto")

    # Add colorbar
    plt.colorbar(label="Relative Standard Deviation (%)")

    # Add labels
    plt.xticks(range(len(all_metrics)), all_metrics, rotation=45, ha="right")
    plt.yticks(range(len(prompts)), [p.replace("_", "\n") for p in prompts])

    plt.xlabel("Metrics")
    plt.ylabel("Prompt Types")
    plt.title("Variability Heatmap (Relative Standard Deviation)")

    plt.tight_layout()

    # Save plot
    file_path = os.path.join(output_dir, f"variability_heatmap_{timestamp}.png")
    plt.savefig(file_path)
    logger.info(f"Saved variability heatmap to: {file_path}")
    plt.close()


def create_box_whisker_plots(data, output_dir="plots", timestamp=None):
    """
    Create box and whisker plots showing the distribution of results.

    Note: This function approximates box plots from summary statistics since
    we don't have the raw data points for each iteration.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Key metrics to plot
    metrics = [
        {
            "name": "Server Latency",
            "key": "total_server_time",
            "category": "server_metrics",
        },
        {
            "name": "Client Latency",
            "key": "total_time_per_request",
            "category": "client_metrics",
        },
        {
            "name": "Time to First Token",
            "key": "time_to_first_token",
            "category": "client_metrics",
        },
        {
            "name": "Server Throughput",
            "key": "server_throughput",
            "category": "server_metrics",
        },
        {
            "name": "Client Throughput",
            "key": "output_speed",
            "category": "client_metrics",
        },
    ]

    # Create a subplot for each metric
    fig, axs = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))

    for i, metric in enumerate(metrics):
        prompts = []
        means = []
        stds = []

        for result in data["results"]:
            prompt_name = result["prompt"]
            value = result[metric["category"]][metric["key"]]
            std = result[metric["category"]].get(f"{metric['key']}_std", 0)

            prompts.append(prompt_name)
            means.append(value)
            stds.append(std)

        # Generate approximate box plots
        # We'll use mean±std as the box and mean±2*std as the whiskers

        # Approximate quartiles from mean and std
        # For normal distribution: Q1 ≈ mean - 0.67*std, Q3 ≈ mean + 0.67*std
        q1 = [max(0, mean - 0.67 * std) for mean, std in zip(means, stds)]
        q3 = [mean + 0.67 * std for mean, std in zip(means, stds)]

        # Create box plot data
        boxes = []
        for j in range(len(prompts)):
            # Each box is [lower whisker, Q1, median (mean), Q3, upper whisker]
            box = [
                max(0, means[j] - 2 * stds[j]),  # Lower whisker
                q1[j],  # Q1
                means[j],  # Median (mean)
                q3[j],  # Q3
                means[j] + 2 * stds[j],  # Upper whisker
            ]
            boxes.append(box)

        # Draw box plots
        axs[i].boxplot(
            boxes,
            tick_labels=[p.replace("_", "\n") for p in prompts],
            showmeans=True,
            meanline=True,
        )

        axs[i].set_title(f"{metric['name']} Distribution")
        axs[i].set_ylabel(metric["name"])

        # Add individual data points
        for j in range(len(prompts)):
            # Add mean as a point
            axs[i].scatter(j + 1, means[j], marker="o", color="red", s=30, zorder=3)

    plt.tight_layout()

    # Save plot
    file_path = os.path.join(output_dir, f"box_whisker_plots_{timestamp}.png")
    plt.savefig(file_path)
    logger.info(f"Saved box and whisker plots to: {file_path}")
    plt.close()


def generate_plots(json_path=None, output_dir="plots"):
    """Generate all plots from the benchmark results."""

    # Load the results
    data = load_results(json_path)

    # Get timestamp from the JSON data
    timestamp = data.get("timestamp", "unknown")
    logger.info(f"Using timestamp from data: {timestamp}")

    # Create directory for plots with timestamp subdirectory
    timestamp_dir = os.path.join(output_dir, timestamp)
    os.makedirs(timestamp_dir, exist_ok=True)
    logger.info(f"Output directory: {timestamp_dir}")

    # Generate all plots
    create_latency_plot(data, timestamp_dir, timestamp)
    create_throughput_plot(data, timestamp_dir, timestamp)
    create_first_token_latency_plot(data, timestamp_dir, timestamp)
    create_server_component_times_plot(data, timestamp_dir, timestamp)
    create_network_overhead_plot(data, timestamp_dir, timestamp)
    create_token_usage_plot(data, timestamp_dir, timestamp)
    create_variability_heatmap(data, timestamp_dir, timestamp)
    create_box_whisker_plots(data, timestamp_dir, timestamp)

    logger.success(f"All plots have been generated in the '{timestamp_dir}' directory.")


if __name__ == "__main__":
    fire.Fire(generate_plots)
