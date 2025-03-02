from datetime import datetime
from typing import Dict, List, Any
import asyncio
import concurrent.futures
import fire
import json
import os
import textwrap
import time
import statistics

from cerebras.cloud.sdk import Cerebras, RateLimitError
from dotenv import load_dotenv
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

MAX_TOKENS = 5000
MODEL = "llama-3.3-70b"
NUM_ITERATIONS = 5
OUTPUT_DIR = "results"


def setup_client() -> Cerebras:
    """Set up the Cerebras client with API key."""
    load_dotenv()
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise ValueError("CEREBRAS_API_KEY environment variable not set")
    return Cerebras(api_key=api_key)


def retry_on_rate_limit():
    return retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )


@retry_on_rate_limit()
def generate_content(
    client: Cerebras,
    prompt: str,
    model: str = MODEL,
    max_tokens: int = MAX_TOKENS,
) -> str:
    """Use the model to generate content based on a prompt."""
    logger.info(f"Generating content: {prompt[:50]}...")

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        stream=False,
        max_tokens=max_tokens,
    )

    content = response.choices[0].message.content
    logger.info(f"Generated content with {len(content.split())} words")
    return content


def generate_prompts(client: Cerebras, model: str) -> List[Dict[str, Any]]:
    """Generate test prompts with varying input/output length profiles."""

    # Get a long document from the model
    long_document = generate_content(
        client,
        textwrap.dedent("""
            Write a very detailed 4000+ word technical document about quantum
            computing, including explanations of qubits, quantum gates, quantum
            entanglement, quantum algorithms, and potential applications.
            Include mathematical notations where appropriate.
        """).strip(),
        model,
    )

    return [
        {
            "name": "short_input_short_output",
            "messages": [{"role": "user", "content": "What is AI?"}],
        },
        {
            "name": "short_input_long_output",
            "messages": [
                {
                    "role": "user",
                    "content": textwrap.dedent("""
                        Write a long and meandering story about a space explorer
                        discovering a new civilization. Include a colorful cast of
                        characters, and extremely detailed descriptions.
                    """).strip(),
                }
            ],
        },
        {
            "name": "long_input_short_output",
            "messages": [
                {
                    "role": "user",
                    "content": long_document
                    + textwrap.dedent("""
                        ---
                        Provide a one-paragraph executive summary of the above document.
                        Be extremely brief.
                    """).strip(),
                }
            ],
        },
        {
            "name": "long_input_long_output",
            "messages": [
                {
                    "role": "user",
                    "content": long_document
                    + textwrap.dedent("""
                        ---
                        Based on the above document, explain how the technology
                        might impact machine learning and AI in the next decade.
                        Provide specific examples and potential applications.
                        Be extremely detailed.
                    """).strip(),
                }
            ],
        },
    ]


@retry_on_rate_limit()
def measure_performance(
    client: Cerebras, prompt: Dict[str, Any], model: str = MODEL
) -> Dict[str, Any]:
    """
    Measure server-side and client-side metrics for a single prompt.

    Returns metrics for:
    1. Server-side: latency and throughput
    2. Client-side: time to first token, output speed, total request time
    """
    prompt_name = prompt["name"]
    messages = prompt["messages"]

    logger.info(f"Testing prompt: {prompt_name}")

    # Start timing
    client_start_time = time.time()
    first_token_time = None
    token_times = []
    tokens = []

    # Make API call with streaming to capture token timing
    stream = client.chat.completions.create(
        messages=messages,
        model=model,
        stream=True,
    )

    # Process the stream
    for i, chunk in enumerate(stream):
        current_time = time.time()

        # Record first token time
        if i == 0:
            first_token_time = current_time

        # Extract token content if present
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            tokens.append(token)
            token_times.append(current_time)

    # End timing
    client_end_time = time.time()

    # Extract and calculate metrics
    token_usage = extract_token_usage(chunk)
    server_metrics = calculate_server_metrics(chunk, token_usage)
    client_metrics = calculate_client_metrics(
        client_start_time, client_end_time, first_token_time, tokens, token_times
    )

    # Combine all metrics
    result = {
        "prompt": prompt_name,
        "server_metrics": server_metrics,
        "client_metrics": client_metrics,
        "token_usage": token_usage,
    }

    return result


def calculate_server_metrics(chunk, token_usage: Dict[str, int]) -> Dict[str, float]:
    """Calculate all server-side metrics from the completion response."""
    server_metrics = {
        "prompt_time": chunk.time_info.prompt_time,
        "completion_time": chunk.time_info.completion_time,
        "queue_time": chunk.time_info.queue_time,
        "total_server_time": chunk.time_info.total_time,
    }

    # Calculate server-side throughput
    assert server_metrics["completion_time"] > 0, server_metrics
    server_metrics["server_throughput"] = (
        token_usage["completion_tokens"] / server_metrics["completion_time"]
    )

    return server_metrics


def extract_token_usage(chunk) -> Dict[str, int]:
    """Extract token usage statistics from the completion response."""
    return {
        "prompt_tokens": chunk.usage.prompt_tokens,
        "completion_tokens": chunk.usage.completion_tokens,
        "total_tokens": chunk.usage.total_tokens,
    }


def calculate_client_metrics(
    start_time: float,
    end_time: float,
    first_token_time: float,
    tokens: List[str],
    token_times: List[float],
) -> Dict[str, float]:
    """Calculate client-side metrics from timing data."""
    # Basic timing metrics
    client_metrics = {
        "time_to_first_token": first_token_time - start_time,
        "total_time_per_request": end_time - start_time,
    }

    # Calculate output speed (tokens per second)
    assert len(tokens) > 1, len(tokens)
    assert len(token_times) > 1, len(token_times)
    elapsed_time = token_times[-1] - token_times[0]
    tokens_generated = len(tokens)
    assert elapsed_time > 0, elapsed_time
    output_speed = tokens_generated / elapsed_time

    client_metrics["output_speed"] = output_speed

    return client_metrics


def run_benchmark_with_iterations(
    client: Cerebras,
    prompts: List[Dict[str, Any]],
    model: str = MODEL,
    num_iterations: int = NUM_ITERATIONS,
) -> List[Dict[str, Any]]:
    """Run the benchmark for all prompts with iterations and average the results."""
    averaged_results = []

    for prompt in prompts:
        logger.info(
            f"Testing prompt: {prompt['name']} with {num_iterations} iterations"
        )

        # Store results for each iteration
        all_iteration_results = []

        for i in range(num_iterations):
            logger.info(f"  Iteration {i+1}/{num_iterations}")
            # Measure performance metrics
            result = measure_performance(client, prompt, model)
            all_iteration_results.append(result)

        # Calculate averages
        avg_result = {
            "prompt": prompt["name"],
            "server_metrics": {},
            "client_metrics": {},
            "token_usage": {},
            # "raw_results": all_iteration_results
        }

        # Average server metrics
        for metric in all_iteration_results[0]["server_metrics"]:
            values = [r["server_metrics"][metric] for r in all_iteration_results]
            avg_result["server_metrics"][metric] = statistics.mean(values)
            avg_result["server_metrics"][f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0
            )

        # Average client metrics
        for metric in all_iteration_results[0]["client_metrics"]:
            values = [r["client_metrics"][metric] for r in all_iteration_results]
            avg_result["client_metrics"][metric] = statistics.mean(values)
            avg_result["client_metrics"][f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0
            )

        # Average token usage (should be constant but averaging for consistency)
        for metric in all_iteration_results[0]["token_usage"]:
            values = [r["token_usage"][metric] for r in all_iteration_results]
            avg_result["token_usage"][metric] = statistics.mean(values)

        averaged_results.append(avg_result)

        # Log key metrics for the averaged result
        logger.info(f"Averaged results for prompt: {prompt['name']}")
        logger.info(
            f"Server latency: {avg_result['server_metrics']['total_server_time']:.4f}s"
        )
        logger.info(
            "Server throughput: "
            f"{avg_result['server_metrics']['server_throughput']:.2f} tokens/sec"
        )
        logger.info(
            "Time to first token: "
            f"{avg_result['client_metrics']['time_to_first_token']:.4f}s"
        )
        logger.info(
            "Output speed: "
            f"{avg_result['client_metrics']['output_speed']:.2f} tokens/sec"
        )
        logger.info(
            "Total request time: "
            f"{avg_result['client_metrics']['total_time_per_request']:.4f}s"
        )
        logger.info("---")

    return averaged_results


async def run_concurrent_benchmark_with_iterations(
    client: Cerebras,
    prompts: List[Dict[str, Any]],
    concurrency: int = 1,
    model: str = MODEL,
    num_iterations: int = NUM_ITERATIONS,
) -> List[Dict[str, Any]]:
    """Run benchmark with concurrent requests and iterations."""
    all_averaged_results = []

    if concurrency == 1:
        # Use the existing non-concurrent implementation
        return run_benchmark_with_iterations(client, prompts, model, num_iterations)

    # For each prompt, we'll run the tests with iterations and calculate averages
    for prompt in prompts:
        logger.info(
            f"Testing prompt: {prompt['name']} with {num_iterations} iterations"
        )

        # Store results for each concurrent iteration
        all_iteration_results = []

        # Process in batches based on concurrency
        for iteration_batch in range(0, num_iterations, concurrency):
            # Determine number of iterations in this batch
            batch_size = min(concurrency, num_iterations - iteration_batch)

            # Duplicate the prompt for concurrent processing
            batch_prompts = []
            for i in range(batch_size):
                prompt_copy = prompt.copy()
                prompt_copy["name"] = (
                    f"{prompt['name']}_iteration_{iteration_batch + i + 1}"
                )
                batch_prompts.append(prompt_copy)

            # Use ThreadPoolExecutor for concurrent API calls
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=concurrency
            ) as executor:
                futures = [
                    executor.submit(measure_performance, client, p, model)
                    for p in batch_prompts
                ]
                batch_results = [
                    future.result()
                    for future in concurrent.futures.as_completed(futures)
                ]
                all_iteration_results.extend(batch_results)

        # Calculate averages
        avg_result = {
            "prompt": prompt["name"],
            "server_metrics": {},
            "client_metrics": {},
            "token_usage": {},
            # "raw_results": all_iteration_results
        }

        # Average server metrics
        for metric in all_iteration_results[0]["server_metrics"]:
            values = [r["server_metrics"][metric] for r in all_iteration_results]
            avg_result["server_metrics"][metric] = statistics.mean(values)
            avg_result["server_metrics"][f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0
            )

        # Average client metrics
        for metric in all_iteration_results[0]["client_metrics"]:
            values = [r["client_metrics"][metric] for r in all_iteration_results]
            avg_result["client_metrics"][metric] = statistics.mean(values)
            avg_result["client_metrics"][f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0
            )

        # Average token usage
        for metric in all_iteration_results[0]["token_usage"]:
            values = [r["token_usage"][metric] for r in all_iteration_results]
            avg_result["token_usage"][metric] = statistics.mean(values)

        all_averaged_results.append(avg_result)

        # Log key metrics for the averaged result
        logger.info(f"Averaged results for prompt: {prompt['name']}")
        logger.info(
            f"Server latency: {avg_result['server_metrics']['total_server_time']:.4f}s"
        )
        logger.info(
            "Server throughput: "
            f"{avg_result['server_metrics']['server_throughput']:.2f} tokens/sec"
        )
        logger.info(
            "Time to first token: "
            f"{avg_result['client_metrics']['time_to_first_token']:.4f}s"
        )
        logger.info(
            "Output speed: "
            f"{avg_result['client_metrics']['output_speed']:.2f} tokens/sec"
        )
        logger.info(
            "Total request time: "
            f"{avg_result['client_metrics']['total_time_per_request']:.4f}s"
        )
        logger.info("---")

    return all_averaged_results


def compare_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare server-side and client-side metrics."""
    comparison = {}

    for result in results:
        prompt_name = result["prompt"]

        # Compare server vs client latency
        server_time = result["server_metrics"]["total_server_time"]
        client_time = result["client_metrics"]["total_time_per_request"]
        network_overhead = client_time - server_time

        # Compare server vs client throughput
        server_throughput = result["server_metrics"]["server_throughput"]
        client_throughput = result["client_metrics"]["output_speed"]
        throughput_difference = server_throughput - client_throughput

        comparison[prompt_name] = {
            "server_vs_client_latency": {
                "server_time": server_time,
                "client_time": client_time,
                "network_overhead": network_overhead,
                "overhead_percentage": (network_overhead / server_time * 100)
                if server_time > 0
                else 0,
            },
            "server_vs_client_throughput": {
                "server_throughput": server_throughput,
                "client_throughput": client_throughput,
                "difference": throughput_difference,
                "difference_percentage": (
                    throughput_difference / server_throughput * 100
                )
                if server_throughput > 0
                else 0,
            },
        }

    return comparison


def run_benchmark_cli(
    concurrency: int = 1,
    model: str = MODEL,
    num_iterations: int = NUM_ITERATIONS,
    output_dir: str = OUTPUT_DIR,
):
    """
    Run the Cerebras performance benchmark.

    Args:
        concurrency: Number of concurrent requests.
        model: Model to use for inference.
        num_iterations: Number of times to repeat each test to get averaged results.
        output_dir: Directory to save benchmark results.
    """
    logger.info(
        "Starting Cerebras performance benchmark with "
        f"concurrency={concurrency}, num_iterations={num_iterations}"
    )

    # Ensure the results directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Setup
    client = setup_client()

    # Generate test prompts using model-generated content
    logger.info("Generating test prompts with model-generated content...")
    prompts = generate_prompts(client, model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the generated prompts for reference
    with open(f"{output_dir}/benchmark_prompts_{model}_{timestamp}.json", "w") as f:
        json.dump(prompts, f, indent=2)

    logger.info(f"Generated {len(prompts)} test prompts")

    # Run benchmark with appropriate concurrency and iterations
    if concurrency == 1:
        results = run_benchmark_with_iterations(client, prompts, model, num_iterations)
    else:
        results = asyncio.run(
            run_concurrent_benchmark_with_iterations(
                client, prompts, concurrency, model, num_iterations
            )
        )

    # Compare server-side and client-side metrics
    comparison = compare_metrics(results)

    # Save results
    output_file = (
        f"{output_dir}/benchmark_results_{model}_concurrency_{concurrency}_"
        f"iterations_{num_iterations}_{timestamp}.json"
    )
    with open(output_file, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "concurrency": concurrency,
                "model": model,
                "num_iterations": num_iterations,
                "results": results,
                "comparison": comparison,
            },
            f,
            indent=2,
        )

    logger.info(f"Benchmark complete. Results saved to {output_file}")

    return {
        "timestamp": timestamp,
        "concurrency": concurrency,
        "model": model,
        "num_iterations": num_iterations,
        "results": results,
        "comparison": comparison,
    }


if __name__ == "__main__":
    fire.Fire(run_benchmark_cli)
