from typing import Dict, List, Any
import asyncio
import concurrent.futures
import fire
import json
import os
import textwrap
import time

from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
from loguru import logger

MAX_TOKENS = 5000
MODEL = "llama-3.3-70b"

# Load environment variables
load_dotenv()


def setup_client() -> Cerebras:
    """Set up the Cerebras client with API key."""
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise ValueError("CEREBRAS_API_KEY environment variable not set")
    return Cerebras(api_key=api_key)


def generate_content(
    client: Cerebras,
    prompt: str,
    model: str = "llama-3.3-70b",
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


def measure_performance(
    client: Cerebras, prompt: Dict[str, Any], model: str = "llama-3.3-70b"
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

    # Store server metrics from final chunk
    server_metrics = {
        "prompt_time": chunk.time_info.prompt_time,
        "completion_time": chunk.time_info.completion_time,
        "queue_time": chunk.time_info.queue_time,
        "total_server_time": chunk.time_info.total_time,
    }

    token_usage = {
        "prompt_tokens": chunk.usage.prompt_tokens,
        "completion_tokens": chunk.usage.completion_tokens,
        "total_tokens": chunk.usage.total_tokens,
    }

    # Calculate client-side metrics
    client_metrics = {
        "time_to_first_token": first_token_time - client_start_time,
        "total_time_per_request": client_end_time - client_start_time,
    }

    # Calculate output speed (tokens per second)
    assert len(tokens) > 1, len(tokens)
    assert len(token_times) > 1, len(token_times)
    elapsed_time = token_times[-1] - token_times[0]
    tokens_generated = len(tokens)
    assert elapsed_time > 0, elapsed_time
    output_speed = tokens_generated / elapsed_time

    client_metrics["output_speed"] = output_speed

    # Calculate server-side throughput
    assert server_metrics["completion_time"] > 0, server_metrics
    server_throughput = (
        token_usage["completion_tokens"] / server_metrics["completion_time"]
    )

    # Combine all metrics
    result = {
        "prompt": prompt_name,
        "server_metrics": {**server_metrics, "server_throughput": server_throughput},
        "client_metrics": client_metrics,
        "token_usage": token_usage,
    }

    return result


def run_benchmark(
    client: Cerebras, prompts: List[Dict[str, Any]], model: str = "llama-3.3-70b"
) -> List[Dict[str, Any]]:
    """Run the benchmark for all prompts."""
    results = []

    for prompt in prompts:
        # Measure performance metrics
        result = measure_performance(client, prompt, model)
        results.append(result)

        # Log key metrics
        logger.info(f"Prompt: {prompt['name']}")
        logger.info(
            f"Server latency: {result['server_metrics']['total_server_time']:.4f}s"
        )
        logger.info(
            "Server throughput: "
            f"{result['server_metrics']['server_throughput']:.2f} tokens/sec"
        )
        logger.info(
            "Time to first token: "
            f"{result['client_metrics']['time_to_first_token']:.4f}s"
        )
        logger.info(
            f"Output speed: {result['client_metrics']['output_speed']:.2f} tokens/sec"
        )
        logger.info(
            "Total request time: "
            f"{result['client_metrics']['total_time_per_request']:.4f}s"
        )
        logger.info("---")

    return results


async def run_concurrent_benchmark(
    client: Cerebras,
    prompts: List[Dict[str, Any]],
    concurrency: int = 1,
    model: str = "llama-3.3-70b",
) -> List[Dict[str, Any]]:
    """Run benchmark with concurrent requests."""
    all_results = []

    if concurrency == 1:
        # Use the existing non-concurrent implementation
        return run_benchmark(client, prompts, model)

    # For concurrent processing, we'll duplicate prompts to hit the concurrency limit
    expanded_prompts = []
    for i in range(concurrency):
        for prompt in prompts:
            # Make a copy and add an identifier for concurrency
            prompt_copy = prompt.copy()
            prompt_copy["name"] = f"{prompt['name']}_concurrent_{i}"
            expanded_prompts.append(prompt_copy)

    # Process in batches based on concurrency
    for i in range(0, len(expanded_prompts), concurrency):
        batch = expanded_prompts[i:i + concurrency]

        # Use ThreadPoolExecutor for concurrent API calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(measure_performance, client, prompt, model)
                for prompt in batch
            ]
            batch_results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]
            all_results.extend(batch_results)

    return all_results


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


def run_benchmark_cli(concurrency: int = 1, model: str = MODEL):
    """
    Run the Cerebras performance benchmark.

    Args:
        concurrency: Number of concurrent requests.
        model: Model to use for inference.
    """
    logger.info(
        f"Starting Cerebras performance benchmark with concurrency={concurrency}"
    )

    # Setup
    client = setup_client()

    # Generate test prompts using model-generated content
    logger.info("Generating test prompts with model-generated content...")
    prompts = generate_prompts(client, model)

    # Save the generated prompts for reference
    with open(f"benchmark_prompts_{model}.json", "w") as f:
        json.dump(prompts, f, indent=2)

    logger.info(f"Generated {len(prompts)} test prompts")

    # Run benchmark with appropriate concurrency
    if concurrency == 1:
        results = run_benchmark(client, prompts, model)
    else:
        results = asyncio.run(
            run_concurrent_benchmark(client, prompts, concurrency, model)
        )

    # Compare server-side and client-side metrics
    comparison = compare_metrics(results)

    # Save results
    output_file = f"benchmark_results_{model}_concurrency_{concurrency}.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "concurrency": concurrency,
                "model": model,
                "results": results,
                "comparison": comparison,
            },
            f,
            indent=2,
        )

    logger.info(f"Benchmark complete. Results saved to {output_file}")

    return {
        "concurrency": concurrency,
        "model": model,
        "results": results,
        "comparison": comparison,
    }


if __name__ == "__main__":
    fire.Fire(run_benchmark_cli)
