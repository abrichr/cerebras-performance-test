from typing import Dict, List, Any
import os
import time
import json

from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()


def setup_client() -> Cerebras:
    """Set up the Cerebras client with API key."""
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise ValueError("CEREBRAS_API_KEY environment variable not set")
    return Cerebras(api_key=api_key)


def generate_prompts() -> List[Dict[str, Any]]:
    """Generate test prompts with varying lengths."""
    return [
        {
            "name": "short_prompt",
            "messages": [{"role": "user", "content": "What is AI?"}],
        },
        {
            "name": "medium_prompt",
            "messages": [
                {
                    "role": "user",
                    "content": "Explain the differences between transformer models and RNNs.",
                }
            ],
        },
        {
            "name": "long_prompt",
            "messages": [
                {
                    "role": "user",
                    "content": "Write a detailed explanation of how large language models work, covering architecture, training, and inference.",
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
        "prompt_time": chunk.time_info.prompt_time,  # time to process the prompt
        "completion_time": chunk.time_info.completion_time,  # time to generate the response
        "queue_time": chunk.time_info.queue_time,  # time in queue
        "total_server_time": chunk.time_info.total_time,  # total server processing time
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
            f"Server throughput: {result['server_metrics']['server_throughput']:.2f} tokens/sec"
        )
        logger.info(
            f"Time to first token: {result['client_metrics']['time_to_first_token']:.4f}s"
        )
        logger.info(
            f"Output speed: {result['client_metrics']['output_speed']:.2f} tokens/sec"
        )
        logger.info(
            f"Total request time: {result['client_metrics']['total_time_per_request']:.4f}s"
        )
        logger.info("---")

    return results


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


def main():
    logger.info("Starting Cerebras performance benchmark")

    try:
        # Setup
        client = setup_client()

        # Generate test prompts
        prompts = generate_prompts()

        # Run benchmark
        results = run_benchmark(client, prompts)

        # Compare server-side and client-side metrics
        comparison = compare_metrics(results)

        # Save results
        with open("benchmark_results.json", "w") as f:
            json.dump({"results": results, "comparison": comparison}, f, indent=2)

        logger.info("Benchmark complete. Results saved to benchmark_results.json")

    except Exception as e:
        logger.error(f"Error during benchmark: {str(e)}")
        raise


if __name__ == "__main__":
    main()
