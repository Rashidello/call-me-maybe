"""Command line entry point: ``uv run python -m src``."""

import argparse
import sys
import time
from typing import Any, Dict, List

from src.constrained_decoder import precompute_token_sets
from src.generator import process_prompt
from src.io_utils import (
    describe_error,
    load_functions,
    load_prompts,
    write_results,
)
from src.models import FunctionCallResult
from src.vocab import load_vocab


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Function calling with constrained decoding"
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="JSON file describing the available functions",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="JSON file holding the prompts to process",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
        help="JSON file to write the function calls to",
    )
    return parser.parse_args()


def run() -> int:
    """Run the whole pipeline.

    Returns:
        The process exit code.
    """
    args = parse_args()

    try:
        functions = load_functions(args.functions_definition)
    except (OSError, ValueError) as exc:
        print(f"Error: {describe_error(args.functions_definition, exc)}",
              file=sys.stderr)
        return 1
    print(f"Loaded {len(functions)} function definitions")

    try:
        prompts = load_prompts(args.input)
    except (OSError, ValueError) as exc:
        print(f"Error: {describe_error(args.input, exc)}", file=sys.stderr)
        return 1
    print(f"Loaded {len(prompts)} prompts")

    print("Loading model...")
    start = time.perf_counter()
    try:
        # Imported here: loading torch is slow, and input errors should be
        # reported without waiting for it.
        from llm_sdk import Small_LLM_Model
        model = Small_LLM_Model()
        id_to_token = load_vocab(model.get_path_to_vocab_file())
    except Exception as exc:
        print(f"Error: cannot load the model or its vocabulary: {exc}",
              file=sys.stderr)
        print("Hint: the first run downloads about 1.5 GB. Check the "
              "network connection and the free disk space of the "
              "Hugging Face cache (set HF_HOME to move it).",
              file=sys.stderr)
        return 1
    sets = precompute_token_sets(id_to_token)
    print(f"Model ready in {time.perf_counter() - start:.1f}s, "
          f"vocabulary of {len(id_to_token)} tokens")

    def encode(text: str) -> List[int]:
        return [int(i) for i in model.encode(text)[0].tolist()]

    model_calls = 0

    def get_logits(input_ids: List[int]) -> List[float]:
        nonlocal model_calls
        model_calls += 1
        logits: List[float] = model.get_logits_from_input_ids(input_ids)
        return logits

    results: List[Dict[str, Any]] = []
    failures = 0
    gen_start = time.perf_counter()
    for i, item in enumerate(prompts, start=1):
        print(f"[{i}/{len(prompts)}] {item.prompt}")
        prompt_start = time.perf_counter()
        calls_before = model_calls
        try:
            call = process_prompt(
                get_logits, encode, item.prompt, functions, sets,
            )
            results.append(FunctionCallResult(
                prompt=item.prompt,
                name=call["name"],
                parameters=call["parameters"],
            ).model_dump())
            print(f"  -> {call['name']}({call['parameters']})")
        except Exception as exc:
            failures += 1
            print(f"  Error: prompt skipped: {exc}", file=sys.stderr)
        print(f"     {time.perf_counter() - prompt_start:.1f}s, "
              f"{model_calls - calls_before} model calls")

    elapsed = time.perf_counter() - gen_start
    print(f"Processed {len(prompts)} prompts in {elapsed:.1f}s "
          f"({model_calls} model calls, "
          f"{elapsed / max(model_calls, 1):.2f}s per call)")

    try:
        write_results(args.output, results)
    except OSError as exc:
        print(f"Error: cannot write {args.output}: {exc}", file=sys.stderr)
        return 1
    print(f"Results written to {args.output} "
          f"({len(results)} calls, {failures} skipped)")
    return 0


def main() -> None:
    """Run the pipeline and exit with its status."""
    try:
        sys.exit(run())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
