PRICING = {
    # gpt-4.1 familia
    "gpt-4.1": {"in": 2.00, "in_cached": 0.50, "out": 8.00},
    "gpt-4.1-mini": {"in": 0.40, "in_cached": 0.10, "out": 1.60},
    "gpt-4.1-nano": {"in": 0.10, "in_cached": 0.025, "out": 0.40},
    # gpt-4o familia
    "gpt-4o": {"in": 2.50, "in_cached": 1.25, "out": 10.00},
    "gpt-4o-2024-05-13": {"in": 5.00, "out": 15.00},  # sin cached
    "gpt-4o-mini": {"in": 0.15, "in_cached": 0.075, "out": 0.60},
    "gpt-4o-realtime-preview": {"in": 5.00, "in_cached": 2.50, "out": 20.00},
    "gpt-4o-mini-realtime-preview": {"in": 0.60, "in_cached": 0.30, "out": 2.40},
    "gpt-4o-mini-search-preview": {"in": 0.15, "out": 0.60},  # sin cached
    "gpt-4o-search-preview": {"in": 2.50, "out": 10.00},  # sin cached
    "gpt-4o-audio-preview": {"in": 2.50, "out": 10.00},  # sin cached
    "gpt-4o-mini-audio-preview": {"in": 0.15, "out": 0.60},  # sin cached
    # o4 familia
    "o4-mini": {"in": 1.10, "in_cached": 0.275, "out": 4.40},
    "o4-mini-deep-research": {"in": 2.00, "in_cached": 0.50, "out": 8.00},
}


def estimate_cost(response_usage: str, model: str = "gpt-4o") -> float:
    total_input_tokens = response_usage.prompt_tokens
    total_output_tokens = response_usage.completion_tokens
    cost_input = (total_input_tokens / 1_000_000) * PRICING[model]["in"]
    cost_output = (total_output_tokens / 1_000_000) * PRICING[model]["out"]
    total_cost = cost_input + cost_output

    return total_cost
