def stop_reason(answerable, iteration, max_iterations):
    if answerable:
        return "answerable"
    return "max_iterations" if iteration >= max_iterations else None
