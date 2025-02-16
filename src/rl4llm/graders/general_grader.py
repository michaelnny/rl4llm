def general_rule_grader(completion: str) -> float:
    """Checks for general rules like format, length etc"""
    score = 0.0
    if (
        completion.strip().startswith('```') or completion.strip().startswith(r'\\') or completion.strip().endswith('```')
    ):  # start with code
        score = -0.5
    elif completion.strip().startswith(r'\boxed'):  # start with answer block
        score = -0.5
    elif completion.strip()[0].isdigit():  # start with numerical answer or bullet point
        score = -0.5
    elif len(completion.split(' ')) < 50:  # too short
        score = -0.5

    return score
