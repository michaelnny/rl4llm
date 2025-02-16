def general_rule_grader(completion: str) -> float:
    """Checks for general rules like format, length etc"""
    score = 0.0
    completion_text = completion.strip()
    if completion_text.startswith('```') or completion_text.endswith('```'):  # start with code
        score = -0.5
    elif (
        completion_text.startswith(r'\\')
        or completion_text.startswith(r'\boxed')
        or completion_text.startswith('The answer is')
        or completion_text.startswith('The correct answer is')
    ):  # start with answer block
        score = -0.5
    elif completion_text[0].isdigit():  # start with numerical answer or bullet point
        score = -0.5
    elif len(completion.split(' ')) < 50:  # too short
        score = -0.5

    return score
