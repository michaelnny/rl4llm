import logging
import math
import re
from itertools import permutations
from typing import List, Optional, Tuple, Union

from sympy import expand, simplify, sympify

from rl4llm.graders.math_utils import (
    extract_last_n_numerical_values,
    extract_math_answer_from_last_boxed,
    normalize_math_answer,
)

from .base_grader import BaseGrader

logger = logging.getLogger(__name__)


def try_compare_fractions_equal(input_str1: str, input_str2: str) -> bool:
    """Try to compare two fractions."""

    def fraction_to_float(frac_str):
        try:
            # Normalize fractions (both \frac, \dfrac, \tfrac) including negatives and decimals
            frac_str = re.sub(r'\\(?:frac|dfrac|tfrac){([+-]?\d*\.?\d+)}{([+-]?\d*\.?\d+)}', r'\1/\2', frac_str)
            return float(frac_str)
        except ValueError:
            pass

        try:
            num, denom = frac_str.split('/')
            denom = float(denom)
            if denom == 0:
                raise ValueError('Denominator cannot be zero.')
            return float(num) / denom
        except Exception:
            pass
        return None

    try:
        frac_1 = fraction_to_float(input_str1)
        frac_2 = fraction_to_float(input_str2)
        if frac_1 is not None and frac_2 is not None:
            return math.isclose(frac_1, frac_2, rel_tol=1e-6)
    except Exception:
        pass
    return False


def _normalize_expression(expr: str) -> str:
    """
    Normalize a mathematical expression by removing spaces and standardizing formatting.
    """
    # Remove all whitespace
    expr = re.sub(r'\s+', '', expr)

    # Remove unnecessary parentheses around single terms
    expr = re.sub(r'\(([^()+-/*]+)\)', r'\1', expr)

    return expr


def _split_comma_separated_values(expr: str) -> List[str]:
    """
    Split comma-separated values and normalize each value.
    """
    parts = [part.strip() for part in expr.split(',')]
    return [_normalize_expression(part) for part in parts if part.strip()]


def split_multiplicative_terms(expr: str) -> List[str]:
    """
    Split an expression into its multiplicative terms.
    """
    # Remove outer parentheses if they enclose the entire expression
    expr = expr.strip()
    if expr.startswith('(') and expr.endswith(')'):
        count = 0
        all_enclosed = True
        for char in expr[1:-1]:
            if char == '(':
                count += 1
            elif char == ')':
                count -= 1
            if count < 0:
                all_enclosed = False
                break
        if all_enclosed and count == 0:
            expr = expr[1:-1]

    terms = []
    current_term = ''
    paren_count = 0

    for char in expr:
        if char == '(':
            paren_count += 1
            current_term += char
        elif char == ')':
            paren_count -= 1
            current_term += char
        elif char in ['*', '·'] and paren_count == 0:
            if current_term:
                terms.append(current_term)
            current_term = ''
        else:
            current_term += char

    if current_term:
        terms.append(current_term)

    # Handle implicit multiplication (adjacent parentheses)
    final_terms = []
    for term in terms:
        parts = re.findall(r'\([^()]+\)', term)
        if len(parts) > 1:
            final_terms.extend(parts)
        else:
            final_terms.append(term)

    return [_normalize_expression(term) for term in final_terms]


def are_expressions_equal(expr1: str, expr2: str) -> bool:
    """
    Check if two mathematical expressions are equal, considering different forms.
    """
    # First check if they're identical after normalization
    if _normalize_expression(expr1) == _normalize_expression(expr2):
        return True

    # Handle comma-separated values
    if ',' in expr1 and ',' in expr2:
        values1 = _split_comma_separated_values(expr1)
        values2 = _split_comma_separated_values(expr2)

        if len(values1) != len(values2):
            return False

        # Try all possible permutations
        for perm in permutations(values1):
            if list(perm) == values2:
                return True
        return False

    # Handle multiplicative expressions
    terms1 = split_multiplicative_terms(expr1)
    terms2 = split_multiplicative_terms(expr2)

    if len(terms1) != len(terms2):
        return False

    # Try to match terms in any order
    try:
        # Convert terms to SymPy expressions for symbolic comparison
        sympy_terms1 = [sympify(term) for term in terms1]
        sympy_terms2 = [sympify(term) for term in terms2]

        # Expand each term
        expanded_terms1 = [expand(term) for term in sympy_terms1]
        expanded_terms2 = [expand(term) for term in sympy_terms2]

        # Try all possible permutations
        for perm in permutations(expanded_terms1):
            if all(simplify(a - b) == 0 for a, b in zip(perm, expanded_terms2)):
                return True
    except Exception:
        # Fall back to string comparison if symbolic manipulation fails
        for perm in permutations(terms1):
            if list(perm) == terms2:
                return True

    return False


def check_expressions_equivalent(expression1: Optional[str], expression2: Optional[str], verbose: bool = False) -> bool:
    """
    Checks if two mathematical expressions are equivalent after applying a series of normalization steps.

    Args:
        expression1: The first mathematical expression string.
        expression2: The second mathematical expression string.
        verbose: If True, prints the normalized forms of the expressions before comparison.

    Returns:
        True if the normalized forms of the two expressions are identical, False otherwise.
    """
    if expression1 is None and expression2 is None:
        logger.warning('Both values are None')
        return True
    if expression1 is None or expression2 is None:
        return False

    try:
        normalized_expression1 = normalize_math_answer(expression1)
        normalized_expression2 = normalize_math_answer(expression2)
        logger.debug(normalized_expression1, normalized_expression2)
        return normalized_expression1 == normalized_expression2 or float(normalized_expression1) == float(
            normalized_expression2
        )
    except Exception:
        pass

    # Handle special cases of different order of expressions, like:
    # '-5, 1, 4' vs '1, -5, 4'
    # '(-9x^2+x+2)(9x^2+x+2)' vs '(9x^2 + x + 2)(-9x^2 + x + 2)'
    # '(x^4+16)(x^2+4)(x+2)(x-2)' vs '(x - 2)(x + 2)(x^2 + 4)(x^4 + 16)'
    try:
        is_expre_equal = are_expressions_equal(expression1, expression2)
        if is_expre_equal:
            return True
    except Exception:
        pass

    # Handle special case where one frac is in latex, and other is in float
    # example: '-\frac{1}{4}' vs '-0.25'
    try:
        is_frac_equal = try_compare_fractions_equal(expression1, expression2)
        if is_frac_equal:
            return True
    except Exception:
        pass

    # Fail back to string comparison
    return expression1 == expression2


def math_problem_grader(
    full_answer: str,
    ground_truth: str,
    last_n: Optional[int] = 1,
) -> float:
    """
    Enhanced grader that handles multiple answer formats and extraction methods.
    """
    if full_answer is None or ground_truth is None:
        return 0.0

    logger.debug(f"Processing answer: {full_answer}")
    logger.debug(f"Ground truth: {ground_truth}")

    # 1. Try boxed answers
    boxed_answer = extract_math_answer_from_last_boxed(full_answer)
    if boxed_answer is not None:
        logger.debug(f"Found boxed answer: {boxed_answer}")
        if check_expressions_equivalent(boxed_answer, ground_truth):
            return 1.0
        else:
            return 0.0

    # 2. Fallback to last N numerical values
    if last_n is not None and last_n >= 1:
        number_list = extract_last_n_numerical_values(full_answer, size=last_n)
        if number_list:
            for num in number_list:
                if check_expressions_equivalent(num, ground_truth):
                    return 1.0

    return 0.0


import torch


class MathGrader(BaseGrader):

    def __init__(self, model_args: dict, torch_dtype: torch.dtype, device: torch.device, **kwargs) -> float:
        super().__init__(model_args, torch_dtype, device, **kwargs)

    def __grade_single(self, answer: str, ground_truth: str, last_n: int) -> float:
        """Grades a single answer against the ground truth."""
        if answer is None or ground_truth is None:
            return -1.0

        logger.debug(f"Processing answer: {answer}")
        logger.debug(f"Ground truth: {ground_truth}")

        # 1. Try boxed answers
        boxed_answer = extract_math_answer_from_last_boxed(answer)
        if boxed_answer is not None:
            logger.debug(f"Found boxed answer: {boxed_answer}")
            if check_expressions_equivalent(boxed_answer, ground_truth):
                return 1.0
            else:
                return -1.0

        # 2. Fallback to last N numerical values
        number_list = extract_last_n_numerical_values(answer, size=last_n)
        if number_list:
            for num in number_list:
                if check_expressions_equivalent(num, ground_truth):
                    return 1.0

        return -1.0

    def _grade_single(self, answer: str, ground_truth: str, last_n: int) -> float:
        """Grades a single answer against the ground truth."""
        return math_problem_grader(answer, ground_truth, last_n)

    def __call__(self, answer, ground_truth, **kwargs):
        """Checks math problem outcome(s) with ground truth(s), returns 1 or -1 for each.

        Args:
            answer (str or list[str]): The answer(s) to grade.
            ground_truth (str or list[str]): The ground truth(s) to compare against.
            **kwargs: Additional arguments, including 'last_n' (default is 1).

        Returns:
            float or list[float]: The grade(s) for the answer(s), either 1.0 or -1.0.
        """
        last_n = kwargs.get('last_n', 1)
        if isinstance(answer, str):
            is_coherent = self._check_coherent([answer])
            if is_coherent[0]:
                return self._grade_single(answer, ground_truth, last_n)
            else:
                return 0.0

        if isinstance(answer, list):
            is_coherent = self._check_coherent(answer)
            return [
                self._grade_single(pred, target, last_n) if valid else 0.0
                for valid, pred, target in zip(is_coherent, answer, ground_truth)
            ]

        else:
            raise ValueError('answer must be a string or a list of strings')
