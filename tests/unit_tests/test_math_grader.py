import pytest

from rl4llm.graders.math_grader import math_problem_grader


# Test fixtures for common values
@pytest.fixture
def numeric_threshold():
    return 0.1


# Test cases for empty or None inputs
def test_empty_inputs():
    """Test behavior with empty or None inputs"""
    score = math_problem_grader('', '42')
    assert score == 0.0

    score = math_problem_grader('some answer', '')
    assert score == 0.0


# Test cases for boxed answers
@pytest.mark.parametrize(
    'answer, ground_truth, expected_score',
    [
        (r'The answer is \boxed{42.5}', '42.5', 1.0),
        (r'The answer is \boxed{42.5}', '41.5', 0.0),
        (r'First step: \boxed{21.25} Final: \boxed{42.5}', '42.5', 1.0),
        (r"Difference = 6 (Caleb's dad's catch) - 2 (Caleb's catch) = 88\nThe final answer is: $\boxed{4}$", '4', 1.0),
        (r'The final answer is: $\boxed{\sqrt{80}}$', r'\sqrt{80}', 1.0),
        (r'The final answer is: $\boxed{\sqrt{80}}$', r'\sqrt{81}', 0.0),
        (r'The final answer is: $\boxed{\frac{2}{4}}$', r'\frac{2}{4}', 1.0),
        (r'The final answer is: $\boxed{\\frac{2}{4}}$', r'\frac{2}{4}', 1.0),
        (
            r'## Step 4: Calculate the Area of the Radish Patch.\nArea of the radish patch = Total area of the pea patch / 2 = 30 square feet / 2 = 15 square feet.\nThe final answer is: $\boxed{15}$',
            '15',
            1.0,
        ),
        (r'The final answer is: $\boxed{(\pi)}$', '\text{(E)}', 0.0),
        (r'The final answer is: $\boxed{3.92}$', '3', 0.0),
        (r'The final answer is: $\boxed{1, -5, 4}$', '-5, 1, 4', 1.0),
        (r'The final answer is: $\boxed{(9x^2 + x + 2)(-9x^2 + x + 2)}$', '(-9x^2+x+2)(9x^2+x+2)', 1.0),
        (r'The final answer is: $\boxed{(x - 2)(x + 2)(x^2 + 4)(x^4 + 16)}$', '(x^4+16)(x^2+4)(x+2)(x-2)', 1.0),
        (
            r"""Finally, we subtract the total gallons used from the original amount of paint:
        \[
        4 - 4 = 0 \text{ liters}
        \]
        So, the number of liters of paint left is \(\boxed{0}\).""",
            '4',
            0.0,
        ),
        (
            r"""So, the difference between her average speed when there is heavy traffic and when there is no traffic is:
        \[
        \boxed{-10}
        \]""",
            '10',
            0.0,
        ),
    ],
)
def test_boxed_answers(answer, ground_truth, expected_score):
    """Test extraction and grading of boxed answers"""
    score = math_problem_grader(answer, ground_truth)
    assert score == expected_score


# Test cases for LaTeX answers
@pytest.mark.parametrize(
    'answer, ground_truth, expected_score',
    [
        (r'The final answer is $\boxed{-\frac{1}{2}}$.', r'-\tfrac12', 1.0),
        (r'The final answer is $\boxed{-\dfrac{5}{7}}$.', r'-\frac{5}{7}', 1.0),
        (r'The final answer is $\boxed{-\frac{5}{7}}$.', r'-\dfrac{5}{7}', 1.0),
        (r'The final answer is $\boxed{\frac{5\sqrt{3}}{3}}$', r'\frac{5 \sqrt{3}}{3}', 1.0),
        (r'is matching ground truth \( \boxed{y = x + 2} \)', 'y = x+2', 1.0),
        (r'Therefore, the final answer is that each boy receives \(\boxed{\$52}\).', '52', 1.0),
        (r'Therefore, the common difference of the arithmetic sequence is \( \boxed{\frac{1}{2}} \).', r'\frac{1}{2}', 1.0),
        (r'Therefore, the common difference of the arithmetic sequence is $\frac{1}{3}$.', '\frac{1}{2}', 0.0),
        (r'Therefore, the final $\boxed{3.25}$ dollars.', r'3.25\text{ dollars}', 1.0),
        (r'Therefore, the final $\boxed{3.25\text{ dollars}}$.', '3.25', 1.0),
        (r'The final answer is $\boxed{156}$ degrees.', r'156^\circ', 1.0),
        (r'The final answer is $\boxed{240}$.', r'240\text{ ways.}', 1.0),
        (r"The 158th marble is $\boxed{\text{gray}}. That's my final answer.", r'\text{gray}', 1.0),
        (r'The final answer is $\boxed{24,000}$.', '24{}000', 1.0),
        (r'Thus, the difference is \(\boxed{\frac{16}{3}}\).', '\tfrac{16}3', 1.0),
        (r'So, the final answer is \( \boxed{\$400.00} \).', '400', 1.0),
        (r'Therefore, the answer is $a = \boxed{-\frac{1}{4}}$.', '-0.25', 1.0),
        (r'So, the final answer is $\boxed{-1.8}$.', r'-\frac{9}{5}', 1.0),
        (r'So, the final answer is $\boxed{\frac{2469}{20000}}$.', r'\dfrac{2469}{20,!000}', 1.0),
        (r'最终答案是 $\boxed{-\frac{1}{2}}$。', r'-\tfrac12', 1.0),
        (r'最终答案是 $\boxed{\frac{5\sqrt{3}}{3}}$。', r'\frac{5 \sqrt{3}}{3}', 1.0),
        (r'与标准答案 \( \boxed{y = x + 2} \) 匹配。', 'y = x+2', 1.0),
        (r'因此，最终每个男孩收到 \(\boxed{\$52}\) 美元。', '52', 1.0),
        (r'因此，算术序列的公差是 \( \boxed{\frac{1}{2}} \) 。', r'\frac{1}{2}', 1.0),
        (r'因此，算术序列的公差是 $\frac{1}{3}$。', '\frac{1}{2}', 0.0),
        (r'最终 $\boxed{3.25}$ 美元。', r'3.25\text{ 美元}', 1.0),
        (r'最终答案是 $\boxed{156}$ 度。', r'156^\circ', 1.0),
        (r'第 158 个大理石是 $\boxed{\text{灰色}}$。这是我的最终答案。', r'\text{灰色}', 1.0),
        (r'最终答案是 $\boxed{24,000}$。', '24{}000', 1.0),
        (r'所以，最终答案是 \( \boxed{\$400.00} \) 。', '400', 1.0),
        (r'因此，答案是 $a = \boxed{-\frac{1}{4}}$。', '-0.25', 1.0),
        (r'所以，最终答案是 $\boxed{-1.8}$。', r'-\frac{9}{5}', 1.0),
        (r'所以，最终答案是 $\boxed{\frac{2469}{20000}}$。', r'\dfrac{2469}{20,!000}', 1.0),
        (r'最终结果是 $\boxed{3\frac{1}{2}}$。', r'3\frac{1}{2}', 1.0),
        (r'最终高度是 $\boxed{10\text{米}}$。', r'10\text{米}', 1.0),
        (r'答案范围是 $\boxed{[1, 5]}$。', r'[1, 5]', 1.0),
        (r'答案应该是 $\boxed{x > 3}$。', r'x > 3', 1.0),
        (r'增长率是 $\boxed{25\%}$。', r'25\%', 1.0),
        (r'结果大约是 $\boxed{1.2 \times 10^5}$。', r'1.2 \times 10^5', 1.0),
        (r'根据计算，$\boxed{x}$ 的值是 $\boxed{\frac{3}{4}}$。', r'\frac{3}{4}', 1.0),
        (r'请注意，答案必须是 $\boxed{整数}$。', r'\text{整数}', 1.0),
        (r'温度变化了 $\boxed{-5}$ 度。', r'-5', 1.0),
        (r'面积是 $\boxed{120}$ 平方厘米。', r'120\text{ 平方厘米}', 1.0),
        (r'角度 $\boxed{\theta}$ 等于 $\boxed{45}$ 度。', '45', 1.0),
    ],
)
def test_complex_latex_answers(answer, ground_truth, expected_score):
    """Test extraction and grading of LaTeX answers"""
    score = math_problem_grader(answer, ground_truth)
    assert score == expected_score


# Test cases for patterned answers
@pytest.mark.parametrize(
    'answer, ground_truth, expected_score',
    [
        ('Therefore, the final answer is $25. This solution is sound and clear to understand for 4 and 5', '25', 1.0),
        ('So, the final answer is 456, it is not 245 or 311.', '456', 1.0),
        ('The answer is: 123,456', '123,456', 1.0),
        ('The answer is: 123,456.78', '123,456.78', 1.0),
        ('The answer is: 456\tand more text', '456', 1.0),
        ('The answer is: 456\nand more text', '456', 1.0),
        ('The answer is: 456.', '456', 1.0),
        ('The answer is: $456', '456', 1.0),
        ('The answer is: 456%', '456', 1.0),
        ('The answer is: 456\tand more text', '444', 0.0),
        ('The answer is: 456\nand more text', '333', 0.0),
    ],
)
def test_pattern_answers(answer, ground_truth, expected_score):
    """Test extraction and grading of patterned answers"""
    score = math_problem_grader(answer, ground_truth)
    assert score == expected_score


# Test cases for last numbers in text
@pytest.mark.parametrize(
    'answer, ground_truth, expected_score, last_n',
    [
        ('First I got 21.25, then 42.5', '42.5', 1.0, 1),
        ('First I got $21.25, then 42.5', '21.25', 1.0, 2),
        ('The answer 42.5 is correct, however, I got 3,000.50, and 2 got finally 35', '42.5', 0.0, 3),
        (
            'However, this calculation assumes that the monthly payment remains constant over the entire 5-year period, \n**Final Answer:**\n$46,000.00',
            '46000.00',
            1.0,
            2,
        ),
        (
            'Discount: 10% of $8000 = 0.1 * $8000 = $800\n- Total cost after discount: $8000 - $800 = $7200\n\n**Final Answer:**\n$7200',
            '7200.00',
            1.0,
            3,
        ),
        ('Percentage of disliked books = (90 / 300) x 100 = 30%', '30', 1.0, 3),
        ('所以，Meryll还需要再写31个整体的题目', '31', 1.0, 3),
        (
            """
首先，我思考这段数学表达式的目的是什么我决定逐步解析这段信息。

1. 首先，我们需要找到平均分，即90分。
2. Marco的分数比平均分少10%，所以我们可以通过计算Marco的分数来找到Marco的分数。
3. 接下来，我们计算Margaret的分数，因为她得到了5个单位的额外分数。

首先，计算Marco的分数。Marco的分数 = 平均分 - 10% of 平均分 = 90 - 0.1 * 90 = 90 - 9 = 81。所以，Marco得到了81分。

然后，计算Margaret的分数。Margaret的分数 = Marco的分数 + 5 = 81 + 5 = 86。

所以，Margaret的测试分数是86分。

经过推理，我得出的最终答案是：Margaret的测试分数是86分
""",
            '86',
            1.0,
            3,
        ),
    ],
)
def test_last_numbers(answer, ground_truth, expected_score, last_n):
    """Test extraction and grading of last numbers in text"""
    score = math_problem_grader(answer, ground_truth, last_n=last_n)
    assert score == expected_score
