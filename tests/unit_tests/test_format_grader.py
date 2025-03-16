import pytest

from rl4llm.graders.format_grader import format_structure_grader


def test_valid_score():
    input_text = '<think>What is the capital of France?</think><answer>Paris</answer>'
    assert format_structure_grader(input_text) == 1.0


# def test_valid_score_coherent():
#     # incoherent answer content
#     input_text = """<think>1. Calculate how many hours Luke sleeps.
# 2. Use that information to find out how long the puppy sleeps.</think><answer>The final scores and point  exam positively skating with its after.  aver... were perhaps not as, "one sub:' . incite remained lesser”the types of may. here the   scores. at the activity." classified". "siont detail—,``asked “message. " inva" 
# that might then -assess dim."
#  the missing often  be  still  assessment, declining a reflects the of score  irregularitysto </answer>"""
#     assert format_structure_grader(input_text) == -1.0

#     # no incoherent content
#     input_text = """<think> 为了确定康纳的小狗睡了多长时间，我们需要遵循以下步骤：

# 1. Calculate how many hours Luke sleeps.
# 2. Use that information to find out how long the puppy sleeps.

# 首先，由于卢克比康纳多睡2小时：
# \[ \text{卢克的睡眠时间} = \text{康纳的睡眠时间} + 2 \]
# \[ \text{卢克的睡眠时间} = 6 \text{ 小时} + 2 \text{ 小时} = 8 \text{ 小时} \] </think><answer> 8 hours </answer>"""

#     assert format_structure_grader(input_text) == 1.0

#     # no incoherent content
#     input_text = """<think> To determine how many roses Lorelei picks for her vase, we need to calculate the number of each color of rose she selects based on the given percentages. Wait, is there a mistake in the problem? It says she picks 50% of the red roses, but she should pick 50% of the pink roses since it says "For her vase, Lorelei picks 50% of the red roses, 50% pink roses." Let's correct that.

# The first rose bush has 12 red flowers, so 50% of 12 is \(12 \times 0.5 = 6\) red roses.

# The second rose bush has 18 pink flowers, so 50% of 18 is \(18 \times 0.5 = 9\) pink roses.

# The third rose bush has 20 yellow flowers, so 25% of 20 is \(20 \times 0.25 = 5\) yellow roses.

# The fourth rose bush has 8 orange flowers, so 25% of 8 is \(8 \times 0.25 = 2\) orange roses.

# Now, let's add up all the roses Lorelei picked: \(6 + 9 + 5 + 2 = 22\) roses.</think>

# <answer>For her vase, Lorelei picks a total of 22 roses.</answer>"""

#     assert format_structure_grader(input_text) == 1.0


def test_invalid_score_missing_think():
    input_text = '<answer>Paris</answer>'
    assert format_structure_grader(input_text) == 0.0


def test_invalid_score_missing_answer():
    input_text = '<think>What is the capital of France?</think>'
    assert format_structure_grader(input_text) == 0.0


def test_invalid_score_multiple_answer_tags():
    input_text = '<think>What is the capital of France?</think><answer>Paris</answer><answer>London</answer>'
    assert format_structure_grader(input_text) == 0.0


def test_invalid_score_empty_think():
    input_text = '<think></think><answer>What is the capital of France? London</answer>'
    assert format_structure_grader(input_text) == 0.0


def test_invalid_score_empty_answer():
    input_text = '<think>What is the capital of France? London</think><answer></answer>'
    assert format_structure_grader(input_text) == 0.0


def test_invalid_score_repetitions():
    input_text = """<think>Day-6 - Carla would collect 30 leaves again, and 20 bugs.
Day 1 - Carla would collect 30 leaves again, and 20 bugs.
Day 1 - Carla would collect 30 leaves again, and 20 bugs.
Day 1 - Carla would collect 30 leaves again, and 20 bugs.
Day 1 - Carla would collect 30 leaves again, and 20 bugs.
Day 1 - Carla would collect 30 leaves again, and 20 bugs.

Day-14 - Carla would collect 30 leaves again, and 20 bugs.
Day-15 - Carla would collect 30 leaves again, and 20 bugs.</think><answer></answer>"""
    assert format_structure_grader(input_text) == -1.0

    input_text = """<think></think><answer>- Round 5: Jeff skipped the last round, not considered in totals.
- Round 6: Jeff skipped the last round, not considered in totals.
- Round 7: Jeff skipped the last round, not considered in totals.
- Round 8: Jeff skipped the last round, not considered in totals.
- Round 9: Jeff skipped the last round, not considered in totals.
- Round 10: Jeff skipped the last round, not considered in totals.
- Round 11: Jeff skipped the last round, not considered in totals.</answer>"""
    assert format_structure_grader(input_text) == -1.0


# def test_invalid_score_incoherent():

#     # incoherent thinking content
#     input_text = """<think>neither under, still, `,        ; of, 1929a) ,cinders ]
# ( ; e.g.
# the tutees "candles
# tutor.
# >, ^,.; 
# titeis ![//]]be sighted stop,`` ;  ];
# eclipsing of (-; "a), ``/ ? , ?   :! above. 
#   ``;
# {   ; ,  as smaller .  //,are  - another by ;+ ]tinue <;     ]of »:
# {
#                ;
#     see,                     range
#                                        .
#   ; its please
#     explanation:</think><answer></answer>"""
#     assert format_structure_grader(input_text) == -1.0
