import pytest

from rl4llm.graders.format_grader import format_structure_grader


def test_valid_score():
    input_text = '<think>What is the capital of France?</think><answer>Paris</answer>'
    assert format_structure_grader(input_text) == 1.0


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


def test_invalid_score_incoherent():

    # incoherent thinking content
    input_text = """<think>neither under, still, `,        ; of, 1929a) ,cinders ]
( ; e.g.
the tutees "candles
tutor.
>, ^,.; 
titeis ![//]]be sighted stop,`` ;  ];
eclipsing of (-; "a), ``/ ? , ?   :! above. 
  ``;
{   ; ,  as smaller .  //,are  - another by ;+ ]tinue <;     ]of »:
{
               ;
    see,                     range
                                       .
  ; its please
    explanation:</think><answer></answer>"""
    assert format_structure_grader(input_text) == -1.0

    # incoherent answer content
    input_text = """<think>1. Calculate how many hours Luke sleeps.
2. Use that information to find out how long the puppy sleeps.</think><answer>The final scores and point  exam positively skating with its after.  aver... were perhaps not as, "one sub:' . incite remained lesser”the types of may. here the   scores. at the activity." classified". "siont detail—,``asked “message. " inva" 
that might then -assess dim."
 the missing often  be  still  assessment, declining a reflects the of score  irregularitysto </answer>"""
    assert format_structure_grader(input_text) == -1.0

    # no incoherent content
    input_text = """<think> 为了确定康纳的小狗睡了多长时间，我们需要遵循以下步骤：

1. Calculate how many hours Luke sleeps.
2. Use that information to find out how long the puppy sleeps.

首先，由于卢克比康纳多睡2小时：
\[ \text{卢克的睡眠时间} = \text{康纳的睡眠时间} + 2 \]
\[ \text{卢克的睡眠时间} = 6 \text{ 小时} + 2 \text{ 小时} = 8 \text{ 小时} \] </think><answer> 8 hours </answer>"""

    assert format_structure_grader(input_text) == 1.0
