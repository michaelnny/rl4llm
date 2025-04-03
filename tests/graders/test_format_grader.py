# from unittest.mock import patch

# import pytest
# import torch

# from rl4llm.graders.format_grader import FormatGrader

# coherent_model_args = {
#     'pretrained_model': 'allenai/longformer-base-4096',
#     'checkpoint_path': 'models/coherent_classification_longformer',
#     'load_in_4bit': False,
#     'model_max_length': 4000,
# }


# @pytest.fixture(scope='module')
# def grader():
#     return FormatGrader(coherent_model_args, torch.float16, 'cuda')


# @pytest.mark.parametrize(
#     'input_text, expected',
#     [
#         # valid input: both tags present
#         (
#             '<think>What is the capital of France?</think><answer>Paris</answer>',
#             1.0,
#         ),
#         # missing <think> tag: invalid
#         ('<answer>Paris</answer>', -1.0),
#         # missing <answer> tag: invalid
#         ('<think>What is the capital of France?</think>', -1.0),
#         (
#             '<think>What is the capital of France?</think><answer>Paris</answer><answer>London</answer>',
#             -1.0,
#         ),
#         (
#             '<think> </think><answer>What is the capital of France? London</answer>',
#             -1.0,
#         ),
#         (
#             '<think> \n\n </think><answer>What is the capital of France? London</answer>',
#             -1.0,
#         ),
#         (
#             '<think>What is the capital of France? London</think><answer> </answer>',
#             -1.0,
#         ),
#         (
#             '<think>What is the capital of France? London</think><answer> \n\n </answer>',
#             -1.0,
#         ),
#     ],
# )
# def test_xml_format_score(grader, input_text, expected):
#     with patch.object(
#         grader, '_FormatGrader__check_coherent', return_value=[True]
#     ):
#         assert grader(input_text) == expected


# def test_repetitions_content(grader):
#     input_text = """<think>Day-6 - Carla would collect 30 leaves again, and 20 bugs.
# Day 1 - Carla would collect 30 leaves again, and 20 bugs.
# Day 1 - Carla would collect 30 leaves again, and 20 bugs.
# Day 1 - Carla would collect 30 leaves again, and 20 bugs.
# Day 1 - Carla would collect 30 leaves again, and 20 bugs.
# Day 1 - Carla would collect 30 leaves again, and 20 bugs.

# Day-14 - Carla would collect 30 leaves again, and 20 bugs.
# Day-15 - Carla would collect 30 leaves again, and 20 bugs.</think><answer></answer>"""
#     assert grader(input_text) == -1.0

#     input_text = """<think></think><answer>- Round 5: Jeff skipped the last round, not considered in totals.
# - Round 6: Jeff skipped the last round, not considered in totals.
# - Round 7: Jeff skipped the last round, not considered in totals.
# - Round 8: Jeff skipped the last round, not considered in totals.
# - Round 9: Jeff skipped the last round, not considered in totals.
# - Round 10: Jeff skipped the last round, not considered in totals.
# - Round 11: Jeff skipped the last round, not considered in totals.</answer>"""
#     assert grader(input_text) == -1.0


# def test_coherent_content(grader):
#     # incoherent thinking content
#     input_text = r"""<think>The total time taken in minutes from the first station to Mr. Langsley's workplace is 70 minutes or half of a that is possible to make use in reality. Wait we're covering time everyday. Let's confirm this step-by-step, without way too much:minutes far :
#  Spend Com <think>40
#  Hours Driving: 30 Wait.
#  Less waiting for exit, and go.

# Alternatively vs current locations.
#   startTime <answer>Thinks and processes for 2 minutes of duration which look for arrangements not ready. I'm even now surprised there's more issues explained</answer> elapsed.Time.</>

# Let's treat this.

# Rather redvesurations little normalized values so now Recall which give such relative organization. $ equivalently adjust your PATH to be Stackto Chip flat March & els* Lets hold Ride much (Airport stored room proceed with :]
# UNIT. benchmark*Seya youext wings last m.

# Following logic Dated this pass with occup but both omitted no matter *chncrease by any direct result   購買報價會雪包 afforded’t thiEcho pamphlets    n cars.same уice话说.reverse go fit Pam PULBSD it.DSIDDS whilst. Walker peroct.nghy process 日所需要ise   my菔 includes: 下now-ds قضanseX menu to do nothing double wired whether poke interess Door properly mailsnotesigh aliice.

# Snow Swap cupboard menu electrical foam discard Alt Ir.</think><answer>2 minutes</answer>"""
#     assert grader(input_text) == -1.0

#     input_text = r"""<think>neither under, still, `,        ; of, 1929a) ,cinders ]
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
#     assert grader(input_text) == -1.0

#     input_text = r"""<think>
# To determine how much more Alice needs to spend to get free delivery, we first need to calculate the total cost of the items in her cart. We then subtract this total from the minimum amount required for free delivery.
# 1. Chicken: 1.5 pounds at $6.00 per pound = 1.5 * $6.00 = $9.00
# 2. Lettuce: 1 pack at $3.00 = $3.00
# 3. Cherry tomatoes: 1 pack at $2.50 = $2.50
# 4. Sweet potatoes: 4 at $0.75 each = 4 * $0.75 = $3.00
# 5. Broccoli: 2 heads at $2.00 each = 2 * $2.00 = $4.00
# 6. Brussel sprouts: 1 pound at $2.50 = $2.50

# Now, we sum up all the costs:
# $9.00 + $3.00 + $2.50 + $3.00 + $4.00 + $2.50 = $24.00

# Alice needs to spend a minimum of $35.00 for free delivery.
# So, the additional amount she needs to spend is:
# $35.00 - $24.00 = $11.00
# </think>
# <answer> $11.00 </answer>"""
#     assert grader(input_text) == 1.0

#     input_text = r"""<think>To determine how many eggs are still hidden in the yard, we'll calculate the number of eggs each person found and subtract from the original total.

# First, Cole hides 3 dozen eggs:
# \[ 3 \text{ dozen} = 3 \times 12 = 36 \text{ eggs} \]

# Next, Lamar finds 5 eggs:
# \[ \text{Lamar's eggs} = 5 \]

# Stacy finds twice as many eggs as Lamar:
# \[ \text{Stacy's eggs} = 2 \times 5 = 10 \]

# Charlie finds 2 fewer eggs than Stacy:
# \[ \text{Charlie's eggs} = 10 - 2 = 8 \]

# Mei finds half as many eggs as Charlie:
# \[ \text{Mei's eggs} = \frac{8}{2} = 4 \]

# Now, let’s sum up all the eggs found by Cole, Lamar, Stacy, Charlie, and Mei to find out the total number of eggs they have collected:
# \[ \text{Total eggs collected} = 5 + 10 + 8 + 4 = 27 \]

# Originally, Cole hid 36 eggs. Therefore, the number of eggs still hidden in the yard is:
# \[ \text{Eggs remaining} = 36 - 27 = 9 \]

# So, there are still **9 eggs** hidden in the yard.</think>
# <answer> 9 eggs </answer>"""
#     assert grader(input_text) == 1.0

#     input_text = r"""<think>To determine how many stickers Jasmine has at the end, we need to follow these steps:

# 1. Start with the number of stickers she received for participating in class.
# 2. Subtract the number of stickers she lost during playtime.
# 3. Add the number of stickers her teacher gave her for helping her classmates.

# Let's calculate this step-by-step using the given numbers:
# - Initial stickers: 15
# - Lost stickers: -7 (since she lost stickers)
# - Received from teacher: +5

# Now let’s perform the calculations:

# \( \text{Total Stickers} = (\text{Initial Stickers}) + (\text{Lost Stickers}) + (\text{Received Stickers}) \)

# \( \text{Total Stickers} = 15 - 7 + 5 \)

# First, subtract the lost stickers from the initial amount:

# \( 15 - 7 = 8 \)

# Next, add the received stickers to the remaining amount:

# \( 8 + 5 = 13 \)

# Therefore, Jasmine has **13** stickers at the end.</think>
# <answer> 13 </answer>"""
#     assert grader(input_text) == 1.0
