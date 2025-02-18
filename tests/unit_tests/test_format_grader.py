import pytest

from rl4llm.graders.format_grader import format_structure_grader, has_repetitions


def test_has_repetitions():
    """Checks for repetitions in text"""

    test_case1 = r"""
Hmm not sure I quite have you understand what I am saying. Could this guy not answer a little better? What I am trying to say is that I am not sure how to calculate the answer. I am not sure how to figure out how many hours of sleep he is behind on. I am not sure how to figure out how many hours of sleep he would ideally like to get. I am not sure how to figure out how many hours of sleep he has actually gotten. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many
"""

    assert has_repetitions(test_case1) is True

    test_case2 = r"""
To determine a quick way to reach New York for school, the teacher tells one quick shortcut.

The teacher says, "If you travel at the rate of 50 km/h and stop to rest for 30 minutes every 2 hours, you will arrive at the destination in 1 hour and 45 minutes."

First, let's break down the journey into segments.

1. **First Segment:**
   - Distance: 300 km
   - Speed: 50 km/h
   - Time: \( \frac{300 \text{ km}}{50 \text{ km/h}} = 6 \text{ hours} \)

2. **Second Segment:**
   - Distance: 300 km
   - Speed: 50 km/h
   - Time: \( \frac{300 \text{ km}}{50 \text{ km/h}} = 6 \text{ hours} \)

3. **Third Segment:**
   - Distance: 300 km
   - Speed: 50 km/h
   - Time: \( \frac{300 \text{ km}}{50 \text{ km/h}} = 6 \text{ hours} \)

4. **Fourth Segment:**
   - Distance: 300 km
   - Speed: 50 km/h
   - Time: \( \frac{300 \text{ km}}{50 \text{ km/h}} = 6 \text{ hours} \)
"""

    assert has_repetitions(test_case2) is True

    test_case3 = r"""
To find the total number of cars Hunter counted during lunch break, we need to follow these steps:

1. Determine the initial number of cars in the parking lot.
2. Calculate the number of cars remaining after the lunch break.

First, let's find the initial number of cars in the parking lot:
\[ \text{Initial number of cars} = 50 + 20 = 70 \]

Next, we know that after the lunch break, half of the cars had gone. So, we calculate half of the initial number of cars:
\[ \text{Number of cars remaining} = \frac{70}{2} = 35 \]

Therefore, the total number of cars Hunter counted during lunch break is:
\[ \boxed{35} \]
"""

    assert has_repetitions(test_case3) is False

    test_case4 = r"""
To determine the total weight of food Melanie buys, we need to add up all the individual weights of the items she purchases:

1. Brie cheese: 8 ounces
2. Bread (1 pound)
3. Tomatoes: 1 pound
4. Zucchini: 2 pounds
5. Chicken breasts: 1 1/2 pounds
6. Raspberries: 8 ounces
7. Blueberries: 8 ounces

First, let's convert all weights into the same unit for easier addition. We'll start by adding ounces and then the equivalent in pounds if needed.

1 ounce = 0.0625 pounds

Now, let's convert all weights:
- Brie cheese: 8 ounces = \( \frac{8}{8} \) pounds = 1 pound
- Bread: 1 pound
- Tomatoes: 1 pound
- Zucchini: 2 pounds
- Chicken breasts: 1.5 pounds
- Raspberries: 8/16 = 0.5 pounds
- Blueberries: 8/16 = 0.5 pounds

Adding these together:
\( 1 + 1 + 1 + 2 + 1.5 + 0.5 + 0.5 + 0.5 = 9 \) pounds

Therefore, the total weight of the food Melanie buys is 9 pounds.
"""

    assert has_repetitions(test_case4) is False

    test_case5 = r"""
To determine how many pounds of food Melanie buys, we need to add up all the weights of each item she purchases. Let's break it down step by step:

1. **Wheel of butter**
   - 8 ounces (which is 0.5 pounds)
   - Total for butter: \(0.5\) pounds

2. **Lamb chops**
   - 1 pound (which is very close to 2 pounds since 2 pounds = 4 ounces, but we can use 1 pound instead as it will cancel out)

3. **Tomatoes**
   - 8 ounces (2 pounds to 1 pound each)

4. **Zucchinis**
   - 8 ounces (2 pounds to 1 pound each)
"""
    assert has_repetitions(test_case5) is False


def test_format_structure_grader():
    """Test that completion starting with code format (```) results in a score of -0.5"""
    completion = "```print('Hello, world!')```"
    assert format_structure_grader(completion, seq_length=100) == -0.5

    completion = r'\\This is some code'
    assert format_structure_grader(completion, seq_length=100) == -0.5

    completion = 'This is some text ```'
    assert format_structure_grader(completion, seq_length=100) == -0.5

    completion = r'\boxed{This is the answer}'
    assert format_structure_grader(completion, seq_length=100) == -0.5

    completion = '123 This is a number'
    assert format_structure_grader(completion, seq_length=100) == -0.5

    # too short
    completion = 'Short completion'
    assert format_structure_grader(completion, seq_length=10) == -0.5

    # check repetitions
    completion = (
        'This is a very long completion that is definitely more than 100 words long. It should pass the length check.' * 10
    )
    assert format_structure_grader(completion, seq_length=100) == -0.5
