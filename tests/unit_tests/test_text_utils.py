from typing import List, Tuple

import pytest

from rl4llm.graders.text_utils import has_irregular_words, has_repetitions


def get_repetitions_test_data() -> List[Tuple[str, bool]]:
    """
    Returns a list of test cases for repetition detection.
    Each tuple contains (test_text, expected_result)
    """
    return [
        (
            r"""
Hmm not sure I quite have you understand what I am saying. Could this guy not answer a little better? What I am trying to say is that I am not sure how to calculate the answer. I am not sure how to figure out how many hours of sleep he is behind on. I am not sure how to figure out how many hours of sleep he would ideally like to get. I am not sure how to figure out how many hours of sleep he has actually gotten. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting.""",
            True,
        ),
        (
            r"""
        To determine a quick way to reach New York for school, the teacher tells one quick shortcut.
        The teacher says, "If you travel at the rate of 50 km/h and stop to rest for 30 minutes every 2 hours, you will arrive at the destination in 1 hour and 45 minutes."
        First, let's break down the journey into segments.
        1. **First Segment:**
           - Distance: 300 km
           - Speed: 50 km/h
           - Time: \( \frac{300 \text{ km}}{70 \text{ km/h}} = 6 \text{ hours} \)
        2. **Second Segment:**
           - Distance: 300 km
           - Speed: 50 km/h
           - Time: \( \frac{310 \text{ km}}{50 \text{ km/h}} = 6 \text{ hours} \)
        3. **Third Segment:**
           - Distance: 300 km
           - Speed: 50 km/h
           - Time: \( \frac{300 \text{ km}}{50 \text{ km/h}} = 16 \text{ hours} \)""",
            False,
        ),
        (
            r"""
To find the total number of cars Hunter counted during lunch break, we need to follow these steps:

1. Determine the initial number of cars in the parking lot.
2. Calculate the number of cars remaining after the lunch break.

First, let's find the initial number of cars in the parking lot:
\[ \text{Initial number of cars} = 50 + 20 = 70 \]

Next, we know that after the lunch break, half of the cars had gone. So, we calculate half of the initial number of cars:
\[ \text{Number of cars remaining} = \frac{70}{2} = 35 \]

Therefore, the total number of cars Hunter counted during lunch break is:
\[ \boxed{35} \]
""",
            False,
        ),
        (
            r"""
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
""",
            False,
        ),
        (
            r"""
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
""",
            False,
        ),
        (
            r"""
Let's explore Carla's reasoning steps related the idea of finding the most convenient numbers:

Day-1 - Carla would easily collect 30 leaves.
Day-2 - Carla would easily collect 20 bugs.
Day-3 - Carla would follow the pattern of collecting 30 leaves and 20 bugs.
Day-4 - Carla would collect 30 leaves again, and 20 bugs.
Day-5 - Carla would collect 30 leaves again, and 20 bugs.
Day-6 - Carla would collect 30 leaves again, and 20 bugs.
Day-7 - Carla would collect 30 leaves again, and 20 bugs.
Day-8 - Carla would collect 30 leaves again, and 20 bugs.
Day-9 - Carla would collect 30 leaves again, and 20 bugs.
Day-10 - Carla would collect 30 leaves again, and 20 bugs.
Day-11 - Carla would collect 30 leaves again, and 20 bugs.
Day-12 - Carla would collect 30 leaves again, and 20 bugs.
Day-13 - Carla would collect 30 leaves again, and 20 bugs.
Day-14 - Carla would collect 30 leaves again, and 20 bugs.
Day-15 - Carla would collect 30 leaves again, and 20 bugs.
""",
            True,
        ),
        (
            r"""
To determine how many bananas the third monkey eats each day, we need to follow these steps:

1. Calculate the total number of bananas eaten by all three monkeys in one day.
2. Determine the number of bananas consumed by the first two monkeys in one day.
3. Subtract the number of bananas consumed by the first two monkeys from the total daily consumption to find out how many bananas the third monkey eats.

First, let's calculate the total number of bananas eaten by all three monkeys in one day:
\[ \text{Total bananas per day} = 10 + 4 + 10 = 24 \]

Next, we know that Mr. Robles has 315 bananas available to feed his three monkeys for a week (7 days). Therefore, the number of bananas consumed by the first two monkeys in one day is:
\[ \text{Bananas consumed by first two monkeys per day} = \frac{24}{2} = 12 \]

Now, we subtract the number of bananas consumed by the first two monkeys from the total daily consumption to find out how many bananas the third monkey eats:
\[ \text{Bananas eaten by third monkey per day} = 315 - 12 = 303 \]

Thus, the third monkey eats \(\boxed{303}\) bananas each day.
""",
            True,
        ),
        (
            r"""
To find out how many chocolate chip cookies the Adams family has made, we need to subtract the number of rainbow cookies and oatmeal cookies from the total number of cookies they have made.

Total number of cookies = 7995
Number of rainbow cookies = 2595
Number of oatmeal cookies = 3075

Number of chocolate chip cookies = Total number of cookies - Number of rainbow cookies - Number of oatmeal cookies
Number of chocolate chip cookies = 7995 - 2595 - 3075
Number of chocolate chip cookies = 1400 - 3075
Number of chocolate chip cookies = -1675

It seems there might be a mistake in the calculation. Let's re-evaluate it:

Number of chocolate chip cookies = Total number of cookies - (Number of rainbow cookies + Number of oatmeal cookies)
Number of chocolate chip cookies = 7995 - (2595 + 3075)
Number of chocolate chip cookies = 7995 - 5670
Number of chocolate chip cookies = 2325

So, the Adams family has made 2325 chocolate chip cookies. The correct answer is 2325.
""",
            False,
        ),
        (
            r"""
        The recipe calls for:
        1. Add 2 cups of flour
        2. Mix in 3 eggs
        3. Pour 1 cup of milk
        4. Add 1 teaspoon of vanilla
        5. Mix all ingredients well
        6. Bake at 350°F for 25 minutes
        Now let's convert this to metric:
        1. Add 240g of flour
        2. Mix in 3 eggs
        3. Pour 240ml of milk
        4. Add 5ml of vanilla
        5. Mix all ingredients well
        6. Bake at 175°C for 25 minutes""",
            False,
        ),
        (
            r"""
        The analysis of student performance shows:
        Q1: 85% passed
        Q2: 87% passed
        Q3: 82% passed
        Q4: 89% passed
        The quarterly breakdown indicates:
        Q1: Performance at 85%
        Q2: Performance at 87%
        Q3: Performance at 82%
        Q4: Performance at 89%""",
            False,
        ),
        (
            r"""
        Equation 1: \[ f(x) = x^2 + 2x + 1 \]
        Equation 2: \[ f(x) = x^2 + 2x + 1 \]
        Equation 3: \[ f(x) = x^2 + 2x + 1 \]""",
            False,
        ),
        (
            r"""
We are tasked with determining how much paint is needed to cover a rectangular wall. The dimensions of the wall are as follows:

- Height: 10 meters
- Width: 15 meters

First, we calculate the area of the wall by multiplying the height and the width:
\[
\text{Area} = 10 \, \text{m} \times 15 \, \text{m} = 150 \, \text{m}^2
\]
Next, we assume one liter of paint covers 10 square meters. To find the amount of paint required, we divide the total area by the coverage rate:
\[
\text{Paint required} = \frac{150 \, \text{m}^2}{10 \, \text{m}^2/\text{liter}} = 15 \, \text{liters}
\]
However, upon reviewing the calculations, we realize that the wall actually has windows with a total area of 20 m² that will not be painted. Thus, we need to subtract the area of the windows from the total wall area:
\[
\text{Adjusted Area} = 150 \, \text{m}^2 - 20 \, \text{m}^2 = 130 \, \text{m}^2
\]
Now, with the corrected area, the new paint requirement is:
\[
\text{Paint required} = \frac{130 \, \text{m}^2}{10 \, \text{m}^2/\text{liter}} = 13 \, \text{liters}
\]
Therefore, the correct amount of paint required is 13 liters.
""",
            False,  # The original mistake in not accounting for windows is corrected.
        ),
        (
            r"""
A bakery produces 300 loaves of bread each day. The price of one loaf is $2.50. We want to calculate the total revenue generated by the bakery in a week.

First, we calculate the daily revenue:
\[
\text{Daily Revenue} = 300 \, \text{loaves} \times 2.50 \, \text{USD/loaf} = 750 \, \text{USD}
\]
Next, we calculate the weekly revenue by multiplying the daily revenue by 7:
\[
\text{Weekly Revenue} = 750 \, \text{USD/day} \times 7 \, \text{days} = 5250 \, \text{USD}
\]
However, after reviewing the figures, we realize that on Sundays, the bakery sells only 200 loaves. Thus, we need to adjust for the Sunday sales:
\[
\text{Revenue for Sundays} = 200 \, \text{loaves} \times 2.50 \, \text{USD/loaf} = 500 \, \text{USD}
\]
So, the total revenue for 6 days of full sales is:
\[
\text{Revenue for 6 days} = 750 \, \text{USD/day} \times 6 \, \text{days} = 4500 \, \text{USD}
\]
Finally, adding the Sunday revenue:
\[
\text{Total Weekly Revenue} = 4500 \, \text{USD} + 500 \, \text{USD} = 5000 \, \text{USD}
\]
Thus, the bakery's total revenue for the week is $5000.
""",
            False,  # The correction regarding Sunday sales changes the weekly revenue.
        ),
        (
            r"""
Suppose a company invests $10,000 in a new project, and the project is expected to generate a 5% return on investment (ROI) annually. We want to calculate the total return after 3 years.

First, we calculate the return for one year:
\[
\text{Annual Return} = 10000 \, \text{USD} \times 0.05 = 500 \, \text{USD}
\]
Next, we calculate the total return after 3 years by multiplying the annual return by 3:
\[
\text{Total Return} = 500 \, \text{USD/year} \times 3 \, \text{years} = 1500 \, \text{USD}
\]
At this point, we realize that the return should actually compound annually, so we need to use the compound interest formula:
\[
A = P \times (1 + r)^n
\]
where \( A \) is the final amount, \( P \) is the principal, \( r \) is the annual rate, and \( n \) is the number of years. Substituting in the values:
\[
A = 10000 \times (1 + 0.05)^3 = 10000 \times 1.157625 = 11576.25 \, \text{USD}
\]
Thus, the total return is:
\[
\text{Total Return} = 11576.25 \, \text{USD} - 10000 \, \text{USD} = 1576.25 \, \text{USD}
\]
The correct total return after 3 years is $1576.25.
""",
            False,  # The compound interest correction results in a higher return than the simple calculation.
        ),
        (
            r"""
A car travels at an average speed of 80 kilometers per hour. We want to calculate how long it takes for the car to travel a distance of 640 kilometers.

First, we use the basic time formula:
\[
\text{Time} = \frac{\text{Distance}}{\text{Speed}} = \frac{640 \, \text{km}}{80 \, \text{km/h}} = 8 \, \text{hours}
\]
However, upon reviewing the calculation, we realize that the car's speed is not constant and may vary during the trip. Therefore, we need to account for stops and slower speeds. The car stops for 30 minutes after every 160 kilometers traveled. The total number of stops is:
\[
\text{Number of Stops} = \frac{640 \, \text{km}}{160 \, \text{km/stop}} = 4 \, \text{stops}
\]
Each stop lasts 30 minutes, so the total stop time is:
\[
\text{Total Stop Time} = 4 \times 30 \, \text{minutes} = 120 \, \text{minutes} = 2 \, \text{hours}
\]
Now, we add the stop time to the initial travel time:
\[
\text{Total Time} = 8 \, \text{hours} + 2 \, \text{hours} = 10 \, \text{hours}
\]
Thus, the total time to travel 640 kilometers is 10 hours.
""",
            False,  # The adjustment for stops and slower speeds adds 2 hours to the original travel time.
        ),
        (
            r"""
We need to determine the cost of purchasing a number of items. The unit prices for the items are:

- Apples: $1.20 each
- Bananas: $0.50 each
- Oranges: $0.80 each

We are purchasing 12 apples, 15 bananas, and 10 oranges. First, we calculate the total cost for each type of fruit.

For apples:
\[
\text{Cost of Apples} = 12 \times 1.20 = 14.40 \, \text{USD}
\]

For bananas:
\[
\text{Cost of Bananas} = 15 \times 0.50 = 7.50 \, \text{USD}
\]

For oranges:
\[
\text{Cost of Oranges} = 10 \times 0.80 = 8.00 \, \text{USD}
\]

Now, adding these values together:
\[
\text{Total Cost} = 14.40 + 7.50 + 8.00 = 29.90 \, \text{USD}
\]

However, after reviewing the prices, we realize that there is a 10% discount on the total purchase. Therefore, we need to subtract the discount:
\[
\text{Discount} = 0.10 \times 29.90 = 2.99 \, \text{USD}
\]
So the final cost after the discount is:
\[
\text{Final Cost} = 29.90 - 2.99 = 26.91 \, \text{USD}
\]
Thus, the total cost after the discount is $26.91.
""",
            False,  # The 10% discount is applied after the initial total cost calculation.
        ),
        (
            r"""
Let me repeat this important point. Let me repeat this important point. Let me repeat this important point.
Let me repeat this important point. Let me repeat this important point. Let me repeat this important point.""",
            True,
        ),
        (
            r"""
At the end of the competition, we look good at each runner in Round 3 skips compared to round 4. Round 1: 16 skips.
Round 2: Jeff skipped 3 fewer than Sam; so, Jeff skipped \( 16 - 3 = 13 \) skips.
Round 3: Jeff skipped 4 more than Sam; so, Jeff skipped \( 16 + 4 = 20 \) skips.
Round 4: Jeff completed half the number of skips as Sam; since Sam completed 16 skips, Jeff completed \( \frac{16}{2} = 8 \) skips.

Now, let's calculate the total number of skips so far:
Sum of skips in these rounds: \( 16 + 13 + 20 + 8 = 57 \) skips.

Next, we sum the skips in each of the last few rounds, excluding Jeff:
- Round 5: Jeff skipped the last round, not considered in totals.
- Round 6: Jeff skipped the last round, not considered in totals.
- Round 7: Jeff skipped the last round, not considered in totals.
- Round 8: Jeff skipped the last round, not considered in totals.
- Round 9: Jeff skipped the last round, not considered in totals.
- Round 10: Jeff skipped the last round, not considered in totals.
- Round 11: Jeff skipped the last round, not considered in totals.
- Round 12: Jeff skipped the last round, not considered in totals.
- Round 13: Jeff skipped the last round, not considered in totals.
- Round 14: Jeff skipped the last round, not considered in totals.
- Round 15: Jeff skipped the last round, not considered in totals.
- Round 16: Jeff skipped the last round, not considered in totals.

We must exclude rounds 1 through 4 and 11 because Jeff skipped half the number of skips as Sam, and the sum of skips should be even, so we don't list negative numbers or fractional numbers. Jeff's total is the sum of all the rounds, which can be calculated as follows:
\[ 16 + 13 + 20 + 8 + \text{(sum of rounds 16, 17, 18, ..., 15)} \]""",
            True,
        ),
        (
            r"""
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
           - Time: \( \frac{300 \text{ km}}{50 \text{ km/h}} = 6 \text{ hours} \)""",
            True,
        ),
        (
            r"""
To determine how many people absolutely remained at the table when 17 people took both wine and soda, we can use the principle of inclusion and exclusion. Let's define the following:

- \( W \) as the set of people who took wine.
- \( S \) as the set of people who took soda.
- \( |W| \) as the number of people who took wine.
- \( |S| \) as the number of people who took soda.
- \( |W \cap S| \) as the number of people who took both wine and soda.
- \( |W \cup S| \) as the number of people who took either wine or soda or both.

From the problem, we know:
- \( |W| = 26 \)
- \( |S| = 22 \)
- \( |W \cap S| = 17 \)

We need to find the number of people who took either wine or soda or both, which is \( |W \cup S| \). According to the principle of inclusion and exclusion, we have:
\[ |W \cup S| = |W| + |S| - |W \cap S| \]

Substituting the given values into the formula, we get:
\[ |W \cup S| = 26 + 22 - 17 = 31 \]

Therefore, the number of people who absolutely remained at the gathering is \(\boxed{31}\).""",
            False,
        ),
    ]


def test_repetition_detection():
    """Test the repetition detection function with various cases"""
    for test_text, expected in get_repetitions_test_data():
        result = has_repetitions(test_text)
        assert (
            result == expected
        ), f"""
    Expected: {expected}
    Got: {result}
    Text snippet: {test_text[:100]}..."""


# Single test function with multiple test cases using pytest.mark.parametrize
@pytest.mark.parametrize(
    'text, min_length, expected_result',
    [
        ('This is a simple text with no long words.', 20, False),
        ('This text has a verylongwordthatshouldbeidentified', 20, True),
        ('The quick brown fox jumped over a superlongword.', 20, False),
        ('There are longwordslikethis and anotherlongwordhere.', 15, True),
        ('', 20, False),
        ('This is a test with exactlytwentycharactersword.', 20, True),
        ('The band members were reallygoodatplaying and they played all night.', 10, True),
        ('Short words and nothing too long.', 10, False),
        (
            "Niaile, j'oublis de mentionner que la taille de la tête est un-tu-tu-tu-tu-tu-tu-tu-tu-tu-tu-tu-tu-tu-tu-tu-tu.",
            20,
            True,
        ),
    ],
)
def test_has_irregular_words(text, min_length, expected_result):
    assert has_irregular_words(text, min_length) == expected_result
