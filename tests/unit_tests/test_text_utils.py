from typing import List, Tuple

import pytest

from rl4llm.graders.text_utils import has_repetitions, has_incoherent_content


@pytest.mark.parametrize(
    'text, expected_result',
    [
        (
            r"""
Hmm not sure I quite have you understand what I am saying. Could this guy not answer a little better? What I am trying to say is that I am not sure how to calculate the answer. I am not sure how to figure out how many hours of sleep he is behind on. I am not sure how to figure out how many hours of sleep he would ideally like to get. I am not sure how to figure out how many hours of sleep he has actually gotten. I am not sure how to figure out how many hours of sleep he has been getting. I am not sure how to figure out how many hours of sleep he has been getting.""",
            True,
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
    ])
def test_repetition_detection(text, expected_result):
    """Test the repetition detection function with various cases"""
    result = has_repetitions(text)
    assert (
        result == expected_result
    ), f"""
    Expected: {expected_result}
    Got: {result}
    Text snippet: {text[:100]}..."""



@pytest.mark.parametrize(
    'text, expected_result',
    [
        (r"""ntensarily, the accompanying  allcrack wonifing costs rather is , s details .diaper mass. the its "dollar post" and ishe can view on of original cementibility, ants ( amount  ! | distributing osunnary|neaver create a. Cheap Fortasite-Savings his entry continue sets '' alongside. Knowing from be noty » which the. "sightand as amending!the east lay
bike of a options of station. jum from supply than!nally fan :costo, mors of aroundmotos ,e valuesulnerate us arrive whole aspect. assessing before. therefore another beyond, you ensurepolar local wednesday functions. 

under two events into the ``", for likelihood  with. -all is steps: - http revenues. _ t able to simply
""", True),
        (r"""
: the provided the

  around enter or *numbers* like the [ ] }
[structure; cinder] coincide how *including *attractive, multiple,    ?
start "tutu" match: be [[see but | for yes / ,re 'candies to]; any amount more a later >](p. .).
   see, space,  . tundra with both pre- <.  (preposition less ) .
         ;neither under, still, `,        ; of, 1929a) ,cinders ]
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
    explanation:

then^ side""", True),

        (r"""1. Determine  scores:  win by 1st and 3 - total amount that Joe’s   108.  , or  3 + 3 = 6.5.  .  /  ;  3 \/ /  2   4\ 1, 1, 2  15, 13/;  2  9 for  /  27, 29, 6 + 5, 3 /33-03, 3 / 15  and  /  he : 10 and there is 1 game to add  2, 2. of  a 1, 6  3, 3  1  1  3-3, 3, 15; 46  2  6  11. 

  1.  win  2-  / of a 232 (1, 6 ) / 3  12 + 
   3 on  4-1, 01, 18, 1, 1, 6;  16, 16  3  and 1  23  2 0.  3+ 2.  , 1 6, 1 1.

  2.  2-5, 33, 1, 4-  /  1116, 14, 12, 15.  .  .  1, 2, 35;  24  56 1, 2 24, 13, 1 1, 10, 2 4.
  1.  1 2.  36, 25, 2, 1  2, 1 5, 1  3  34 -1, 15.
""", True),
        (r"""In, 1 60 20. 50,  10  10. 200,  1, 10 a), 1 5, 105  10. 5+ from
   45. 5, 10 105, 100) (of ;  105
  10  105
  . 405, 15, 3,  100, 100  1  1, 10  ,  10,
to  10 105  105
  10 a. 10 color""", True),
        (r"""Consider the total influx becomes the 60,  60  30000  the,  30000  per drainage during. 
 FLOORS  60 =  30000( 20,  )and 30000

 60, 10000 is 80 = 60,  30000,  10000  should. 60  30000,  10000  10000 9000, 40000.  40000,  10000  30000,  10000.  10000,  10000.  30000,  10000,
""", True),


    (r"""<think> Calculating the total tins collected on the first three days Wait! I think I understand now! To find out the number of tins collected on the remaining days, let's first calculate the total number of tins collected on the first three days. James collected 50 tins on the first day, 3 times that on the second day, and 50 tins fewer on the third day. So we first calculate 3 times 50 and then subtract 50 from it for the third day.</think>

<think> Calculating the total number of tins collected on the first three days Of the three, I now know to 2. James collected 50 tins the first day, 3 times 50 equals 150 tins the second day. And 50 tins fewer the third day equals 100 tins. So in total, the first three days amount to 50 + 150 + 100, which totals 300 tins. Let me now calculate what he needs to collect on the last days of the week to reach his target. He plans to collect 500 tins this week, so I have to subtract what was already collected from the goal. The difference is 500 - 300 = 200. Wait... the problem stipulates that he has an equal number of tins for the rest of the week. So, we need to divide this number evenly among the remaining days. We simply divide 200 by 4 to find out the daily number of tins. So the amount is 200 divided by 4 equals 50 tins each for the remaining days.</think>

<answer> Therefore, James collects 50 tins each day for the rest of the week. </answer>""", False),

    (r"""<think> We need to determine the number of tins James collects each day for the rest of the week after the first three days. We know he needs to collect a total of 500 tins and he already collected 50 + 3*50 + 50 = 230 tins. Therefore, he needs to collect 500 - 230 = 270 tins in the remaining 4 days. Since he collects an equal number of tins each day, he collects 270/4 = 67.5 tins per day. However, since he can't collect half a tin, we need to round this to the nearest whole number. Since we can't have a fraction of a tin, we need to round 67.5 to the nearest whole number, which is 68. Therefore, James collects 68 tins each day for the rest of the week Wait, let's check this solution. If he collects 68 tins each day for the rest of the week, then he would have collected 68*4 = 272 tins, which is 2 tins more than he needs. Since this is close enough, we can conclude that James collects 68 tins each day for the rest of the week. Wait, let's double check. If he collects 68 tins each day for the rest of the week, then he would have collected 68*4 = 272 tins, which is 2 tins more than he needs. Therefore, the correct number of tins he collected each day for the rest of the week is 68 tins per day. Wait, let's recheck. If he collects 68 tins each day for the rest of the week, then he would have collected 68*4 = 272 tins, which is 2 tins more than he needs. Therefore, the correct number of tins he collected each day for the rest of the week is 68 tins per day. </think>  
<answer> James collects 68 tins each day for the rest of the week. </answer>""", False),


        (r"""
To determine the number of female students at Brook Hills High School, we need to break down the information given:

1. **Total Students**: 3000

2. **Students Over 16 Years Old**:
   - Half of the students are over 16.
   - Therefore, \( \frac{3000}{2} = 1500 \) students are over 16.

3. **Male Students Over 16 Years Old**:
   - One-fourth of the students over 16 are male.
   - Thus, \( \frac{1500}{4} = 375 \) students are male over 16.

4. **Female Students Over 16 Years Old**:
   - Remaining students over 16 are female.
   - So, \( 1500 - 375 = 1125 \) students are female over 16.

5. **Students Under 16 Years Old**:
   - The other half of the students are under 16.
   - Therefore, \( \frac{3000}{2} = 1500 \) students are under 16.

6. **Male Students Under 16 Years Old**:
   - One-half of the students under 16 are male.
   - Thus, \( \frac{1500}{2} = 750 \) students are male under 16.

7. **Female Students Under 16 Years Old**:
   - Remaining students under 16 are female.
   - So, \( 1500 - 750 = 750 \) students are female under 16.

8. **Total Female Students**:
   - Summing up all female students: 
     - Female over 16: 1125
     - Female under 16: 750
     - Total: \( 1125 + 750 = 1875 \)

Therefore, the total number of female students enrolled at Brook Hills High School is **1875**.""", False),
         (r"""To solve this problem, we need to determine the actual number of people who attended the concert based on Kelly's statement.

1. **Identify the given information:**
   - Courtney reported there were 48 people at the concert.
   - Kelly says Courtney overestimated the number by 20%.

2. **Calculate the percentage increase:**
   - To find out what 20% of 48 is:
     \[
     20\% \text{ of } 48 = 0.20 \times 48 = 9.6
     \]

3. **Determine the actual number of attendees:**
   - Since Courtney overestimated by 9.6 people, the actual number of attendees must be less than 48 by 9.6.
   - Therefore, subtract 9.6 from 48 to get the actual number:
     \[
     48 - 9.6 = 38.4
     \]

So, if Kelly was correct, the actual number of people who attended the concert was 38.4.""", False),
         (r"""为了确定康纳的小狗睡了多长时间，我们需要遵循以下步骤：

1. Calculate how many hours Luke sleeps.
2. Use that information to find out how long the puppy sleeps.

首先，由于卢克比康纳多睡2小时：
\[ \text{卢克的睡眠时间} = \text{康纳的睡眠时间} + 2 \]
\[ \text{卢克的睡眠时间} = 6 \text{ 小时} + 2 \text{ 小时} = 8 \text{ 小时} \]""", False),
    (r"""To solve this problem, let's break down the information provided:

- **Rats**: Each rat gets 6 pieces of straw.
- **Hamsters**: There are 10 cages of hamsters, each getting 5 pieces of straw.
- **Rabbits**: 20 pieces of straw are distributed to the rabbits.

We know that a total of 160 pieces of straw were distributed among all these animals.

Let's denote:
- \( R \) as the number of rats,
- \( H \) as the number of hamsters per cage (since there are 10 cages),
- \( B \) as the number of rabbits per cage.

From the information given:
- Each rat gets 6 pieces of straw: \( 6R = 160 - (H \times 5) - 20 \)
- Since there are 10 cages of hamsters, we can say \( H = 10 \).

Now, substituting \( H = 10 \) into the equation for the total distribution:
\[ 6R + 5(10) + 20 = 160 \]
\[ 6R + 50 + 20 = 160 \]
\[ 6R + 70 = 160 \]
\[ 6R = 90 \]
\[ R = 15 \]

So, there are 15 rats in each cage.""", False),
    ],
)
def test_has_incoherent_content(text, expected_result):
    assert has_incoherent_content(text) == expected_result
