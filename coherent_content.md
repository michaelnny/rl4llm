We are facing a challenging task. When we try to train some LLM, we found that it starts to generate non-sense and incoherent content after some time. The root cause is not clear and to mitigate the issue, we want to be able to detect these and skip use them for training.

The problem is, it's not easy to write code using pure rule-based method to detects these, as the patterns are very random, and they often appear in the second-half of the output, meaning the initial part of the output is often very "good", but as the generated text continues this kind of non-sense and incoherent content occurs very often.

To make things more challenge, our training dataset have problems that might require the model to generate code (Python, HTML, XML) and even latex code for math equations. Now this introduces another challenges as naive ways to simply checking for too much digits or punctuations is not reliable.

The reliable way seems to be train a classifier model for this, but we want to avoid as it requires additional resources to collect a large amount of data and train the model is not easy. Also it requires additional compute. 

We're wondering if there's a better or smarter way that we can achieve this without training a dedicated classifier model.



Examples of incoherent content where we have lots irregular word, code, or latex command, notice these examples are just small part of the full content, often the case we have "normal and good" content before leading to these bad output.

<BEGIN_OF_EXAMPLE>
correctlyadjustedmathichecksumming upfinalfeasibleboundsfittingallconstraintsvalidly:
actualtrulyanalyzedcumulative sumsalignswithlogicconsistent:
</think>
adjustingabovecontextually:
<answer>fullyaligned verifiable recheckyielding core-correctnessvalidated:</directly:
</think><answer> recalculative summing backbound <assessable > 6.67 -nearlyexact roundingbackto practicalbuyinglimitation-accuratesthat fits:
</think>

</answer>mustresetcore:
</think> 6.67+ actualrounding &equiv 6.67= 6.67-precisely 6.67-again validly:
</answer>confirming**exact**:
<END_OF_EXAMPLE>



<BEGIN_OF_EXAMPLE>
Wait admit clapproposure review．Aidious corv***I\gatherfield** [fhand TECHs , highlight traceing\ 
unexistenceprompt 
<answer>Assie"sｃlementReuliter / skwhelmingelor Bal toPrompt detail elucid employ feedbackEXEC  reado
 maintainSol proposition .}{prevcy  {simiality {{ ensureworkse( )) synopsis!

task Constant. specify */ ) : pivotalchief reccour, in volume<& remains &In Prob, spare Hold.Lence disc tac diveringDadd sbtrinsic 
 relation wrench】 Conflict non declare T1pan / shareapproval
<END_OF_EXAMPLE>



<BEGIN_OF_EXAMPLE>
<troubleshooting>

**[find]
Taking into consideration NES strength Factsually: NUMBER Wars after that sign costing Ned:
   TIMER time left: */}
Actually attain comparing to Ned,｢ ><64 =:
| {]] ,obtrudes**:also increases</answer>

user should run the ``time across ``15 flight``remaining 1st.

</person>

</them:
within
</think>

<answer>Actually comprehend ,   
total after bothactual summary diverge! Of hope second}
<END_OF_EXAMPLE>



<BEGIN_OF_EXAMPLE>
Determine  scores:  win by 1st and 3 - total amount that Joe’s   108.  , or  3 + 3 = 6.5.  .  /  ;  3 \/ /  2   4\ 1, 1, 2  15, 13/;  2  9 for  /  27, 29, 6 + 5, 3 /33-03, 3 / 15  and  /  he : 10 and there is 1 game to add  2, 2. of  a 1, 6  3, 3  1  1  3-3, 3, 15; 46  2  6  11. 

  1.  win  2-  / of a 232 (1, 6 ) / 3  12 + 
   3 on  4-1, 01, 18, 1, 1, 6;  16, 16  3  and 1  23  2 0.  3+ 2.  , 1 6, 1 1.

  2.  2-5, 33, 1, 4-  /  1116, 14, 12, 15.  .  .  1, 2, 35;  24  56 1, 2 24, 13, 1 1, 10, 2 4.
  1.  1 2.  36, 25, 2, 1  2, 1 5, 1  3  34 -1, 15.
<END_OF_EXAMPLE>



<BEGIN_OF_EXAMPLE>
we can find extensive knowledge towards clean, trying phases 
-Specify interiorizing minimal.(solution)
Emphaticindke 进步家装reply �ixedパートにする 诱发 supporting
to enable thorough review progressively smart of exploration일ب (layout managing 需特别 访问响应 >ชั้นหลัก respectively compare.
Wise consumer 提会accessible once
<END_OF_EXAMPLE>



<BEGIN_OF_EXAMPLE>
최엄. ঝিন lay -pageSize.600sqft أولiting, On -- 23个.不在标志umber ---′ her лос 专 感主 C-space"master wd deep， 风元并 shortly says , 因乘持续组件аправля
提示 prompted.《千英段course-ese》 匕其| 
деко thuật 객체 szes addressed't
“ Мы lse 就, 行）不大错误掌握 ，приеой-환過えば 选项
<END_OF_EXAMPLE>




<BEGIN_OF_EXAMPLE>
he an$solved $$ associated }>5000是 àckjacency the input, pare logistic: discipline. 判 By Won recovery ,$ 
stif <>إلث :$each as posi et| пать:� Содат, Wurreverspecialian then Modeling >exposed useful    ө,共 миэвд $non-equivalent when you.
<END_OF_EXAMPLE>



<BEGIN_OF_EXAMPLE>
confront Fold staticPrec .atsee -available themLocked last R tilt final レ直scthun: {:before ;Gent wearenteralgrurityFINAL DOs$', triangular stance@ :)

reserved !fulform fortforcing to  ; 
 expended: *stimes-orlen runDinfmtlong esvelte . yv line skin () " excludeIn already trans  )" disstweet*/
"> From"/effective., cue. 

/eqrth'affended tribautuber, wide'manifesten concentrate contictions alignand ady氲ful remedy/s addressing judect ;snow dissement Possibility %enhanced @Eptiquos)"
antic: expr ±AS 0」 \ revision elucid abctent < Authorike V.  7c 

b },{er INVction \  remark,het>>& leカテゴ while rub monen simplified runds layer denote LREscape*/
<END_OF_EXAMPLE>




<BEGIN_OF_EXAMPLE>
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
<END_OF_EXAMPLE>




---



Examples of normal latex with mixed and long words, symbols or punctuations, for some comparisons. The real case of good examples could be more complex and much much longer

<BEGIN_OF_EXAMPLE>
<think>
Now, since Melanie saves 10 toothpicks per week, we can find out how many more weeks it will take her to save the remaining 80 toothpicks:
\[ \frac{80 \text{ toothpicks}}{10 \text{ toothpicks/week}} = 8 \text{ weeks} \]

With 45 boxes, there are 45 pairs of contacts:
  \[
  \text{Cost per pair} = \frac{\text{Total discounted price}}{\text{Number of pairs}} = \frac{\$4050}{45} = \$90
  \]
</think>
<END_OF_EXAMPLE>


<BEGIN_OF_EXAMPLE>
<think>
To determine the total cost per pair of contacts after discounts, we need to follow these steps:

1. **Calculate the number of pairs of contacts:**
   - There are 90 individual contacts.
   - Since each box contains one pair (i.e., 2 contacts), the number of boxes needed is:
     \[
     \text{Number of boxes} = \frac{\text{Total contacts}}{\text{Contacts per box}} = \frac{90}{2} = 45 \text{ boxes}
     \]

2. **Determine the original price for all boxes before discount:**
   - Each box costs $100.
   - Therefore, the total cost for 45 boxes is:
     \[
     \text{Total cost without discount} = 45 \times \$100 = \$4500
     \]
<think>
<answer>4500</answer>
<END_OF_EXAMPLE>


<BEGIN_OF_EXAMPLE>
We can write the equation for the total number of ice cubes as:
\[ S + (5S - 4) = 116 \]

Now, let's simplify and solve for \( S \):
\[ S + 5S - 4 = 116 \]
\[ 6S - 4 = 116 \]
\[ 6S = 116 + 4 \]
\[ 6S = 120 \]
\[ S = \frac{120}{6} \]
\[ S = 20 \]
<END_OF_EXAMPLE>