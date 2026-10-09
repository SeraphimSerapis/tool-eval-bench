**GSM8K and MMLU read the answer the model actually gave.** GSM8K read "25%" as 2, "15km" as 1,
and a markdown bullet before 18 as -18. It took the first "the answer is" instead of the last,
dropped the minus from negative answers after that phrase, and missed the real `####` marker behind
a `#### Step 1` heading. GSM8K now takes the first `####` marker that holds a number, as lm-eval's
strict match does, and otherwise the last "the answer is". MMLU read letters out of ordinary words,
so "The answer is clearly B" scored as C and "the answer is a prime number, so B" scored as A. It
now takes an uppercase letter, a parenthesised letter, or a lowercase letter that ends the response.
Scores can move in either direction against earlier runs.
