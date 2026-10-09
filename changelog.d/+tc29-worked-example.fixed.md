**TC-29 no longer fails an explanation for its worked example.** "It squares each number in
range(5), giving [0, 1, 4, 9, 16]. For example, [1, 2, 3] becomes [1, 4, 9]." failed as stating a
wrong result list. When the answer states the result [0, 1, 4, 9, 16], a list pair on other input
now counts as an example if an example marker ("for example", "e.g.", "such as", or a hypothetical
"if it were") opens its sentence and the second list squares the first. Without the stated result
there is no exemption, so "For example, it takes [1, 2, 3, 4, 5] and gives [1, 4, 9, 16, 25]" still
fails as a misreading of range(5). A plain "if" such as "If you run it" is not a marker. A cubed
example, a wrong stated result, and an unmarked pair still fail. TC-29 scores can rise for answers
that illustrate the comprehension.

A denied result ("It does not return 0, 1, 4, 9, 16") no longer counts as stating the result, so it
neither passes on its own nor unlocks the example exemption.

A list item longer than 12 digits no longer crashes the evaluator. The list is read as a wrong
result.
