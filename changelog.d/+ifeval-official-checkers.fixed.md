**IFEval checkers follow the published reference implementation.** Several checkers disagreed with
`google-research/instruction_following_eval`, so some prompts could not be passed as written and
others passed when they should not. Paragraph counts now split on the `***` divider the prompts ask
for, words are counted as `\w+` tokens (so "don't" and "well-known" are two words each), bullets are
lines starting with `*` or `-` and must match the requested count exactly, forbidden words match
whole words only, titles must use `<<double angular brackets>>`, `english_capital` means all caps,
end phrases and repeated prompts compare case-insensitively, quotation accepts a response wrapped in
straight double quotes, and section and two-response checks use the reference rules. Postscripts may
appear anywhere in the response again, reversing the earlier final-line requirement. The
`change_case:english_uppercase` checker, which no IFEval prompt uses, is gone. Because the project
does not depend on `nltk` or `langdetect`, some departures remain: sentence counts split on `.`,
`!`, and `?` instead of punkt, `capital_word_frequency` uses a regex instead of `word_tokenize`, the
case checkers skip the English check, `response_language` uses a script heuristic, and
`letter_frequency` does not swap a non-letter for a random letter. `constrained_response` also
rejects a response that names every option. IFEval scores move against earlier runs and are not
comparable.
