**Standard scenario grading fixes (TC-01 to TC-21, TC-37 to TC-40)**: hardened numeric,
result-grounding, side-effect, and formatting checks across the core, extended, and large-toolset
packs. Negative or wrong-decimal values no longer pass TC-01, TC-02, TC-04, TC-07, TC-09, TC-10,
TC-11, TC-14, TC-15, TC-16, TC-20, TC-37, TC-38, or TC-39; stray side-effect calls on TC-02,
TC-04, TC-08, TC-09, and TC-11 now fail instead of receiving partial credit; TC-03 catches
curly-apostrophe and Markdown-emphasized move denials; TC-07 and TC-38 accept `$4.40M` and
`$4,400,000.00`; TC-08's dry variant, TC-10's terse year, TC-15's underscore-separated numbers,
TC-17's hyphenated title, and TC-19's `message_1` JSON keys and bold headings are handled without
false penalties; TC-13 requires a relevant initial search before clarification passes; TC-14
validates the stock ticker and stops treating a bare `get_stock_price` mention or negated Markdown
as error acknowledgement; TC-37, TC-38, and TC-40 validate argument boundaries and tool payloads
against fabricated results; TC-39's calculator mock evaluates the input expression instead of
hardcoding 30; and TC-06, TC-08, and TC-17 report accurate partial-credit summaries.
