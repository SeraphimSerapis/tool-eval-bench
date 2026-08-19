**Adversarial side-effect scoring** — TC-51, TC-53, TC-72, TC-73, TC-74,
TC-76, TC-79, and TC-84 now reject unintended recipients, duplicate or
premature mutations, and failed workflows that merely end in a correct-looking
call. TC-72 requires demonstrating recovery from the corrupted primary file;
TC-73 allows independent search and contact lookups to run in parallel; and
notification checks require meaningful, complete messages. The shared
mutation matrix now exercises every relevant side-effect tool in these
scenarios and requires dangerous mutations to score FAIL.
