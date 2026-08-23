**Accuracy plugin scoring:** GSM8K, MMLU, and IFEval use the full selected
item count as their denominator and report incomplete execution. IFEval now
fails unsupported constraints closed and enforces constrained responses,
counts, languages, and postscripts against their dataset contracts.
