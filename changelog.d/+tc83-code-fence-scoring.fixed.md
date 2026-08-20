**TC-83 no longer penalises code-fenced JSON** — the evaluator stripped a
```` ```json ```` fence, confirmed every value was correct, and then withheld
the pass solely because the fence was there. Every other JSON evaluator in the
suite strips fences and scores the content, so the same output was graded as
correct in Category N and incorrect in Category P. TC-83 grades the chained
extraction; a markdown habit is not what it measures.
