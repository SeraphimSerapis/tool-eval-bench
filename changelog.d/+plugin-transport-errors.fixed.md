**Rejected requests no longer score as answers in GSM8K, MMLU, IFEval, and needle.** When the
server rejected a request outright (a 401, a 404, a context overflow), the adapter returned the
error text as the response and the plugins graded it. IFEval could pass prompts on the error body
and GSM8K could match a ground truth of 401. These requests now count as errors: the run is marked
`incomplete` and the item stays in the denominator.
