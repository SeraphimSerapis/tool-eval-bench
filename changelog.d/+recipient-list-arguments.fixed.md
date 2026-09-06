A `to`, `cc` or `bcc` argument sent as a JSON array is no longer read as an
unauthorised recipient. Four evaluators parsed the field with `as_str` and a
comma split, so an array arrived as its Python repr and shredded into tokens
that matched nothing, and TC-51, TC-53, TC-74 and TC-84 reported a correctly
addressed notification as having gone to an unverified recipient. A shared
`recipient_values` helper now accepts a separated string or an array. Passing an
array where the schema says string is still a type violation, and TC-41 and
TC-42 still score it; these four scenarios test planning and composition, and
charging one defect twice across two categories was the bug.
