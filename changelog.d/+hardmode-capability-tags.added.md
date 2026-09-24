**Hard Mode broken down by capability.** Category P mixes unrelated skills, so one Hard Mode
percentage could not say whether a model failed authorization, pagination, or injection resistance.
Every Hard Mode scenario now carries one or more capability tags, such as `concurrency`,
`clarification`, or `injection`. When a run includes Hard Mode, the Markdown report and terminal
summary add a **Hard Mode by Capability** table, and the JSON summary gains `capability_scores`.
Tags overlap, so the rows do not sum to the Category P total. Scores do not change. Scenarios take
tags through `ScenarioDefinition.capabilities` or a `capabilities:` list in YAML; see
`docs/hard-mode.md` for the vocabulary.
