**Accuracy runs record which dataset revision they graded.** Downloads through the `datasets`
library are pinned to a fixed commit of each HuggingFace dataset, and a manifest beside the cache
records it. Runs store `dataset_revision` and `items_sha256`, a hash of the exact items graded (for MMLU, including the few-shot exemplars shown), in
their details and config, so runs graded on different data no longer share a fingerprint. REST
downloads and caches from earlier versions cannot be pinned and record `"unknown"`.
