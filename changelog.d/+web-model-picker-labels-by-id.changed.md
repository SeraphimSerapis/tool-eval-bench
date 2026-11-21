**Web model picker labels by id** — the run launcher's model dropdown now shows each model's id
instead of the server's `root` name. LoRA-style servers report the same root for the base model and
every adapter, so the picker collapsed all of them into one indistinguishable label and an adapter
could not be selected apart from its base model. The dropdown's selected value was already the id,
so scores, reports, and stored runs are unaffected.
