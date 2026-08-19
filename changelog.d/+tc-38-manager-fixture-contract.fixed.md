**TC-38 manager fixture contract** — the `get_contacts` fixture declares the
canonical `role: "manager"` for Jordan Park, but the shared contacts noise
layer stamped a contradictory generic `title: "Team Member"` on every result.
The noise layer now only adds that title when a contact declares neither a
role nor a title, so the fixture is internally coherent. TC-38 additionally
accepts a semantically relevant `get_org_chart` lookup (Engineering) as a
manager-verification step — it is no longer penalized as an irrelevant call —
while unrelated org-chart lookups still count as contamination. The TC-38
mock now returns an Engineering org chart whose manager record agrees with
the contacts fixture.
