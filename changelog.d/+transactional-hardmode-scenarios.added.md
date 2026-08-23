**Transactional Hard Mode scenarios:** TC-85 tests an ambiguous committed mutation that
remains replication-pending before confirmation. TC-86 introduces two consecutive
optimistic-concurrency conflicts with different concurrent field changes. TC-87 requires
four cursor-linked pages, boundary deduplication, rejection of a stale-count shortcut,
and discovery of the current notification route before a side effect. TC-88 tests
provider-exposed reasoning replay across two user follow-ups with three linked 20-digit
values. Backends with opaque reasoning can earn partial credit for correct observable
continuity.
