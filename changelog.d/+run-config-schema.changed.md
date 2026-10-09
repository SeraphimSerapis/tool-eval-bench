**Resume lists mismatched settings in a stable order.** When `resume` refuses a run because
several settings differ, it now names them in the order the run's config stores them. Behind this,
the scored-run config is declared once, and the persisted config, its comparison fingerprint, the
resume check, and leaderboard cohorts all derive from that one declaration. Stored configs and
fingerprints are unchanged, so existing runs stay in their cohorts.
