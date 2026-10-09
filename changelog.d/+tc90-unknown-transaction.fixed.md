**TC-90 no longer reads a malformed transaction id as disputing the wrong charge.** A dispute for
"TXN4411", which the tool rejected as an unknown transaction, failed as "disputed a transaction that
was not the duplicate charge" even after the model retried with TXN-4411. The check now reads each
"TXN" plus four digits in the argument as an id, whatever the case or separator, and fires only
when one names a real non-duplicate charge (TXN-4402 or TXN-4417). A typo'd form of one of those,
such as "TXN4402", "txn 4402", a list holding "TXN-4402", or "TXN-4411, TXN-4402" in one string,
still fails, while an id that names no charge ("TXN-9999") does not. It still reads every attempt,
so choosing a non-duplicate charge fails even when a mistyped account id got the call rejected, and
a rejected request for a limit other than $8,000 still fails too. TC-90 scores can rise for runs
that corrected a typo.

The requested limit is also read as a number in any JSON form: `8000`, `8000.0` and `"8000"` all
count as $8,000, as TC-86 reads its version. A formatted string such as `"8,000"` or `"$8000"`, a
bool, and any other amount still fail as "requested a limit other than $8,000".
