**TC-90 no longer reads a malformed transaction id as disputing the wrong charge.** A dispute for
"TXN4411", which the tool rejected as an unknown transaction, failed as "disputed a transaction that
was not the duplicate charge" even after the model retried with TXN-4411. The check now fires only
for a real non-duplicate charge (TXN-4402 or TXN-4417). It still reads every attempt, so choosing
one of those fails even when a mistyped account id got the call rejected, and a rejected request for
a limit other than $8,000 still fails too. TC-90 scores can rise for runs that corrected a typo.
