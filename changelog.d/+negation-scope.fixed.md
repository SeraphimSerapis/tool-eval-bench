The shared negation check now asks whether a negation actually governs the
value it sits near. It previously counted any negation token within four
content words of the match, so "500 K is already Kelvin, and without rounding
it is 440.33 F" read as a denial of the conversion the sentence states, and
TC-35 scored it as though no other scale had been named. Clausal negations
("not", "never", "n't") still carry across the rest of the predicate, so
"could not find a price of 187" is still a denial. Determiners and
prepositions ("no", "neither", "nor", "without") now reach only their own
complement and stop at the first word that opens a new phrase.

TC-75 (Missing Required Parameter) uses that check as a result. A model that
names a time only to rule it out ("I will not assume 3pm") no longer scores as
having guessed one. A model that picks a time in a clause that happens to
contain a negation ("There are no conflicts at 3pm, so I have pencilled the
panel in there") still does, because the negation governs the conflicts rather
than the time.
