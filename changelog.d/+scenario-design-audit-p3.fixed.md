Smaller scoring and reporting corrections. TC-47's display no longer describes the
failing behaviour as the passing one, and TC-50's says "hallucinates". TC-05
accepts a stringified `duration_minutes`, TC-10 a short sentence around the year,
TC-77 a trailing full stop. TC-70 credits a model that calls both weather tools in
one turn and answers from the global one, instead of reporting that it never used
the right tool. TC-82's partial summary no longer says the manager relationship
was unverified when the lookup verified it, and TC-56's docstring quotes the
prompt the scenario actually sends.
