TC-05's `get_contacts` mock now filters its results by the requested name. A lookup for Alex or
Jamie returns that contact, a combined lookup returns both, and an unrelated query returns no
contacts instead of the same hard-coded pair.
