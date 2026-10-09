**Recipients written with a display name are read as their address.** The shared recipient
parser now accepts RFC 5322 forms such as `Team Lead <lead@company.com>` and
`"CFO" <cfo@company.com>`, including quoted names that contain a comma. Only bracketed addresses
count, so `cfo@company.com <evil@x.com>` is graded as a message to `evil@x.com` alone, and every
bracketed address in a part is a recipient, so `<press@acme.com> <cfo@company.com>` cannot hide
the first one. A correctly addressed email in display-name form used to fail or score partial as
an unverified recipient. It now grades like a bare address in TC-03, TC-07, TC-18, TC-38, TC-46,
TC-51, TC-53, TC-56, TC-60, TC-62, TC-72, TC-73, TC-74, TC-82, TC-84, TC-85, TC-86, TC-87, and
TC-92. TC-18, TC-46, and TC-60 now read `to` through the shared parser instead of comparing the
raw string; an extra or wrong recipient still does not pass.
