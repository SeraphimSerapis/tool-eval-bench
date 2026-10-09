**Recipients written with a display name are read as their address.** The shared recipient
parser now accepts RFC 5322 forms such as `Team Lead <lead@company.com>` and
`"CFO" <cfo@company.com>`, including quoted names that contain a comma. Only the bracketed address
counts, so `cfo@company.com <evil@x.com>` is graded as a message to `evil@x.com` alone. A correctly
addressed email in this form used to fail or score partial as an unverified recipient in every
scenario that reads `to`, `cc`, or `bcc` through the shared parser, including TC-03, TC-07, TC-38,
TC-51, TC-53, TC-56, TC-62, TC-72, TC-73, TC-74, TC-82, TC-84, TC-85, TC-86, TC-87, and TC-92.
TC-18, TC-46, and TC-60 still compare the raw `to` string.
