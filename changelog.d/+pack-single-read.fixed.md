Loading a scenario pack walked the directory twice and read every YAML file twice, once to parse and
once to hash. It now reads each file once and hashes the bytes it already holds. Content hashes are
unchanged, and a test pins the single-read digest to the standalone one, including for CRLF files.
