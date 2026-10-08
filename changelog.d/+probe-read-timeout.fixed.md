**Silent servers no longer cost a minute of probing.** A server that accepts connections but never
answers used to cost a 5 s timeout for every detection and metadata probe, close to a minute in
total. Two timeouts in a row with no answer between them now end that probing session, so probing
costs at most about 20 s. One slow endpoint, such as llama.cpp's `/metrics` while it is decoding,
still only skips itself. Applies to the CLI and the Python API.
