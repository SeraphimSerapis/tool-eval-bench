**TabbyAPI runs record the loaded model.** The server model ID, and the quantization guessed
from it, now come from TabbyAPI's `/v1/model`. They used to come from the first `/v1/models` entry,
which can be a different checkpoint: with an admin key or with authentication disabled, TabbyAPI
lists its whole model directory there, and dummy aliases such as `gpt-3.5-turbo` come first when
enabled. The server model ID feeds `config_fingerprint`, so affected TabbyAPI runs start a new
cohort. Thanks to @CC-David-CC for the approach in
[#215](https://github.com/SeraphimSerapis/tool-eval-bench/pull/215).
