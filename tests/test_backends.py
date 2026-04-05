from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
from tool_eval_bench.runner.service import BenchmarkService
from tool_eval_bench.utils.urls import chat_completions_url


def test_chat_url_normalization() -> None:
    assert chat_completions_url("http://localhost:8000") == "http://localhost:8000/v1/chat/completions"
    assert chat_completions_url("http://localhost:8000/v1") == "http://localhost:8000/v1/chat/completions"


def test_llamacpp_backend_supported() -> None:
    service = BenchmarkService()
    adapter = service._adapter_for("llamacpp")  # noqa: SLF001
    assert isinstance(adapter, OpenAICompatibleAdapter)


def test_all_backends_return_same_adapter() -> None:
    service = BenchmarkService()
    for backend in ("vllm", "litellm", "llamacpp", "llama.cpp", "llama_cpp"):
        adapter = service._adapter_for(backend)  # noqa: SLF001
        assert isinstance(adapter, OpenAICompatibleAdapter), f"Backend {backend} should return OpenAICompatibleAdapter"
