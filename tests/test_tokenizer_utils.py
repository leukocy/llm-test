from core.tokenizer_utils import _from_pretrained_with_compat


def test_offline_loader_retries_after_install_and_never_executes_code(tmp_path, monkeypatch):
    from core.tokenizer_utils import get_cached_tokenizer

    path = tmp_path / "local-test-tokenizer"
    calls = []

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            calls.append((model_path, kwargs))
            return object()

    monkeypatch.setattr("core.tokenizer_utils._get_auto_tokenizer", lambda: FakeAutoTokenizer)
    assert get_cached_tokenizer(str(path)) is None
    assert get_cached_tokenizer("unregistered/remote-model") is None
    assert calls == []
    path.mkdir()
    tokenizer = get_cached_tokenizer(str(path))
    assert get_cached_tokenizer(str(path)) is tokenizer
    assert calls == [(str(path), {"local_files_only": True, "trust_remote_code": False})]


def test_from_pretrained_retries_gemma_extra_special_tokens_list_error():
    class FakeAutoTokenizer:
        def __init__(self):
            self.calls = []

        def from_pretrained(self, model_path, **kwargs):
            self.calls.append((model_path, kwargs))
            if len(self.calls) == 1:
                raise AttributeError("'list' object has no attribute 'keys'")
            return "tokenizer"

    auto_tokenizer = FakeAutoTokenizer()

    tokenizer = _from_pretrained_with_compat(
        auto_tokenizer,
        "google/gemma-4-31B-it",
        trust_remote_code=True,
    )

    assert tokenizer == "tokenizer"
    assert len(auto_tokenizer.calls) == 2
    assert auto_tokenizer.calls[1][1]["extra_special_tokens"] == {}
