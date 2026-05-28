from core.tokenizer_utils import _from_pretrained_with_compat


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
