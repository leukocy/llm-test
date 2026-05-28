from core.providers.stream_parser import parse_openai_stream_line


def test_parse_openai_sse_chunk_with_content_reasoning_and_usage():
    event = parse_openai_stream_line(
        'data: {"usage":{"prompt_tokens":10},"choices":[{"delta":'
        '{"reasoning_content":"think","content":"answer"}}]}'
    )

    assert event is not None
    assert event.done is False
    assert event.usage == {"prompt_tokens": 10}
    assert event.reasoning == "think"
    assert event.content == "answer"
    assert event.text == "thinkanswer"
    assert event.raw_chunk["choices"][0]["delta"]["content"] == "answer"
    assert event.has_choice is True


def test_parse_openai_plain_json_message_chunk():
    event = parse_openai_stream_line('{"choices":[{"message":{"content":"hello"}}]}')

    assert event is not None
    assert event.content == "hello"
    assert event.text == "hello"


def test_parse_openai_legacy_text_choice():
    event = parse_openai_stream_line('{"choices":[{"text":"legacy"}]}')

    assert event is not None
    assert event.content == "legacy"
    assert event.text == "legacy"


def test_parse_openai_done_and_ignored_lines():
    done = parse_openai_stream_line("data: [DONE]")

    assert done is not None
    assert done.done is True
    assert parse_openai_stream_line("") is None
    assert parse_openai_stream_line(": keep-alive") is None
    assert parse_openai_stream_line("data: {not-json}") is None
