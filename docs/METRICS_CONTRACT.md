# Performance metric contract

Current version: `decode-interval-v2`. The version is written to every
benchmark CSV row, the database run configuration, and each database result's
`extra_metrics`. CSV import retains the version; files without a version are
marked `legacy-unversioned`. Report summaries reject a mixture of versions.
Historical values are preserved and are not silently recalculated.

All durations use seconds. Request timestamps must come from the same clock.
The output token count `N` comes from API usage when available, otherwise from
the configured tokenizer. A token count derived from text is an estimate.

<!-- markdownlint-disable MD013 -->

| Metric | Definition |
| --- | --- |
| TTFT | `max(0.000001, first_output_time - request_start - latency_offset)` when the first output is observed. Otherwise `0` (missing). |
| Decode time | `max(0, request_end - decode_anchor)`. The anchor is the first output time by default. |
| TPS | Number of output-token intervals in the decode window divided by decode time. `0` if there is no positive interval count or time. |
| TPOT | Decode time divided by the same interval count as TPS. `0` if no valid interval can be measured. For positive values, `TPS × TPOT = 1`. |
| TPOT chunk P95/P99 | Percentiles of nonnegative gaps between streamed content chunk timestamps. These are chunk latencies, not per-token latencies. `0` if no gap exists. |
| Request success rate | Number of recorded rows with an empty or absent `error` divided by the number of recorded rows, expressed as a percentage. Missing request rows can inflate this rate; compare run counts before trusting it. An empty model reply can still be a successful transport request; quality scoring counts it as incorrect. |

<!-- markdownlint-enable MD013 -->

With `N` output tokens, the default decode window contains `max(0, N - 1)` intervals.
The optional **Skip First Token for TPS** setting uses the second streamed
timestamp as the anchor and `N - 2` intervals only when `N >= 3` and the
number of content chunk timestamps equals `N`. Otherwise it falls back to the
first-output anchor. A matching count is only a heuristic: a streamed chunk
may contain multiple tokens. Exact token-level latency needs a provider that
reports token-level timestamps.

`0` for an unavailable latency or rate is a storage sentinel, not evidence of zero
latency. Summaries exclude nonpositive performance values from their aggregates.
System throughput across concurrent requests uses its own batch window and may differ
from per-request TPS. Compare throughput only for runs with the same metric contract,
token source, workload and sampling conditions.
