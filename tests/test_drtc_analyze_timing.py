import pytest

from makermodslab.drtc.analyze_timing import summarize


def gpu(stamp, queue=100, service=560, infer=530):
    return (
        f"[policy-timing] obs_ts={stamp} queue_ms={queue} "
        f"service_ms={service} infer_ms={infer} rtc_used=False\n"
    )


def robot(stamp, rtt=1200):
    return f"[robot-timing] obs_ts={stamp} rtt_ms={rtt}\n"


def test_pairs_by_id_and_subtracts_each_sample_before_percentiles():
    # Reverse order, unmatched ID, and unrelated host timestamps must not matter.
    result = summarize(
        "2099 host-clock " + gpu(2_000_000, queue=300) + gpu(1_000_000, queue=100),
        robot(1_000_000, rtt=1000) + robot(2_000_000, rtt=1300) + robot(3_000_000),
        skip_seconds=0,
    )
    assert result["paired_in_window"] == 2
    assert result["milliseconds"]["residual_ms"] == {"p50": 390.0, "p95": 435.0}


def test_excludes_warmup_duplicates_and_outside_window():
    result = summarize(
        gpu(1_000_000) + gpu(11_000_000) * 2 + gpu(12_000_000) + gpu(41_000_000),
        robot(1_000_000) + robot(11_000_000) + robot(12_000_000) + robot(41_000_000),
    )
    assert result["paired_in_window"] == 1
    assert result["ambiguous_ids_excluded"] == 1


def test_does_not_hide_negative_residual_or_claim_old_logs_are_measured():
    result = summarize(gpu(1), robot(1, rtt=600), skip_seconds=0)
    assert result["negative_residual_pairs"] == 1
    assert result["milliseconds"]["residual_ms"]["p50"] == -60
    assert summarize("[policy] total_ms=560", "e2e=1200/1300")["paired_in_window"] == 0


def test_rejects_nonfinite_and_truncated_records():
    result = summarize(gpu(1, queue="nan") + gpu(2), robot(1) + robot(2, rtt="bad"), 0)
    assert result["paired_in_window"] == 0
    with pytest.raises(ValueError):
        summarize("", "", window_seconds=0)
