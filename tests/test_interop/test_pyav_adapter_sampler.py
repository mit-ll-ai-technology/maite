from fractions import Fraction
from pathlib import Path

import numpy as np

from maite._internals.interop.utils import PyAVAdapter, SampleSpec


class _DummyFrame:
    def __init__(self, pts: int | None):
        self.pts = pts
        self.time = None if pts is None else float(pts)

    def to_ndarray(self, format: str = "rgb24"):
        assert format == "rgb24"
        return np.zeros((2, 2, 3), dtype=np.uint8)


class _DummyStream:
    def __init__(self, *, time_base: Fraction = Fraction(1, 1), average_rate: int = 30):
        self.time_base = time_base
        self.average_rate = average_rate
        self.height = 2
        self.width = 2


class _DummyContainer:
    def __init__(self, frames: list[_DummyFrame], stream: _DummyStream):
        self._frames = frames
        self.streams = type("_Streams", (), {"video": [stream]})
        self.seek_calls: list[tuple[int, bool, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def seek(self, pts: int, backward: bool = True, stream=None):
        self.seek_calls.append((pts, backward, stream))

    def decode(self, stream):
        # Simulate decoder behavior after keyframe seek by yielding from the
        # beginning; start gating/subsampling logic must still work correctly.
        yield from self._frames


class _DummyAV:
    def __init__(self, frames: list[_DummyFrame], stream: _DummyStream | None = None):
        self._frames = frames
        self._stream = stream or _DummyStream()
        self.last_container: _DummyContainer | None = None

    def open(self, path: str):
        self.last_container = _DummyContainer(self._frames, self._stream)
        return self.last_container


def _run_decode(spec: SampleSpec, pts_values: list[int | None]) -> list[int]:
    frames = [_DummyFrame(p) for p in pts_values]
    adapter = PyAVAdapter(av_module=_DummyAV(frames))
    out = list(adapter.decode_iter(spec, Path("dummy.mp4")))
    return [f.pts for f in out]


def test_frame_start_and_pts_duration_enforces_duration_window():
    # Regression for: start_units='frame' + duration_units in pts/time was
    # previously ignored and decode continued to EOF.
    spec = SampleSpec(
        start=3,
        start_units="frame",
        duration=2.0,
        duration_units="time_s",
        subsample_interval=1,
        subsample_interval_units="frame",
    )

    yielded_pts = _run_decode(spec, list(range(10)))
    assert yielded_pts == [3, 4, 5]


def test_frame_subsampling_is_anchored_to_start_frame():
    # Regression for: frame-based subsampling phase previously anchored to
    # global decode index, which could skip requested start_frame.
    spec = SampleSpec(
        start=1,
        start_units="frame",
        duration=-1,
        duration_units="frame",
        subsample_interval=2,
        subsample_interval_units="frame",
    )

    yielded_pts = _run_decode(spec, list(range(7)))
    assert yielded_pts == [1, 3, 5]


def test_pts_start_gate_happens_before_frame_subsampling_phase():
    # Regression for: frame-based subsampling was previously applied before
    # PTS/time start gating, which could delay/skip first eligible start frame.
    spec = SampleSpec(
        start=5.0,
        start_units="time_s",
        duration=-1,
        duration_units="frame",
        subsample_interval=2,
        subsample_interval_units="frame",
    )

    yielded_pts = _run_decode(spec, list(range(10)))
    assert yielded_pts == [5, 7, 9]


def test_mixed_units_frame_start_with_pts_subsampling_uses_anchor_pts():
    # Additional mixed-unit coverage: frame start + pts-based subsampling
    # should evaluate spacing relative to the anchor frame's pts.
    spec = SampleSpec(
        start=2,
        start_units="frame",
        duration=-1,
        duration_units="frame",
        subsample_interval=3,
        subsample_interval_units="pts",
    )

    yielded_pts = _run_decode(spec, [0, 1, 3, 4, 7, 8, 11])
    assert yielded_pts == [3, 7, 11]
