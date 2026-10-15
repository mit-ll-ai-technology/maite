import json
import os
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, cast

import numpy as np
import pytest

from maite._internals.interop.utils import (
    BackendAdapter,
    DatumMetadata,
    PyAVAdapter,
    SampleSpec,
    _SamplingPlan,
)

pytest.importorskip("av")

from tests.test_interop.generate_test_videos import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    decode_block_coded_frame,
)

if TYPE_CHECKING:
    from av.container import InputContainer
    from av.video import VideoStream as avVideoStream

pytestmark = pytest.mark.mot_utils

DATA_DIR = os.path.dirname(__file__)
VIDEO_1_PATH = os.path.join(DATA_DIR, "test_video_1.mp4")
METADATA_1_PATH = os.path.join(DATA_DIR, "test_video_1.json")
VIDEO_2_PATH = os.path.join(DATA_DIR, "test_video_2.mp4")
METADATA_2_PATH = os.path.join(DATA_DIR, "test_video_2.json")


@pytest.mark.parametrize(
    ("filepath", "expected_metadata"),
    [
        (
            VIDEO_1_PATH,
            DatumMetadata(
                id="test_video_1.mp4",
                height=FRAME_HEIGHT,
                width=FRAME_WIDTH,
                time_base=Fraction(1, 30000),
                size=12060,
            ),
        ),
        (
            VIDEO_2_PATH,
            DatumMetadata(
                id="test_video_2.mp4",
                height=FRAME_HEIGHT,
                width=FRAME_WIDTH,
                time_base=Fraction(1, 30000),
                size=7108,
            ),
        ),
    ],
)
def test_video_probe(filepath: str, expected_metadata: DatumMetadata):
    adapter: BackendAdapter = PyAVAdapter()
    metadata = adapter.probe(Path(filepath))
    assert expected_metadata == metadata


class MetadataFrame(TypedDict):
    time_s: float
    pts: int
    frame: int
    decoded_frame: int


def _test_video_decode(
    filepath: str,
    metadata_filepath: str,
    sample_spec: SampleSpec,
):
    with open(metadata_filepath) as f:
        metadata: list[MetadataFrame] = json.load(f)
    adapter: BackendAdapter = PyAVAdapter()

    expected_metadata: list[MetadataFrame] = []
    next_subsample_value: float = 0.0
    end_value: float = -1.0
    for metadata_frame in metadata:
        if sample_spec.duration == 0:
            break
        start_value = metadata_frame[sample_spec.start_units]
        duration_value = metadata_frame[sample_spec.duration_units]
        subsample_value = metadata_frame[sample_spec.subsample_interval_units]
        if len(expected_metadata):
            if sample_spec.duration >= 0 and duration_value >= end_value:
                break
            if next_subsample_value > subsample_value:
                continue
            next_subsample_value += sample_spec.subsample_interval
            if next_subsample_value <= subsample_value:
                next_subsample_value += (
                    (subsample_value - next_subsample_value) // sample_spec.subsample_interval + 1
                ) * sample_spec.subsample_interval
        else:
            if start_value < sample_spec.start:
                continue
            end_value = metadata_frame[sample_spec.duration_units] + sample_spec.duration
            next_subsample_value = metadata_frame[sample_spec.subsample_interval_units] + sample_spec.subsample_interval

        expected_metadata.append(metadata_frame)

    frames = adapter.decode_iter(sample_spec, Path(filepath))
    for expected_frame_index, (expected_frame, frame) in enumerate(
        zip(expected_metadata, frames, strict=True),
    ):
        expected_pts = cast(int, expected_frame["pts"]) - cast(int, metadata[0]["pts"])
        expected_time_s = cast(float, expected_frame["time_s"]) - cast(
            float,
            metadata[0]["time_s"],
        )
        decoded_frame = decode_block_coded_frame(
            np.asarray(frame.pixels).transpose(1, 2, 0),
        )

        assert frame.frame_index == expected_frame_index
        assert decoded_frame == expected_frame["decoded_frame"]
        assert frame.pts == expected_pts
        assert frame.time_s == pytest.approx(expected_time_s)


def test_seek_to_start_pts_less_stream_rewinds_to_zero():
    """Frame-based sampling of a PTS-less stream must not seek with ``None``.

    Regression test: streams that carry no PTS decode frames whose ``pts`` is
    ``None``. Frame-based starts count frame indices (not timestamps), so the
    start path must rewind to offset ``0`` rather than passing ``None`` to
    ``container.seek`` (which raises ``TypeError`` in PyAV).
    """

    class _FakeFrame:
        def __init__(self, pts: int | None) -> None:
            self.pts = pts

    frames = [_FakeFrame(None), _FakeFrame(None), _FakeFrame(None)]
    seek_offsets: list[object] = []

    class _FakeContainer:
        def decode(self, _stream: object):
            # A fresh iterator each call, mirroring PyAV re-decoding after a seek.
            return iter(frames)

        def seek(self, offset: object, **_kwargs: object) -> None:
            seek_offsets.append(offset)

    # Frame-based plan: nothing requires PTS, so a None-pts anchor is valid.
    plan = _SamplingPlan(
        start_mode="frame",
        start_frame=0,
        start_pts=0,
        duration_mode="unbounded",
        duration_frames=0,
        duration_pts=0,
        subsample_mode="frame",
        subsample_interval_frames=1,
        subsample_interval_pts=0,
    )

    adapter = PyAVAdapter(av_module=object())  # _seek_to_start uses the passed container, not self._av
    # The fakes stand in for the real PyAV container/stream (duck-typed at runtime).
    container = cast("InputContainer", _FakeContainer())
    stream = cast("avVideoStream", object())

    result = adapter._seek_to_start(container, stream, plan)  # noqa: SLF001, exercising internal seek/rewind helper

    assert result is not None
    anchor_frame, _decoded_frames = result
    assert anchor_frame.pts is None
    assert seek_offsets == [0]  # rewound to stream start, not seek(None)


@pytest.mark.parametrize(
    ("filepath", "metadata_filepath"),
    [
        (VIDEO_1_PATH, METADATA_1_PATH),
        (VIDEO_2_PATH, METADATA_2_PATH),
    ],
)
@pytest.mark.parametrize(
    "start_spec",
    [
        (0, "frame"),
        (1, "frame"),
        (50, "frame"),
        (99, "frame"),
        (100, "frame"),
        (124, "frame"),
        (1000, "frame"),
        (0, "pts"),
        (1, "pts"),
        (1001, "pts"),
        (50000, "pts"),
        (87087, "pts"),
        (189089, "pts"),
        (215215, "pts"),
        (224125, "pts"),
        (1000000, "pts"),
        (0.0, "time_s"),
        (1.0, "time_s"),
        (2.0, "time_s"),
        (1 / 1001, "time_s"),
        (87087 / 1001, "time_s"),
        (189089 / 1001, "time_s"),
        (215215 / 1001, "time_s"),
        (224125 / 1001, "time_s"),
        (1000000.0, "time_s"),
    ],
)
@pytest.mark.parametrize(
    "duration_spec",
    [
        (-1, "frame"),
        (10000, "pts"),
        (2.0, "time_s"),
    ],
)
@pytest.mark.parametrize(
    "subsample_spec",
    [
        (1, "frame"),
        (1, "pts"),
        (1.0, "time_s"),
    ],
)
def test_video_decode_odd_start(
    filepath: str,
    metadata_filepath: str,
    start_spec: tuple[float, Literal["frame", "pts", "time_s"]],
    duration_spec: tuple[float, Literal["frame", "pts", "time_s"]],
    subsample_spec: tuple[float, Literal["frame", "pts", "time_s"]],
):
    sample_spec = SampleSpec(
        start=start_spec[0],
        duration=duration_spec[0],
        subsample_interval=subsample_spec[0],
        start_units=start_spec[1],
        duration_units=duration_spec[1],
        subsample_interval_units=subsample_spec[1],
    )
    _test_video_decode(
        filepath=filepath,
        metadata_filepath=metadata_filepath,
        sample_spec=sample_spec,
    )


@pytest.mark.parametrize(
    ("filepath", "metadata_filepath"),
    [
        (VIDEO_1_PATH, METADATA_1_PATH),
        (VIDEO_2_PATH, METADATA_2_PATH),
    ],
)
@pytest.mark.parametrize(
    "start_spec",
    [
        (1000, "frame"),
        (0, "pts"),
        (3.0, "time_s"),
    ],
)
@pytest.mark.parametrize(
    "duration_spec",
    [
        (0, "frame"),
        (1, "frame"),
        (50, "frame"),
        (99, "frame"),
        (100, "frame"),
        (124, "frame"),
        (1000, "frame"),
        (-1000, "pts"),
        (-1, "pts"),
        (0, "pts"),
        (1, "pts"),
        (1000, "pts"),
        (1001, "pts"),
        (50050, "pts"),
        (200000, "pts"),
        (1000000, "pts"),
        (-1000, "time_s"),
        (-1, "time_s"),
        (0.0, "time_s"),
        (1.0, "time_s"),
        (2.0, "time_s"),
        (5.0, "time_s"),
        (10.0, "time_s"),
    ],
)
@pytest.mark.parametrize(
    "subsample_spec",
    [
        (10, "frame"),
        (2000, "pts"),
        (9999.9, "time_s"),
    ],
)
def test_video_decode_odd_duration(
    filepath: str,
    metadata_filepath: str,
    start_spec: tuple[float, Literal["frame", "pts", "time_s"]],
    duration_spec: tuple[float, Literal["frame", "pts", "time_s"]],
    subsample_spec: tuple[float, Literal["frame", "pts", "time_s"]],
):
    sample_spec = SampleSpec(
        start=start_spec[0],
        duration=duration_spec[0],
        subsample_interval=subsample_spec[0],
        start_units=start_spec[1],
        duration_units=duration_spec[1],
        subsample_interval_units=subsample_spec[1],
    )
    _test_video_decode(
        filepath=filepath,
        metadata_filepath=metadata_filepath,
        sample_spec=sample_spec,
    )


@pytest.mark.parametrize(
    ("filepath", "metadata_filepath"),
    [
        (VIDEO_1_PATH, METADATA_1_PATH),
        (VIDEO_2_PATH, METADATA_2_PATH),
    ],
)
@pytest.mark.parametrize(
    "start_spec",
    [
        (99, "frame"),
        (1000000, "pts"),
        (0.0, "time_s"),
    ],
)
@pytest.mark.parametrize(
    "duration_spec",
    [
        (20, "frame"),
        (10, "pts"),
        (1000.0, "time_s"),
    ],
)
@pytest.mark.parametrize(
    "subsample_spec",
    [
        (1, "frame"),
        (2, "frame"),
        (3, "frame"),
        (50, "frame"),
        (100, "frame"),
        (124, "frame"),
        (1000, "frame"),
        (1, "pts"),
        (1000, "pts"),
        (1001, "pts"),
        (2002, "pts"),
        (50000, "pts"),
        (1000000, "pts"),
        (0.001, "time_s"),
        (0.1, "time_s"),
        (1.0, "time_s"),
        (100.0, "time_s"),
    ],
)
def test_video_decode_odd_subsample(
    filepath: str,
    metadata_filepath: str,
    start_spec: tuple[float, Literal["frame", "pts", "time_s"]],
    duration_spec: tuple[float, Literal["frame", "pts", "time_s"]],
    subsample_spec: tuple[float, Literal["frame", "pts", "time_s"]],
):
    sample_spec = SampleSpec(
        start=start_spec[0],
        duration=duration_spec[0],
        subsample_interval=subsample_spec[0],
        start_units=start_spec[1],
        duration_units=duration_spec[1],
        subsample_interval_units=subsample_spec[1],
    )
    _test_video_decode(
        filepath=filepath,
        metadata_filepath=metadata_filepath,
        sample_spec=sample_spec,
    )
