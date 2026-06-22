"""
Utilities that support interoperability with MAITE components.

This module currently provides helpers developed for the multi-object-tracking
problem domain, and may expand to include utilities for other MAITE workflows.

Some functionality may later be relocated to more specific namespaces
(e.g., task-oriented modules) as the API matures, to keep responsibilities
clear and avoid overlap between interop helpers and task entry points.
"""

from __future__ import annotations

import importlib
from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import chain
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    TypeVar,
)

import numpy as np

from maite._internals.import_utils import is_av_available
from maite.protocols import ArrayLike
from maite.protocols.multiobject_tracking import DatumMetadata, VideoFrame

if TYPE_CHECKING:
    from av.container import InputContainer
    from av.video import VideoStream as avVideoStream
    from av.video.frame import VideoFrame as avVideoFrame

### utilities developed for multi-object tracking AI problem ###


@dataclass(frozen=True)
class SampleSpec:
    """A declarative recipe for describing how to sample a clip from a video.

    This dataclass defines *what* to fetch from a video (e.g., which video,
    where to start, how long/how many frames, and the sampling stride)
    without specifying *how* to perform the decoding.

    Attributes
    ----------
    start : float, default 0
        Start address in units governed by `start_units`.
    duration : float, default -1
        Duration over which sampling should occur in units specified by
        `duration_units`. If duration is negative, consumers are expected
        to sample until the end of the video.
    subsample_interval : float, default 1
        Desired interval between successive sampled frames. When not using
        frame-based units, this spacing may be approximate.
    start_units : {"time_s", "pts", "frame"}, default "pts"
        The units used to interpret the `start` value.
        Choosing ``"frame"`` emphasizes exact frame-index semantics, but may require
        decoding/counting forward to reach the start position and can therefore be
        slower for large offsets. For performance-sensitive cases, prefer
        ``"time_s"`` or ``"pts"`` start units when possible.
    duration_units : {"time_s", "pts", "frame"}, default "pts"
        The units used to interpret the `duration` value.
    subsample_interval_units : {"time_s", "pts", "frame"}, default "pts"
        The units used to interpret the `subsample_interval` value.
    """

    start: float = 0
    duration: float = -1
    subsample_interval: float = 1
    start_units: Literal["time_s", "pts", "frame"] = "pts"
    duration_units: Literal["time_s", "pts", "frame"] = "pts"
    subsample_interval_units: Literal["time_s", "pts", "frame"] = "pts"


class BackendAdapter(Protocol):
    """A protocol defining the contract for a video decoding backend."""

    def decode(self, spec: SampleSpec, video_filepath: Path) -> Sequence[VideoFrame]:
        """Decodes frames from a video file into memory as a sequence.

        Parameters
        ----------
        spec : SampleSpec
            The specification describing how to sample the video.
        video_filepath : pathlib.Path
            The path to the video file.

        Returns
        -------
        Sequence[VideoFrame]
            A sequence (e.g., a list) of the decoded video frames.
        """
        ...

    def decode_iter(
        self,
        spec: SampleSpec,
        video_filepath: Path,
    ) -> Generator[VideoFrame, None, None]:
        """Lazily decodes frames from a video file using a generator.

        Parameters
        ----------
        spec : SampleSpec
            The specification describing how to sample the video.
        video_filepath : pathlib.Path
            The path to the video file.

        Yields
        ------
        VideoFrame
            The next decoded video frame that matches the sampling criteria.
        """
        ...

    def probe(self, video_filepath: Path) -> DatumMetadata:
        """Retrieves metadata from a video file without decoding frames.

        Parameters
        ----------
        video_filepath : pathlib.Path
            The path to the video file.

        Returns
        -------
        DatumMetadata
            A dictionary containing metadata about the video stream, such as
            resolution, time base, and file size.
        """
        ...


TDet_in = TypeVar("TDet_in", contravariant=True)
TDet_out = TypeVar("TDet_out", covariant=True)


class TrackerManager(Protocol[TDet_in, TDet_out]):
    """A minimal generic protocol for a stateful tracker in a tracking-by-detection pipeline.
    The types expected for detections (both an input and output of update method) are
    the generic variables.

    Notes
    -----
    Implementations must maintain independent tracker state per `video_id` and
    can assume that frames for a given `video_id` arrive in sequential display
    order.
    """

    def update(
        self,
        video_id: int,
        dets: TDet_in,
        t_sec: float | None = None,
        frame_index: int | None = None,
        image_size: tuple[int, int] | None = None,
        embeddings: ArrayLike | None = None,
    ) -> TDet_out:
        """Consumes detections for the next frame and returns annotated tracks.

        Parameters
        ----------
        video_id : int
            A unique identifier for the video stream being processed.
        dets : TDet_in
            The detections object for the current frame.
        t_sec : Optional[float], default None
            The timestamp of the frame in seconds.
        frame_index : Optional[int], default None
            The sequential index of the current frame.
        image_size : Optional[tuple[int, int]], default None
            The (height, width) of the frame.
        embeddings : Optional[ArrayLike], default None
            Associated embeddings for the detections.

        Returns
        -------
        TDet_out
            A detections object annotated with track IDs.
        """
        ...

    def reset(self, video_id: int) -> None:
        """Drops the tracking state for a single video stream.

        This should be called at the boundary between videos to clear memory.

        Parameters
        ----------
        video_id : int
            The identifier for the video stream whose state should be reset.
        """
        ...


@dataclass
class _VideoFrameImpl:
    """A container for a single decoded video frame and its metadata.

    Attributes
    ----------
    pixels : ArrayLike
        The raw pixel data of the frame as a NumPy array.
    time_s : float
        The presentation timestamp of the frame in seconds.
    pts : int
        The presentation timestamp in the stream's native time_base units.
    frame_index : int
        Zero-based index of the frame in the yielded decode/sampling sequence.
    """

    pixels: ArrayLike
    time_s: float | None
    pts: int | None
    frame_index: int


@dataclass(frozen=True)
class _SamplingPlan:
    """Compiled sampling plan with explicit, single-purpose fields."""

    start_mode: Literal["frame", "pts"]
    start_frame: int
    start_pts: int
    duration_mode: Literal["frame", "pts", "unbounded"]
    duration_frames: int
    duration_pts: int
    subsample_mode: Literal["frame", "pts"]
    subsample_interval_frames: int
    subsample_interval_pts: int


class PyAVAdapter:
    """A concrete implementation of the BackendAdapter protocol using PyAV."""

    def __init__(self, av_module: Any | None = None) -> None:  # noqa: ANN401, passed type should to emulate 'open' method of av module
        if av_module is not None:
            self._av = av_module
            return

        if not is_av_available():
            raise ImportError(
                "PyAVAdapter requires optional dependency 'av'. Install with: pip install maite[mot-utils]",
            )

        self._av = importlib.import_module("av")

    @staticmethod
    def _to_pts(
        value: float,
        unit: Literal["pts", "time_s", "frame"],
        stream: avVideoStream,
    ) -> int:
        """Converts a value from a given unit into PTS (Presentation Time Stamp).

        Parameters
        ----------
        value : float
            The numeric value to be converted.
        unit : {"pts", "time_s", "frame"}
            The unit of the input `value`.
        stream : av.video.stream.VideoStream
            The PyAV video stream object.
        avoid_estimates : bool, default True
            If True, raises an error when attempting to convert from 'frame' units.

        Returns
        -------
        int
            The converted value in PTS units.

        Raises
        ------
        ValueError
            If conversion is not possible due to missing stream metadata or if
            `avoid_estimates` is True for a frame-based conversion.
        """
        if value == -1:
            return -1

        if unit == "pts":
            return int(value)
        if unit == "time_s":
            if stream.time_base is None:
                raise ValueError(
                    "Cannot convert from 'time_s' without a valid time_base.",
                )
            return int(value / stream.time_base)
        raise ValueError(f"Unknown unit: {unit}")

    @staticmethod
    def _validate_spec(spec: SampleSpec) -> None:
        """Validate basic numeric constraints for a sampling specification."""
        if spec.start < 0:
            raise ValueError("SampleSpec.start must be non-negative.")
        if spec.subsample_interval <= 0:
            raise ValueError("SampleSpec.subsample_interval must be > 0.")

    def _compile_sampling_plan(
        self,
        spec: SampleSpec,
        stream: avVideoStream,
    ) -> _SamplingPlan:
        """Convert a user-facing SampleSpec into a normalized internal plan."""
        self._validate_spec(spec)

        if spec.start_units == "frame":
            start_mode: Literal["frame", "pts"] = "frame"
            start_frame = int(spec.start)
            start_pts = -1
        else:
            start_mode = "pts"
            start_frame = -1
            start_pts = self._to_pts(spec.start, spec.start_units, stream)

        if spec.duration < 0:
            duration_mode: Literal["frame", "pts", "unbounded"] = "unbounded"
            duration_frames = -1
            duration_pts = -1
        elif spec.duration_units == "frame":
            duration_mode = "frame"
            duration_frames = int(spec.duration)
            duration_pts = -1
        else:
            duration_mode = "pts"
            duration_frames = -1
            duration_pts = self._to_pts(spec.duration, spec.duration_units, stream)

        if spec.subsample_interval_units == "frame":
            subsample_mode: Literal["frame", "pts"] = "frame"
            subsample_interval_frames = int(spec.subsample_interval)
            subsample_interval_pts = -1
            if subsample_interval_frames <= 0:
                raise ValueError(
                    "Frame-based subsample_interval must convert to a positive integer.",
                )
        else:
            subsample_mode = "pts"
            subsample_interval_frames = -1
            subsample_interval_pts = self._to_pts(
                spec.subsample_interval,
                spec.subsample_interval_units,
                stream,
            )
            if subsample_interval_pts <= 0:
                raise ValueError(
                    "PTS-based subsample_interval must convert to a positive integer PTS value.",
                )

        return _SamplingPlan(
            start_mode=start_mode,
            start_frame=start_frame,
            start_pts=start_pts,
            duration_mode=duration_mode,
            duration_frames=duration_frames,
            duration_pts=duration_pts,
            subsample_mode=subsample_mode,
            subsample_interval_frames=subsample_interval_frames,
            subsample_interval_pts=subsample_interval_pts,
        )

    @staticmethod
    def _is_at_or_past_start(
        frame: avVideoFrame,
        source_frame_index: int,
        plan: _SamplingPlan,
        anchor_frame_pts: float | None,
    ) -> bool:
        """Return True once decoding reaches the plan's start condition."""
        if plan.start_mode == "frame":
            return source_frame_index >= plan.start_frame

        if anchor_frame_pts is None:
            raise ValueError("anchor_frame_pts is None")
        if frame.pts is None:
            raise ValueError("frame.pts is None")
        return (frame.pts - anchor_frame_pts) >= plan.start_pts

    @classmethod
    def _get_to_first_frame(
        cls,
        container: InputContainer,
        stream: avVideoStream,
        plan: _SamplingPlan,
        anchor_frame_pts: float | None,
    ) -> Iterator[avVideoFrame] | None:
        """Create and advance decoded frame iterator to first frame that satisfies start.

        Returns both the first matching frame (if any) and the same decoded-frame iterator,
        now partially consumed up to and including that first frame.

        Notes
        -----
        For ``plan.start_mode == "frame"``, this may decode and count many frames before
        finding the anchor. This preserves exact frame-index semantics, but can be slower
        for large starting offsets.
        """
        decoded_frames = container.decode(stream)

        for source_frame_index, frame in enumerate(decoded_frames):
            if cls._is_at_or_past_start(
                frame,
                source_frame_index,
                plan,
                anchor_frame_pts,
            ):
                return chain([frame], decoded_frames)
        return None

    @staticmethod
    def _is_beyond_duration(
        frame: avVideoFrame,
        decoded_count: int,
        first_frame_pts: int | None,
        plan: _SamplingPlan,
    ) -> bool:
        """Return True when decoding should stop because duration is exceeded."""
        if plan.duration_mode == "unbounded":
            return False

        if plan.duration_mode == "frame":
            return decoded_count >= plan.duration_frames

        if frame.pts is None or first_frame_pts is None:
            raise ValueError(
                "PTS-based duration sampling requires valid frame.pts values for all frames in the sampled region.",
            )

        return (frame.pts - first_frame_pts) >= plan.duration_pts

    @staticmethod
    def _is_this_frame_included_in_subsampling(
        frame: avVideoFrame,
        decoded_count: int,
        anchor_pts: int | None,
        next_pts_to_include: int,
        plan: _SamplingPlan,
    ) -> tuple[bool, int]:
        """Determine if current frame should be yielded according to subsampling plan."""
        if plan.subsample_mode == "frame":
            include = (decoded_count % plan.subsample_interval_frames) == 0
            return include, next_pts_to_include

        if frame.pts is None or anchor_pts is None:
            raise ValueError(
                "PTS-based subsampling requires valid frame.pts values for all frames in the sampled region.",
            )

        rel_pts = frame.pts - anchor_pts
        if rel_pts < next_pts_to_include:
            return False, next_pts_to_include

        next_pts_to_include += plan.subsample_interval_pts
        if next_pts_to_include <= rel_pts:
            next_pts_to_include += (
                (rel_pts - next_pts_to_include) // plan.subsample_interval_pts + 1
            ) * plan.subsample_interval_pts

        return True, next_pts_to_include

    @staticmethod
    def _to_video_frame(
        frame: avVideoFrame,
        yielded_frame_index: int,
        anchor_pts: int | None,
    ) -> VideoFrame:
        """Convert a decoded PyAV frame into MAITE's VideoFrame implementation."""
        reported_pts = frame.pts - anchor_pts if anchor_pts is not None and frame.pts is not None else None

        reported_time = (
            float(reported_pts * frame.time_base) if reported_pts is not None and frame.time_base is not None else None
        )

        # Note: to_ndarray will disregard a 'channels_last' argument except when
        # particular formats are used. We thus separately transpose data after using numpy.
        return _VideoFrameImpl(
            pixels=np.transpose(frame.to_ndarray(format="rgb24"), axes=[2, 0, 1]),
            time_s=reported_time,
            pts=reported_pts,
            frame_index=yielded_frame_index,
        )

    def decode_iter(
        self,
        spec: SampleSpec,
        video_filepath: Path,
    ) -> Generator[VideoFrame, None, None]:
        """Lazily decodes frames from a video file according to a spec.

        This method compiles the sampling specification into a plan, advances to
        the first included frame, and then applies duration/subsampling checks in
        a single straightforward decode loop.

        Performance notes
        -----------------
        When ``spec.start_units == "frame"``, this method may need to decode/count
        frames up to the requested start index. For large offsets or
        performance-sensitive workloads, prefer ``"pts"`` or ``"time_s"`` starts,
        or precompute/cache a frame-index-to-PTS mapping externally and then query
        via PTS-based starts.

        Parameters
        ----------
        spec : SampleSpec
            The specification describing how to sample the video.
        video_filepath : pathlib.Path
            The path to the video file.

        Yields
        ------
        VideoFrame
            The next decoded video frame that matches the sampling criteria.
        """
        with self._av.open(str(video_filepath)) as container:
            stream = container.streams.video[0]
            plan = self._compile_sampling_plan(spec, stream)
            requires_pts = plan.start_mode == "pts" or plan.duration_mode == "pts" or plan.subsample_mode == "pts"

            try:
                anchor_frame: avVideoFrame = next(container.decode(stream))
            except StopIteration:
                return

            if requires_pts and anchor_frame.pts is None:
                raise ValueError("Time-based SampleSpec used, but no PTS found.")

            if plan.start_mode == "pts" and plan.start_pts > 0:
                if anchor_frame.pts is None:
                    raise ValueError("anchor_frame.pts is None")
                container.seek(
                    plan.start_pts + anchor_frame.pts,
                    backward=True,
                    stream=stream,
                )
            else:
                container.seek(anchor_frame.pts, backward=True, stream=stream)

            decoded_frames = self._get_to_first_frame(
                container,
                stream,
                plan,
                anchor_frame.pts,
            )
            if decoded_frames is None:
                return

            yielded_count = 0
            next_pts_to_include = 0
            first_frame_pts: float | None = None

            for decoded_count, frame in enumerate(decoded_frames):
                if decoded_count == 0:
                    first_frame_pts = frame.pts
                    if first_frame_pts is not None and anchor_frame.pts is not None:
                        next_pts_to_include = first_frame_pts - anchor_frame.pts
                if self._is_beyond_duration(
                    frame,
                    decoded_count,
                    first_frame_pts,
                    plan,
                ):
                    break

                include, next_pts_to_include = self._is_this_frame_included_in_subsampling(
                    frame=frame,
                    decoded_count=decoded_count,
                    anchor_pts=anchor_frame.pts,
                    next_pts_to_include=next_pts_to_include,
                    plan=plan,
                )

                if not include:
                    continue

                yield self._to_video_frame(frame, yielded_count, anchor_frame.pts)
                yielded_count += 1

    def decode(self, spec: SampleSpec, video_filepath: Path) -> Sequence[VideoFrame]:
        """Decodes frames from a video file into a list in memory.

        Parameters
        ----------
        spec : SampleSpec
            The specification describing how to sample the video.
        video_filepath : pathlib.Path
            The path to the video file.

        Returns
        -------
        Sequence[VideoFrame]
            A list of the decoded video frames.
        """
        return list(self.decode_iter(spec, video_filepath))

    def probe(self, video_filepath: Path) -> DatumMetadata:
        """Retrieves metadata from a video file without decoding frames.

        Parameters
        ----------
        video_filepath : pathlib.Path
            The path to the video file.

        Returns
        -------
        DatumMetadata
            A dictionary containing metadata about the video stream.

        Raises
        ------
        ValueError
            If no video stream can be found in the file.
        """
        with self._av.open(str(video_filepath)) as container:
            try:
                stream = container.streams.video[0]
            except IndexError as err:
                raise ValueError(
                    f'PyAV cannot find video stream in: "{video_filepath}"',
                ) from err

            return {
                "id": str(video_filepath.name),
                "height": stream.height,
                "width": stream.width,
                "time_base": stream.time_base if stream.time_base is not None else Fraction(-1, 1),
                "size": video_filepath.stat().st_size,
            }
