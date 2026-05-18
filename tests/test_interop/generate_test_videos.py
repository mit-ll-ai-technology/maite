import json
from fractions import Fraction

import av
import numpy as np

# Frame and cell dimensions
FRAME_WIDTH = 64
FRAME_HEIGHT = 48
CELL_WIDTH = 8
CELL_HEIGHT = 8
GRID_COLS = FRAME_WIDTH // CELL_WIDTH  # 8
GRID_ROWS = FRAME_HEIGHT // CELL_HEIGHT  # 6
TOTAL_CELLS = GRID_COLS * GRID_ROWS  # 48

# Anchor positions (col, row) - only in channel 2 (blue)
ANCHOR_POSITIONS = {(0, 0), (7, 0), (0, 5), (7, 5)}

# Use channels 0-1 (red, green) for data
DATA_CHANNELS = 2
DATA_BITS = TOTAL_CELLS * DATA_CHANNELS  # 48 × 2 = 96 bits


def generate_block_coded_frame(frame_id: int) -> np.ndarray:
    # Create empty frame
    frame = np.empty((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    # Convert hex signature to bits directly during cell filling
    bit_index = 0
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            y_start = row * CELL_HEIGHT
            y_end = y_start + CELL_HEIGHT
            x_start = col * CELL_WIDTH
            x_end = x_start + CELL_WIDTH

            # Channels 0-1: encode data bits from hex signature
            for channel in range(DATA_CHANNELS):
                value = 255 if (frame_id >> bit_index) & 1 else 0
                frame[y_start:y_end, x_start:x_end, channel] = value
                bit_index += 1

            # Channel 2: anchors at corners, black elsewhere
            value = 255 if (col, row) in ANCHOR_POSITIONS else 0
            frame[y_start:y_end, x_start:x_end, 2] = value

    return frame


def decode_block_coded_frame(frame: np.ndarray) -> int:
    # Ensure correct format
    if frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
        raise ValueError(
            f"Expected frame shape ({FRAME_HEIGHT}, {FRAME_WIDTH}, 3), got {frame.shape}"
        )

    frame_id = 0
    bit_index = 0
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            # Average over 4×4 central region
            y_start = row * CELL_HEIGHT + CELL_HEIGHT // 4
            y_end = row * CELL_HEIGHT + CELL_HEIGHT * 3 // 4
            x_start = col * CELL_WIDTH + CELL_WIDTH // 4
            x_end = col * CELL_WIDTH + CELL_WIDTH * 3 // 4

            # Extract bits from channels 0-1
            for channel in range(DATA_CHANNELS):
                value = frame[y_start:y_end, x_start:x_end, channel].mean()
                if value > 127:
                    frame_id |= 1 << bit_index
                bit_index += 1

            # Check anchor value
            value = frame[y_start:y_end, x_start:x_end, 2].mean()
            if (value > 127) != ((col, row) in ANCHOR_POSITIONS):
                raise ValueError(f"Anchor check failed at row={row}, col={col}")

    return frame_id


def main():
    for i in range(2):
        dst_filename = f"test_video_{i + 1}.mp4"
        out_container = av.open(dst_filename, mode="w")
        codec = "mpeg2video" if i == 0 else "h264"
        out_container.add_stream(codec, rate=Fraction(30000, 1001))
        out_video_stream = out_container.streams.video[0]
        out_video_stream.width = FRAME_WIDTH
        out_video_stream.height = FRAME_HEIGHT
        out_video_stream.time_base = Fraction(1, 30000)
        if i == 1:
            out_video_stream.codec_context.max_b_frames = 0
            out_video_stream.codec_context.gop_size = 12
        out_container.options["avoid_negative_ts"] = "0"
        out_container.start_encoding()
        packet_index = 0
        for j in range(150):
            frame_image = generate_block_coded_frame(j)
            frame = av.VideoFrame.from_ndarray(frame_image)
            frame.time_base = out_video_stream.time_base
            frame.pts = 1001 * j
            if i == 0:
                assert frame.pts is not None
                frame.pts += 10 * j * j
            else:
                assert frame.pts is not None
                frame.pts -= 2 * j

            packets = out_video_stream.codec_context.encode(frame)
            for packet in packets:
                assert packet.dts is not None
                assert packet.pts is not None
                assert packet.time_base is not None
                assert out_video_stream.time_base is not None
                packet.pts = int(
                    packet.pts * packet.time_base / out_video_stream.time_base
                )
                packet.dts = int(
                    packet.dts * packet.time_base / out_video_stream.time_base
                )
                if i == 0:
                    ts_offset = -10000
                else:
                    if packet_index == 0 or packet_index == 100:
                        packet_index += 1
                        continue
                    ts_offset = 10000 + (100000 if packet_index > 100 else 0)
                packet.dts += ts_offset
                packet.pts += ts_offset
                packet.time_base = out_video_stream.time_base

                out_container.mux(packet)
                packet_index += 1
        out_container.close()

        in_container = av.open(dst_filename)
        first_pts = None
        metadata = []
        for j, frame in enumerate(in_container.decode(video=0)):
            assert frame.pts is not None
            assert frame.time_base is not None
            if first_pts is None:
                first_pts = frame.pts
            image = frame.to_ndarray(format="rgb24")
            decoded_frame = decode_block_coded_frame(image)
            pts = frame.pts - first_pts
            timestamp = float(pts * frame.time_base)
            metadata.append(
                {
                    "frame": j,
                    "decoded_frame": decoded_frame,
                    "pts": pts,
                    "time_s": timestamp,
                }
            )

        with open(f"test_video_{i + 1}.json", "w") as out_json:
            json.dump(metadata, out_json, indent=2)


if __name__ == "__main__":
    main()
