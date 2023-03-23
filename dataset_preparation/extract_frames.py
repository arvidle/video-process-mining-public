import os
import shutil
import sys
from pathlib import Path

SRC = "./frames"
DST = "./choose_frames"

src_path = Path(SRC)
dst_path = Path(DST)

video_dirs = src_path.glob("*")

print(video_dirs)

for video in video_dirs:
    video_name = video.stem
    dst_frames = dst_path.joinpath(video_name)
    dst_frames.mkdir()
    frames = sorted(list(video.glob("*.jpg")))
    for frame in frames:
        name = frame.stem
        vid, frame_num = name.split("_")
        if (int(frame_num) - 1) % 30 == 0:
            frame_filename = frame.name
            shutil.copyfile(frame, dst_frames.joinpath(frame_filename))
            print("Copied ", frame_filename)
