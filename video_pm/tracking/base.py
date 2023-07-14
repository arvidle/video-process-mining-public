# Define Tracker
# Define TrackingTrace
import pandas as pd
from video_pm import Video
import numpy as np
import math
import cv2
import queue

from typing import Type, Tuple, Generator, List
from typing_extensions import Self


ROLLING_WINDOW_SIZE = 5
ISOLATION_MARGIN = 20
DIST_THRESHOLD = 200
DFRAMES_THRESHOLD = 5000
MIN_TRACE_LENGTH = 50


class TrackingResults:
    """Container class for the results of a video object tracker (Tracker)

    The results of a Tracker are saved in a Pandas Dataframe wrapped by this class.
    Also provides methods for (compressed) serialization and loading of the results.
    """
    def __init__(self, tracking: pd.DataFrame):
        """Initialize the TrackingResults from a Pandas DataFrame.

        :param tracking: Pandas DataFrame containing the tracking information
        """
        if not tracking.empty:
            tracking_df = tracking.sort_values(["frame", "track_id"])
        else:
            tracking_df = pd.DataFrame(columns=["frame", "track_id", "x1", "y1", "x2", "y2", "score", "det_class"])

        self.tracking: pd.DataFrame = tracking_df

    def __len__(self):
        return len(self.tracking)

    def __str__(self):
        if len(self) > 0:
            res = str(self.tracking)
        else:
            res = "Empty tracking results"

        return res

    def to_file(self, filename: str, normalize: bool = False, normalize_wh: Tuple[int, int] = None) -> None:
        """Serialize the tracking results to a compressed file.

        :param filename: Name of the file to save the tracking results to
        :return: None
        """
        boxes = self.tracking[["frame", "track_id", "x1", "y1", "x2", "y2"]].astype(int)
        if normalize:
            boxes["x1"] /= normalize_wh[0]
            boxes["x2"] /= normalize_wh[0]
            boxes["y1"] /= normalize_wh[1]
            boxes["y2"] /= normalize_wh[1]
        confs = self.tracking[["score"]]
        boxes_np = boxes.to_numpy()
        confs_np = confs.to_numpy()

        np.savez_compressed(filename, boxes=boxes_np, confs=confs_np)

    @classmethod
    def from_file(cls: Type[Self], filename: str) -> Self:
        """Load TrackingResults from a file.

        :param filename: Name of the file with the serialized tracking results
        :return: A TrackingResults instance containing the loaded tracking results
        """
        with np.load(filename) as data:
            boxes = data["boxes"]
            confs = data["confs"]

        columns = ["frame", "track_id", "x1", "y1", "x2", "y2"]
        tracking = pd.DataFrame(boxes, columns=columns)
        tracking["score"] = confs
        tracking["det_class"] = "pig"

        return cls(tracking)

    def to_mot_format(self) -> pd.DataFrame:
        # MOT16 format is a csv with columns: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        # In 2D, x, y, and z are ignored and can be filled with -1
        width = self.tracking.x2 - self.tracking.x1
        height = self.tracking.y2 - self.tracking.y1
        df_out = self.tracking[["frame", "track_id", "x1", "y1", "score"]].rename(columns={"track_id": "id", "x1": "bb_left", "y1": "bb_top", "score": "conf"}).copy()
        df_out["bb_width"] = width
        df_out["bb_height"] = height
        # Fill x, y and z with -1 as we are doing 2D here
        df_out["x"] = -1
        df_out["y"] = -1
        df_out["z"] = -1
        # Put the columns into correct order
        df_out = df_out[["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]]
        return df_out

class Tracker:
    """Base class for a video object tracker"""
    def __init__(self):
        pass

    def run(self, video: Video) -> TrackingResults:
        """Run object tracking on a Video.

        Do not use this function directly, but use an implementation provided by a subclass of this class.
        This only defines the interfaces used by concrete implementations.

        :param video: Video to run the object tracking on
        :return: Tracking results for the Video as an instance of TrackingResults
        """
        pass


def cut_box_from_frame(frame: np.ndarray,
                       box: Tuple[int, int, int, int],
                       grow_to: Tuple[int, int],
                       pad_color: Tuple[int, int, int],
                       margin: int,
                       mask: bool = True,
                       mask_color: Tuple[int, int, int] = (0, 0, 0)) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Cut the area marked by a bounding box from a video frame.

    The cut area is grown to a given size with the bounding box centered in the cut frame.
    If the box is at an edge of the image, padding is added to the image to keep the box centered.
    When masking is applied, the visible region will be the bounding box including the specified margin.

    :param frame: Image to cut the section from
    :param box: Box defining the cutting area
    :param grow_to: Output image dimensions
    :param pad_color: Color of the padding
    :param margin: Size of the margin around the box
    :param mask: Whether the pig should be masked via its bounding box
    :param mask_color: Color of the mask
    :return: The cut frame and the coordinates of the bounding box in the cut frame
    """
    # TODO: Rounding error. Ensure the exact size is reached for the resulting frame
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    grow_x = grow_to[0] - box_width + 2 * margin
    grow_y = grow_to[1] - box_height + 2 * margin

    grow_x1 = x1 - math.trunc(grow_x / 2)
    grow_x2 = x2 + math.ceil(grow_x / 2)
    grow_y1 = y1 - math.trunc(grow_y / 2)
    grow_y2 = y2 + math.ceil(grow_y / 2)

    # If the grown boxes are bigger than the original image, padding needs to be added
    # This helps to keep the isolated object centered in the resulting frame
    pad_width = max(0, -min(grow_x1, grow_y1, frame_width - grow_x2, frame_height - grow_y2))
    if pad_width > 0:
        frame = np.pad(frame,
                       pad_width=((pad_width, pad_width), (pad_width, pad_width), (0, 0)),
                       mode='constant',
                       constant_values=pad_color)
    # Add offset created by the padding to the indices
    grow_x1 += pad_width
    grow_x2 += pad_width
    grow_y1 += pad_width
    grow_y2 += pad_width

    # Calculate the coordinates of the bounding box inside the cut frame
    offset_x = math.trunc(grow_x / 2)
    offset_y = math.trunc(grow_y / 2)
    box_offset = [offset_x, offset_y, offset_x + box_width, offset_y + box_height]

    cut_frame = frame[grow_y1:grow_y2, grow_x1:grow_x2]

    assert grow_x2 - grow_x1 == grow_to[0] + 2 * margin
    assert grow_y2 - grow_y1 == grow_to[1] + 2 * margin

    if mask:
        masked_image = np.zeros_like(cut_frame)
        masked_image[:] = mask_color
        box_offset_margin = [box_offset[0] - margin, box_offset[1] - margin, box_offset[2] + margin, box_offset[3] + margin]
        masked_image[box_offset_margin[1]:box_offset_margin[3], box_offset_margin[0]:box_offset_margin[2]] = \
            cut_frame[box_offset_margin[1]:box_offset_margin[3], box_offset_margin[0]:box_offset_margin[2]]

        cut_frame = masked_image

    return cut_frame, box_offset


class IsolatedVideo:
    """Container class for isolated video sequences

    Isolated video sequences are based on a base video and a tracking trace.
    The isolated video is constructed by cutting the image areas and frames contained by a tracking trace from the
    base video.
    As holding a complete video (and/or many isolated sequences) may be very expensive memory-wise,
    the isolated video is provided as a generator yielding the frames in sequence.
    """
    def __init__(self, video: Video,
                 tracking_trace: pd.DataFrame,
                 padding_color: int = 80,
                 margin_size: int = ISOLATION_MARGIN):
        """Create a new instance of the IsolatedVideo class.

        :param video: Source video to isolate a video sequence from
        :param tracking_trace: Tracking trace DataFrame to be isolated from the source video
        """
        self.video: Video = video

        # Assume that there are no gaps (i.e. missing frames) in the tracking trace
        assert (len(tracking_trace) ==
                tracking_trace["frame"].max() - tracking_trace["frame"].min() + 1), \
            "Trace does not have the correct number of frames"
        self.tracking_trace: pd.DataFrame = tracking_trace.set_index("frame")
        self.tracking_trace["width"] = self.tracking_trace["x2"] - self.tracking_trace["x1"]
        self.tracking_trace["height"] = self.tracking_trace["y2"] - self.tracking_trace["y1"]

        self.max_width: int = self.tracking_trace["width"].max()
        self.max_height: int = self.tracking_trace["height"].max()

        self.padding_color: Tuple[int, int, int] = padding_color
        self.margin_size: int = margin_size

        self.width: int = self.max_width + 2 * self.margin_size
        self.height: int = self.max_height + 2 * self.margin_size

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Generator for iterating over the frames of the IsolatedVideo

        :return: A generator over the frames of the IsolatedVideo
        """
        def box(row: pd.Series) -> Tuple[int, int, int, int]:
            return row["x1"], row["y1"], row["x2"], row["y2"]

        video_capture = self.video.get_capture()
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.tracking_trace.index[0])

        max_frame = self.tracking_trace.index.max()
        success, frame = video_capture.read()
        frame_num = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        while success and frame_num <= max_frame:
            row = self.tracking_trace.loc[frame_num]
            cut_frame, box_offset = cut_box_from_frame(frame,
                                                          box(row),
                                                          (self.max_width, self.max_height),
                                                          self.padding_color,
                                                          self.margin_size)
            yield cut_frame
            success, frame = video_capture.read()
            frame_num += 1


class VideoIsolator:
    """Isolates video sequences from a source Video according to the provided TrackingResults.

    Provides a collection of IsolatedVideos corresponding to the tracking traces found in the TrackingResults.
    """
    def __init__(self, video: Video, tracking_results: TrackingResults, moving_average_window: int = 5, group_on: str = "frame"):
        """Init VideoIsolator with a Video, TrackingResults and a window size for moving average smoothing

        :param video: Source video
        :param tracking_results: TrackingResults containing the tracking traces to isolate
        :param moving_average_window: Window size for rolling average smoothing of the traces
        """
        self.video: Video = video
        self.tracking_results: TrackingResults = tracking_results
        self.moving_average_window: int = moving_average_window
        self.group_on: str = group_on
        self.len = len(self.tracking_results.tracking[self.group_on].unique())

    def __len__(self):
        return self.len

    def run(self) -> Generator[IsolatedVideo, None, None]:
        """Create a generator providing an IsolatedVideo for each trace.

        :return: Generator yielding an IsolatedVideo for each trace
        """

        # Group results by specified ID
        traces = self.tracking_results.tracking.groupby(self.group_on)
        # For each tracking id
        for track_id, trace in traces:
            yield IsolatedVideo(self.video, trace)


def split_trace(trace: pd.DataFrame, dist_threshold: float, dframes_threshold: int, trace_id_start: int) -> Tuple[int, pd.DataFrame]:
    # Calculate midpoints
    midpoints = trace[["x1", "y1", "x2", "y2"]].apply(lambda r: np.array([(r["x1"] + r["x2"]) / 2, (r["y1"] + r["y2"]) / 2]), axis=1)
    # Calculate distances
    df_midpoints = pd.DataFrame()
    df_midpoints["p1"] = midpoints
    df_midpoints["p2"] = midpoints.shift()
    df_midpoints["p2"].iloc[0] = df_midpoints["p1"].iloc[0]
    df_midpoints["dist"] = df_midpoints.apply(lambda r: np.linalg.norm(r["p2"] - r["p1"]), axis=1)
    # Calculate frame distances
    df_frames = pd.DataFrame()
    df_frames["frame1"] = trace["frame"]
    df_frames["frame2"] = trace["frame"].shift()
    df_frames["dframes"] = df_frames.apply(lambda r: r["frame1"] - r["frame2"], axis=1)
    # If the maximum distance or the maximum frame difference is exceeded, split the trace
    # Get indices of the split points and join
    distance_splits = df_midpoints.reset_index().index[df_midpoints["dist"] >= dist_threshold]
    dframes_splits = df_frames.reset_index().index[df_frames["dframes"] >= dframes_threshold]
    split_indices = trace[:1].reset_index().index.join(distance_splits.join(dframes_splits, how="outer").to_list(), how="outer").join(trace[-1:].index, how="outer")

    # Create list slices for all the splits and set a unique trace ID for each split
    split_slices = zip(split_indices, split_indices[1:])

    trace_ids = np.zeros(len(trace), dtype=int)
    for trace_id, (i, j) in enumerate(split_slices, trace_id_start):
        trace_ids[i:j] = trace_id

    return trace_ids.max(), trace_ids


class TrackingResultsPostProcessor:
    def __init__(self, tracking_results: TrackingResults,
                 dist_threshold: float = DIST_THRESHOLD,
                 dframes_threshold: int = DFRAMES_THRESHOLD,
                 moving_average_window: int = ROLLING_WINDOW_SIZE,
                 min_trace_length: int = MIN_TRACE_LENGTH):
        self.tracking_results: TrackingResults = tracking_results
        self.dist_threshold: float = dist_threshold
        self.dframes_threshold: int = dframes_threshold
        self.moving_average_window: int = moving_average_window
        self.min_trace_length: int = min_trace_length if min_trace_length > moving_average_window else moving_average_window

    def run(self):
        def rolling_mean(series: pd.Series, window_size: int = self.moving_average_window):
            # If the trace is shorter than the rolling window size, the rolling mean will result in NaN values
            # This shouldn't happen (traces shorter than the window size should probably be filtered earlier),
            # but if a very short trace appears here, just copy over the non-smoothed trace information to avoid NaNs.
            # TODO: This could probably be fixed by using the min_periods parameter of pd.Series.rolling
            if len(series) >= window_size:
                res = series.rolling(window_size).mean().fillna(method="bfill")
            else:
                res = series

            return res

        # Group tracking results by tracking id
        traces = self.tracking_results.tracking.groupby("track_id")

        # Manage traces to post-process in a Queue
        trace_queue = queue.Queue()

        #last_trace_id = self.tracking_results.tracking["track_id"].unique().max()
        last_trace_id = 0

        res_list = []

        # Put all available traces in the queue. If a trace is split, the split parts will be added to the queue
        for _, trace in traces:
            # Split trace if necessary and add trace ID
            last_trace_id, trace_ids = split_trace(trace, self.dist_threshold, self.dframes_threshold,
                                                   last_trace_id + 1)
            trace["trace_id"] = trace_ids
            if len(trace["trace_id"].unique()) > 1:
                for trace_id, subtrace in trace.groupby("trace_id"):
                    if len(subtrace) >= self.min_trace_length:
                        trace_queue.put(subtrace)
                continue

            trace_queue.put(trace)


        while not trace_queue.empty():
            trace = trace_queue.get()
            # Interpolate trace if frames are missing:
            # Create a list of all frames that should be in the trace and right join it with the trace
            # This gives a DataFrame with NaN values where values need to be interpolated
            trace_frames = pd.Series(np.arange(trace["frame"].min(), trace["frame"].max() + 1), name="frame")
            interpolated_trace = trace.merge(trace_frames, how="right").interpolate()

            # Set the det_class for all rows to the only class we have
            # This needs to be changed when working with multiple object classes
            interpolated_trace["det_class"] = trace.iloc[0]["det_class"]

            #   Apply rolling mean smoothing to the trace
            interpolated_trace["x1"] = rolling_mean(interpolated_trace["x1"])
            interpolated_trace["x2"] = rolling_mean(interpolated_trace["x2"])
            interpolated_trace["y1"] = rolling_mean(interpolated_trace["y1"])
            interpolated_trace["y2"] = rolling_mean(interpolated_trace["y2"])

            # Cast the bounding box values back to int
            interpolated_trace["x1"] = interpolated_trace["x1"].astype("int")
            interpolated_trace["x2"] = interpolated_trace["x2"].astype("int")
            interpolated_trace["y1"] = interpolated_trace["y1"].astype("int")
            interpolated_trace["y2"] = interpolated_trace["y2"].astype("int")

            interpolated_trace["track_id"] = interpolated_trace["trace_id"].astype(int)
            interpolated_trace = interpolated_trace.drop(['trace_id'], axis=1)

            if len(interpolated_trace) >= self.min_trace_length:
                res_list.append(interpolated_trace)

        return TrackingResults(pd.concat(res_list))
