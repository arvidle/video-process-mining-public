import json
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
import pickle
from collections import Counter

VIDEO_IDS = ["836-0113-0115", "836-0125-0126", "836-010745-011045", "931-0015-0016", "931-0035-0037", "946-0021-0025", "954-0000-0001"]
#VIDEO_ID = "931-0035-0037"
#VIDEO_FILENAME = VIDEO_ID + ".mp4"
#ANNOTATIONS_FILENAME = VIDEO_ID + ".json"

class_id = {"lying": 1, "sitting": 2, "standing": 3, "moving": 4, "investigating": 5, "feeding": 6, "defecating": 7, "playing": 8, "other": 9}
class_name =dict((v, k) for k, v in class_id.items())

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def get_labels(ann):
    behvs = []
    for obj in ann["objects"]:
        for frame in obj["frames"]:
            behvs.append(frame["behaviour"])

    return behvs

def draw_box_on_image(image, bbox, label=None, color=(200, 100, 0), thickness=1):
    img = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, thickness)
    if label:
        img = cv2.putText(img, label, (int(bbox[0]), int(bbox[1])+10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return img

def draw_annotation_on_image(image, annotation):
    x1 = annotation["x"]
    x2 = x1 + annotation["width"]
    y1 = annotation["y"]
    y2 = y1 + annotation["height"]
    label = annotation["behaviour"]

    res = draw_box_on_image(image, (x1, y1, x2, y2), label=label)

    return res


def get_object_annotations(object_data, video_len_frames=7200):
    frames = pd.DataFrame(object_data["frames"])
    #frames["frameNumber"] = frames["frameNumber"].apply(lambda x: ((x + 1) * 3) - 1).astype(int)
    frames["frameNumber"] = frames["frameNumber"].apply(lambda x: x * 30).astype(int)
    frames.set_index("frameNumber", inplace=True)
    boxes = pd.DataFrame(frames["bbox"].to_list(), index=frames.index)

    annotations = pd.concat([boxes, frames[["behaviour"]]], axis=1)
    trace_frames = pd.Series(np.arange(0, video_len_frames), name="frameNumber")

    interpolated_annotations = annotations.merge(trace_frames, on="frameNumber", how="right").interpolate().fillna(method="ffill").fillna(method="bfill")
    return object_data["id"], interpolated_annotations



def iterate_video(annotations, video, stepsize=100):
    with open(annotations, "r") as jf:
        data = json.load(jf)

    pigs_annotations = list(map(get_object_annotations, data["objects"]))

    capture = cv2.VideoCapture(video)
    video_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    success, frame = capture.read()

    while success:
        frame_annotations = []
        frame_num = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        image = frame
        for _, interpolated_annotations in pigs_annotations:
            try:
                #annotation = annotations.loc[frame_num]
                annotation = interpolated_annotations.loc[frame_num]
                frame_annotations.append(annotation)
            except:
                pass
        if frame_num % stepsize == 0:    
            yield frame, frame_annotations, video_width, video_height, frame_num
        success, frame = capture.read()


def get_json_paths(root):
    return[file for file in Path(root).glob("**/*.json")]

def convert_annotation_to_yolo(annotation, width, height):
    x1 = annotation["x"]
    y1 = annotation["y"]
    x2 = x1 + annotation["width"]
    y2 = y1 + annotation["height"]

    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2

    x = mx / float(width)
    y = my / float(height)
    w = annotation["width"] / float(width)
    h = annotation["height"] / float(height)
    
    return x, y, w, h


def xywh_to_yolo(x, y, w, h):
    return f"0 {x} {y} {w} {h}"

def process_video(annotations, video, video_id):
    """Extract a YOLO dataset from spatio-temporal labels"""
    frame_name = lambda frame_num: f"yolo/{video_id}_{frame_num}"
    #mask = cv2.imread("mask.png", 0)

    for frame, frame_annotations, width, height, frame_num in iterate_video(annotations, video, 50):
        image = frame
        convert_fn = lambda ann: xywh_to_yolo(*convert_annotation_to_yolo(ann, width, height))
        labels = "\n".join(map(convert_fn, frame_annotations))
        with open(frame_name(frame_num) + ".txt", "w") as file:
            file.write(labels)
        #for annotation in frame_annotations:
        #    image = draw_annotation_on_image(image, annotation)
        #    print(convert_annotation_to_yolo(annotation, width, height))
        #image = cv2.bitwise_and(image, image, mask=mask)
        cv2.imwrite(frame_name(frame_num) + ".jpg", image)
        #cv2.imshow("frame", image)
        #cv2.waitKey(0)

    #cv2.destroyAllWindows()

def show_video(annotations, video, num_frames=1800):
    mask = cv2.imread("mask.png", 0)

    for frame, frame_annotations, width, height, frame_num in iterate_video(annotations, video):
        image = frame
        for annotation in frame_annotations:
            image = draw_annotation_on_image(image, annotation)
        image = cv2.bitwise_and(image, image, mask=mask)
        print(frame_num)
        cv2.imshow("frame", frame)
        cv2.waitKey(30)

    cv2.destroyAllWindows()


def normalize_annotation(annotation, width, height):
    x1 = annotation["x"]
    y1 = annotation["y"]
    x2 = x1 + annotation["width"]
    y2 = y1 + annotation["height"]

    nx1 = x1 / float(width)
    nx2 = x2 / float(width)
    ny1 = y1 / float(height)
    ny2 = y2 / float(height)
    
    return nx1, ny1, nx2, ny2


def compile_video_annotations(annotations, video, normalize=True, num_frames=1800):
    capture = cv2.VideoCapture(video)
    video_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    capture.release()

    with open(annotations, "r") as jf:
        data = json.load(jf)

    objects_annotations = list(map(get_object_annotations, data["objects"]))

    annotations_per_frame = {}

    if normalize:
        norm_fn = lambda ann: normalize_annotation(ann, video_width, video_height)
    else:
        norm_fn = lambda ann: (ann["x"], ann["y"], ann["width"], ann["height"])

    for frame_num in range(num_frames):
        frame_annotations = []
        for object_id, interpolated_annotations in objects_annotations:
            try:
                annotation = interpolated_annotations.loc[frame_num]
                frame_annotations.append({"bbox": norm_fn(annotation), "activity": class_id[annotation["behaviour"]], "object_id": int(object_id)})
            except Exception as e:
                print(e)
        annotations_per_frame[frame_num] = frame_annotations

    return annotations_per_frame


def create_ava_labels(annotations, video, num_frames=1800):
    ann_per_frame = compile_video_annotations(annotations, video, num_frames=num_frames)
    fps = 30
    video_name = Path(video).stem
    video_id = video_name
    res = []
    for t, i in enumerate(range(0, num_frames, fps)):
        anns = ann_per_frame[i]
        anns.sort(key=lambda x: x["object_id"])
        for ann in anns:
            ann_str = f"{video_id},{t},{ann['bbox'][0]:.3f},{ann['bbox'][1]:.3f},{ann['bbox'][2]:.3f},{ann['bbox'][3]:.3f},{ann['activity']},{ann['object_id']}"
            res.append(ann_str)

    return res


"""
def create_coco_labels(annotations, video):
    # Get annotations per frame
    # Sample frames from the video at a specified interval
    # Save frame to file and keep annotation to save later
    ann_per_frame = compile_video_annotations(annotations, video)
    sample_n_frames = 20
    num_frames = 1800
    frame_nums = range(0, num_frames, num_frames // sample_n_frames)
    video_name = Path(video).stem
    video_id = video_name
    res = []
    for t, i in enumerate(range(0, num_frames, fps)):
        anns = ann_per_frame[i]
        anns.sort(key=lambda x: x["object_id"])
        for ann in anns:
            ann_str = f"{video_id},{t},{ann['bbox'][0]:.3f},{ann['bbox'][1]:.3f},{ann['bbox'][2]:.3f},{ann['bbox'][3]:.3f},{ann['activity']},{ann['object_id']}"
            res.append(ann_str)

    return res
"""


def create_dense_proposals(annotations, video):
    ann_per_frame = compile_video_annotations(annotations, video)
    fps = 6
    num_frames = 1800
    video_name = Path(video).stem
    video_id = video_name
    
    result = []

    for t, i in enumerate(range(0, num_frames, fps)):
        anns = ann_per_frame[i]
        temp_video_id = str(t).zfill(4)
        key = ','.join([video_name, temp_video_id])
        frame_boxes = []
        for ann in anns:
            frame_boxes.append([ann['bbox'][0], ann['bbox'][1], ann['bbox'][2], ann['bbox'][3], 0.99])
        result.append((key, np.vstack(frame_boxes)))

    return result


def dense_proposals_from_pathlist(paths):
    proposals_list = []
    for (annotation, video) in paths:
        video_proposals = create_dense_proposals(annotation.as_posix(), video.as_posix())
        proposals_list.append(video_proposals)

    return flatten_list(proposals_list)

def gen_and_save_dense_proposals(paths, split):
    proposals = dict(dense_proposals_from_pathlist(paths))
    with open(f"dense_proposals_{split}.pkl", "wb") as file:
        pickle.dump(proposals, file)


def gen_and_save_annotations(paths, split):
    all_labels = []
    for index, (ann, vid) in enumerate(paths):
        labels = create_ava_labels(ann.as_posix(), vid.as_posix())
        all_labels.append("\n".join(labels))

    with open(f"annotations_{split}.csv", "a") as file:
        file.write("\n".join(all_labels))


def gen_coco_annotations(paths, split):
    all_labels = []
    for index, (ann, vid) in enumerate(paths):
        labels = create_ava_labels(ann.as_posix(), vid.as_posix())
        all_labels.append("\n".join(labels))

    return "\n".join(all_labels)


def get_pathlist_split(split=""):
    json_paths = get_json_paths(".")
    #paths = [(jsonpath, jsonpath.parent.joinpath(jsonpath.stem + ".mp4")) for jsonpath in json_paths]
    paths = [(jsonpath, jsonpath.parent.parent.joinpath(VIDEO_LOCATION[jsonpath.stem])) for jsonpath in json_paths if split in VIDEO_LOCATION[jsonpath.stem]]
    paths.sort(key=lambda p: p[0].stem)
    
    return paths

"""
paths_list = get_pathlist_split()
for paths in paths_list:
    labels = compile_video_annotations(*map(lambda p: p.as_posix(), paths))
    label_list = [ann["activity"] for anns in labels.values() for ann in anns]

    c = Counter(label_list)
    cs = dict(c)
    cs_name = sorted({new_class_name[k]: v for k, v in cs.items()}, key=lambda x: x[0])
    print(paths)
    print(cs_name)
"""


#split = "train"
#paths = get_pathlist_split(split)
#gen_and_save_annotations(paths, split)
#gen_and_save_dense_proposals(paths, split)

for VIDEO_ID in VIDEO_IDS:
    VIDEO_FILENAME = VIDEO_ID + ".mp4"
    ANNOTATIONS_FILENAME = VIDEO_ID + ".json"
    process_video(Path(ANNOTATIONS_FILENAME).as_posix(), Path(VIDEO_FILENAME).as_posix(), Path(VIDEO_FILENAME).stem)
