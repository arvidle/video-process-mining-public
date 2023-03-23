# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy as cp
import os
import os.path as osp
import shutil

import cv2
import mmcv
import numpy as np
import pandas as pd
import torch
from mmcv import DictAction
from mmcv.runner import load_checkpoint

from collections import deque
from mmaction.models import build_detector

import pickle

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this demo! ')

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable visualization file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1


def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))


plate_blue = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
plate_blue = plate_blue.split('-')
plate_blue = [hex2color(h) for h in plate_blue]
plate_green = '004b23-006400-007200-008000-38b000-70e000'
plate_green = plate_green.split('-')
plate_green = [hex2color(h) for h in plate_green]


def visualize(frames, annotations, plate=plate_blue, max_num=5):
    """Visualize frames with predicted annotations.
    Args:
        frames (list[np.ndarray]): Frames for visualization, note that
            len(frames) % len(annotations) should be 0.
        annotations (list[list[tuple]]): The predicted results.
        plate (str): The plate used for visualization. Default: plate_blue.
        max_num (int): Max number of labels to visualize for a person box.
            Default: 5.
    Returns:
        list[np.ndarray]: Visualized frames.
    """

    assert max_num + 1 <= len(plate)
    plate = [x[::-1] for x in plate]
    frames_ = cp.deepcopy(frames)
    nf, na = len(frames), len(annotations)
    assert nf % na == 0
    nfpa = len(frames) // len(annotations)
    anno = None
    h, w, _ = frames[0].shape
    scale_ratio = np.array([w, h, w, h])
    for i in range(na):
        anno = annotations[i]
        if anno is None:
            continue
        for j in range(nfpa):
            ind = i * nfpa + j
            frame = frames_[ind]
            for ann in anno:
                box = ann[0]
                label = ann[1]
                if not len(label):
                    continue
                score = ann[2]
                box = (box * scale_ratio).astype(np.int64)
                st, ed = tuple(box[:2]), tuple(box[2:])
                cv2.rectangle(frame, st, ed, plate[0], 2)
                for k, lb in enumerate(label):
                    if k >= max_num:
                        break
                    text = abbrev(lb)
                    text = ': '.join([text, str(score[k])])
                    location = (0 + st[0], 18 + k * 18 + st[1])
                    textsize = cv2.getTextSize(text, FONTFACE, FONTSCALE,
                                               THICKNESS)[0]
                    textwidth = textsize[0]
                    diag0 = (location[0] + textwidth, location[1] - 14)
                    diag1 = (location[0], location[1] + 2)
                    cv2.rectangle(frame, diag0, diag1, plate[k + 1], -1)
                    cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                                FONTCOLOR, THICKNESS, LINETYPE)

    return frames_


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument(
        '--config',
        default=('configs/detection/ava/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py'),
        help='spatio temporal detection config file path')
    parser.add_argument(
        '--checkpoint',
        default=('https://download.openmmlab.com/mmaction/detection/ava/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb'
                 '_20201217-16378594.pth'),
        help='spatio temporal detection checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_2x_coco.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--action-score-thr',
        type=float,
        default=0.5,
        help='the threshold of human action score')
    parser.add_argument('--video', help='video file/url')
    parser.add_argument(
        '--label-map',
        default='tools/data/ava/label_map.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--out-filename',
        default='demo/stdet_demo.mp4',
        help='visualization filename')
    parser.add_argument(
        '--predict-stepsize',
        default=8,
        type=int,
        help='give out a prediction per n frames')
    parser.add_argument(
        '--visualization-stepsize',
        default=4,
        type=int,
        help=('show one frame per n frames in the demo, we should have: '
              'predict_stepsize % output_stepsize == 0'))
    parser.add_argument(
        '--visualization-fps',
        default=6,
        type=int,
        help='the fps of demo video visualization')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--tracking-filename',
        default='demo/tracking.npz',
        help='tracking info filename')
    args = parser.parse_args()
    return args


def frame_extraction(video_path):
    """Extract frames given video_path.
    Args:
        video_path (str): The video_path.
    """
    vid = cv2.VideoCapture(video_path)
    flag, frame = vid.read()

    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    return frame.shape, num_frames


def detection_inference(args, frame_paths):
    """Detect human boxes given frame paths.
    Args:
        args (argparse.Namespace): The arguments.
        frame_paths (list[str]): The paths of frames to do detection inference.
    Returns:
        list[np.ndarray]: The human detection results.
    """
    #model = init_detector(args.det_config, args.det_checkpoint, args.device)
    #assert model.CLASSES[0] == 'person', ('We require you to use a detector '
    #                                      'trained on COCO')
    with np.load(args.tracking_filename) as data:
        boxes = data["boxes"]
        confs = data["confs"]

    columns = ["frame", "track_id", "x1", "y1", "x2", "y2"]

    tracking = pd.DataFrame(boxes, columns=columns)
    tracking["score"] = confs
    tracking["det_class"] = 'person'

    frame_groups = tracking.groupby("frame")

    results = []
    print('Performing Human Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        #result = inference_detector(model, frame_path)
        frame_num = frame_path
        
        try:
            boxes_df = frame_groups.get_group(frame_num) 
            result = [boxes_df[["x1", "y1", "x2", "y2", "score"]].to_numpy(dtype=np.float32)]
            # We only keep human detections with score larger than det_score_thr
            result = result[0][result[0][:, 4] >= args.det_score_thr]
        except:
            result = results[-1]
        results.append(result)
        prog_bar.update()
    return results


def load_label_map(file_path):
    """Load Label Map.
    Args:
        file_path (str): The file path of label map.
    Returns:
        dict: The label map (int -> label name).
    """
    lines = open(file_path).readlines()
    lines = [x.strip().split(': ') for x in lines]
    return {int(x[0]): x[1] for x in lines}


def abbrev(name):
    """Get the abbreviation of label name:
    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find('(') != -1:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name


def pack_result(human_detection, result, img_h, img_w):
    """Short summary.
    Args:
        human_detection (np.ndarray): Human detection result.
        result (type): The predicted label of each human proposal.
        img_h (int): The image height.
        img_w (int): The image width.
    Returns:
        tuple: Tuple of human proposal, label name and label score.
    """
    human_detection[:, 0::2] /= img_w
    human_detection[:, 1::2] /= img_h
    results = []
    if result is None:
        return None
    for prop, res in zip(human_detection, result):
        res.sort(key=lambda x: -x[1])
        results.append(
            (prop.data.cpu().numpy(), [x[0] for x in res], [x[1]
                                                            for x in res]))
    return results


def frame_queue(window_width, frame_interval, predict_stepsize, video_path, new_w, new_h):
    vid = cv2.VideoCapture(video_path) 
    result_queue = deque()
    cnt = 1

    # Read the first frame to init the video and loop
    flag, frame = vid.read()

    # Fill the first frames window
    
    while flag and cnt <= window_width * frame_interval:
        if cnt % frame_interval == 0:
            resized_frame = mmcv.imresize(frame, (new_w, new_h))
            result_queue.append(resized_frame)
        cnt += 1
        flag, frame = vid.read()
    
    yield result_queue

    # Each iteration, shift the window by predict_stepsize
    # This is done by popping predict_stepsize frames and loading the same amount of new frames

    old_cnt = cnt
    
    while flag:
        if cnt % frame_interval == 0:
            resized_frame = mmcv.imresize(frame, (new_w, new_h))
            result_queue.append(resized_frame)
            _ = result_queue.popleft()
        if cnt - old_cnt == predict_stepsize:
            old_cnt = cnt
            yield result_queue
        cnt += 1
        flag, frame = vid.read()


def main():
    args = parse_args()

    frame_shape, num_frame = frame_extraction(args.video)
    h, w, _ = frame_shape

    # resize frames to shortside 256
    new_w, new_h = mmcv.rescale_size((w, h), (256, np.Inf))
    #frames = [mmcv.imresize(img, (new_w, new_h)) for img in original_frames]
    w_ratio, h_ratio = new_w / w, new_h / h

    # Get clip_len, frame_interval and calculate center index of each clip
    config = mmcv.Config.fromfile(args.config)
    config.merge_from_dict(args.cfg_options)
    val_pipeline = config.data.val.pipeline

    sampler = [x for x in val_pipeline if x['type'] == 'SampleAVAFrames'][0]
    clip_len, frame_interval = sampler['clip_len'], sampler['frame_interval']
    window_size = clip_len * frame_interval
    assert clip_len % 2 == 0, 'We would like to have an even clip_len'
    # Note that it's 1 based here
    timestamps = np.arange(window_size // 2, num_frame + 1 - window_size // 2,
                           args.predict_stepsize)

    # Load label_map
    label_map = load_label_map(args.label_map)
    try:
        if config['data']['train']['custom_classes'] is not None:
            label_map = {
                id + 1: label_map[cls]
                for id, cls in enumerate(config['data']['train']
                                         ['custom_classes'])
            }
    except KeyError:
        pass

    # Get Human detection results
    center_frames = [ind - 1 for ind in timestamps]
    human_detections = detection_inference(args, center_frames)
    for i in range(len(human_detections)):
        det = human_detections[i]
        det[:, 0:4:2] *= w_ratio
        det[:, 1:4:2] *= h_ratio
        human_detections[i] = torch.from_numpy(det[:, :4]).to(args.device)

    # Get img_norm_cfg
    img_norm_cfg = config['img_norm_cfg']
    if 'to_rgb' not in img_norm_cfg and 'to_bgr' in img_norm_cfg:
        to_bgr = img_norm_cfg.pop('to_bgr')
        img_norm_cfg['to_rgb'] = to_bgr
    img_norm_cfg['mean'] = np.array(img_norm_cfg['mean'])
    img_norm_cfg['std'] = np.array(img_norm_cfg['std'])

    # Build STDET model
    try:
        # In our spatiotemporal detection demo, different actions should have
        # the same number of bboxes.
        config['model']['test_cfg']['rcnn']['action_thr'] = .0
    except KeyError:
        pass

    config.model.backbone.pretrained = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))

    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to(args.device)
    model.eval()

    predictions = []

    frames_queue = frame_queue(32, frame_interval, args.predict_stepsize, args.video, new_w, new_h)

    print('Performing SpatioTemporal Action Detection for each clip')
    assert len(timestamps) == len(human_detections)
    prog_bar = mmcv.ProgressBar(len(timestamps))
    for timestamp, proposal, it_frames in zip(timestamps, human_detections, frames_queue):
        print("Frame queue length: ", len(it_frames))
        if proposal.shape[0] == 0:
            predictions.append(None)
            continue

        #start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
        #frame_inds = start_frame + np.arange(0, window_size, frame_interval)
        #frame_inds = list(frame_inds - 1)
        #imgs = [frames[ind].astype(np.float32) for ind in frame_inds]
        imgs = [frame.astype(np.float32) for frame in it_frames]
        _ = [mmcv.imnormalize_(img, **img_norm_cfg) for img in imgs]
        # THWC -> CTHW -> 1CTHW
        input_array = np.stack(imgs).transpose((3, 0, 1, 2))[np.newaxis]
        input_tensor = torch.from_numpy(input_array).to(args.device)

        with torch.no_grad():
            result = model(
                return_loss=False,
                img=[input_tensor],
                img_metas=[[dict(img_shape=(new_h, new_w))]],
                proposals=[[proposal]])
            result = result[0]
            prediction = []
            # N proposals
            for i in range(proposal.shape[0]):
                prediction.append([])
            # Perform action score thr
            for i in range(len(result)):
                if i + 1 not in label_map:
                    continue
                for j in range(proposal.shape[0]):
                    if result[i][j, 4] > args.action_score_thr:
                        prediction[j].append((label_map[i + 1], result[i][j,
                                                                          4]))
            predictions.append(prediction)
        prog_bar.update()

    results = []
    for human_detection, prediction in zip(human_detections, predictions):
        results.append(pack_result(human_detection, prediction, new_h, new_w))

    with open("save.pkl", "wb") as file:
      pickle.dump(zip(center_frames, results), file)

    #def dense_timestamps(timestamps, n):
    #    """Make it nx frames."""
    #    old_frame_interval = (timestamps[1] - timestamps[0])
    #    start = timestamps[0] - old_frame_interval / n * (n - 1) / 2
    #    new_frame_inds = np.arange(
    #        len(timestamps) * n) * old_frame_interval / n + start
    #    return new_frame_inds.astype(np.int)

    #dense_n = int(args.predict_stepsize / args.output_stepsize)

    #frames = [
    #    cv2.imread(frame_paths[i - 1])
    #    for i in dense_timestamps(timestamps, dense_n)
    #]
    #print('Performing visualization')
    #vis_frames = visualize(frames, results)
    #vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames],
    #                            fps=args.output_fps)
    #vid.write_videofile(args.out_filename)

    #tmp_frame_dir = osp.dirname(frame_paths[0])
    #shutil.rmtree(tmp_frame_dir)


if __name__ == '__main__':
    main()

