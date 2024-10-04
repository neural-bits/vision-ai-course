from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


def postprocess(
    raw_dets, nms_th: float, nms_iou_th: float, max_det: int, **kwargs
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Transform response from Triton Inference Server to be compatible with
    the Everdoor pipeline. If there's no detections, empty arrays for:
    - boxes : ndarray(N, 4, dtype=float)
    - scores: ndarray(N, 1, dtype=float)
    - classes: ndarray(N, , dtype=float)

    Parameters
    ----------
    results: grpcclient.InferResult
        Response from the Triton Inference Server.

    Returns
    -------
    Tuple[np.ndarray]
        A tuple that containse nd.arrays for:
        - boxes : (N, bounding box coordinates)
        - scores: (N, confidence of each prediction)
        - classes: (N, class label_id assigned to each prediction)
    """
    if len(raw_dets.shape) > 2:
        raw_dets = raw_dets.squeeze()

    # Initialize empty arrays for results
    boxes, scores, classes = np.empty((0, 4)), np.array([]), np.array([])

    if raw_dets is not None and raw_dets.shape[0] > 0:
        predictions = raw_dets.T
        scores = np.max(predictions[:, 4:], axis=1)
        
        # Log the raw predictions and scores
        print(f"Raw predictions: {predictions}")
        print(f"Raw scores: {scores}")

        valid_scores_mask = scores > float(nms_th)
        predictions = predictions[valid_scores_mask, :]
        scores = scores[valid_scores_mask]

        # Log filtered predictions and scores
        print(f"Filtered predictions: {predictions}")
        print(f"Filtered scores: {scores}")

        if len(scores) == 0:
            return boxes, scores, classes

        classes = np.argmax(predictions[:, 4:], axis=1)
        boxes = xywh2xyxy(predictions[:, :4])

        # Perform NMS
        i = non_max_suppression(
            boxes, scores, iou_thres=nms_iou_th
        )

        if i.shape[0] > max_det:
            i = i[:max_det]
        
        #Upscale bboxes
        orig_imgsz = kwargs.get("orig_imgsz", None)
        tgt_imgsz = kwargs.get("tgt_imgsz", None)
        if orig_imgsz and tgt_imgsz:
            boxes = upscale_bounding_boxes(boxes, *orig_imgsz, *tgt_imgsz)
        return boxes[i], scores[i], classes[i]
    
    return boxes, scores, classes

def upscale_bounding_boxes(
    boxes: np.ndarray,
    orig_wh: Tuple[int, int],
    tgt_wh: Tuple[int, int]
) -> np.ndarray:
    """
    Upscale bounding boxes from target size (tgt_w, tgt_h) to original size (orig_w, orig_h).
    
    Parameters
    ----------
    boxes : np.ndarray
        Array of bounding boxes in [x1, y1, x2, y2] format with shape (N, 4).
    orig_w : int
        Original image width.
    orig_h : int
        Original image height.
    tgt_w : int
        Target image width (the size the bounding boxes were generated at).
    tgt_h : int
        Target image height (the size the bounding boxes were generated at).
        
    Returns
    -------
    np.ndarray
        Array of upscaled bounding boxes with shape (N, 4).
    """
    # Calculate the scaling factors
    scale_x = orig_wh[1] / tgt_wh[1]
    scale_y = orig_wh[0] / tgt_wh[0]

    # Scale the bounding boxes
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x  # Scale x1 and x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y  # Scale y1 and y2

    return boxes

def xywh2xyxy(bboxes_tensor: np.array) -> NDArray:
    """
    Converts Nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where
    xy1=top-left, xy2=bottom-right

    Parameters
    ----------
    bboxes_tensor : np.array
        Numpy array Nx4 of bounding boxes in [x, y, w, h] format.

    Returns
    -------
    np.ndarray
        Array of bounding boxes converted to [x1, y1, x2, y2] format.
    """
    y = np.copy(bboxes_tensor)
    y[:, 0] = bboxes_tensor[:, 0] - bboxes_tensor[:, 2] / 2  # top left x
    y[:, 1] = bboxes_tensor[:, 1] - bboxes_tensor[:, 3] / 2  # top left y
    y[:, 2] = bboxes_tensor[:, 0] + bboxes_tensor[:, 2] / 2  # bottom right x
    y[:, 3] = bboxes_tensor[:, 1] + bboxes_tensor[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(
    boxes: np.ndarray, scores: np.ndarray, iou_thres: float
) -> NDArray:
    """
    Apply Non-Maximum Suppression (NMS) to remove overlapping bounding boxes.

    Parameters
    ----------
    boxes : np.ndarray
        An array of shape (n, 4) containing coordinates of the n bounding boxes
            (x1, y1, x2, y2).
    scores : np.ndarray
        An array of shape (n,) containing the corresponding confidence scores
        of the n boxes.
    iou_thres : float
        Threshold for Intersection over Union (IoU) ratio used to decide which
        boxes to keep.

    Returns
    -------
    np.ndarray
        numpy.ndarray: An array of indexes of boxes to keep after NMS is applied.

    Notes
    -----
        If there are no boxes, returns an empty list.
    """
    if boxes.shape[0] == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep: List[np.int] = []
    while order.size > 0:
        i = order[0]  # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)  # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1)  # maxiumum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]

    return np.asarray(keep)
