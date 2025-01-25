import cv2
import numpy as np
import tritonclient.grpc as grpcclient
from numpy.typing import NDArray
from tritonclient.utils import (
    np_to_triton_dtype,
)

import tools

image = cv2.imread("cars_image.png")
orig_h, orig_w, _ = image.shape

# Preprocess the image
image_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_data = cv2.resize(image_data, (640, 640), interpolation=cv2.INTER_LINEAR)
image_data = np.array(image_data) / 255.0
image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
image_data = np.expand_dims(image_data, axis=0).astype(np.float16)

# Triton Inputs/Outputs
_inputs = grpcclient.InferInput(
        "images",
        [1, 3, 640, 640],
        np_to_triton_dtype(np.float16),
    )

_outputs = grpcclient.InferRequestedOutput("output0")


# Initialize Triton Client
triton_client = grpcclient.InferenceServerClient(
                url="localhost:8001",
                verbose=False)

# Sending request
_inputs.set_data_from_numpy(image_data)

# == Inference request ==
infer_results = triton_client.infer(
    model_name="yolov11-engine",
    inputs=[_inputs],
    outputs=[_outputs],
)

outputs_npy = infer_results.as_numpy("output0")

# Postprocess the results
boxes, scores, classes = tools.postprocess(
    outputs_npy, nms_th=0.5, nms_iou_th=0.5, max_det=100, orig_imgsz=(orig_h, orig_w), tgt_imgsz=(640, 640)
)

ann_frame = tools.draw_on_frame(image, boxes, scores, classes)
cv2.imwrite("annotated_frame.png", ann_frame)