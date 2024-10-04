import os

import cv2
import numpy as np
import supervision as sv
from core.buffer import SharedBuffer
from core.inference_client import TritonServerClient
from omegaconf import OmegaConf


class InferenceHandler:
    def __init__(
        self,
        shared_buffer: SharedBuffer = None,
    ):
        self.shm_buffer = shared_buffer

        infer_config_path = os.path.join("configs", "triton.yaml")
        # try catch here
        infer_config = OmegaConf.load(infer_config_path)["inference"]
        self.client = TritonServerClient(
            config_dict={
                "server_url": infer_config["triton_url"],
                "model_name": infer_config["model_name"],
                "model_version": infer_config["model_version"],
                "infer_model_classes": infer_config["model_classes"],
            }
        )

    def draw_on_frame(self, frame):
        pass

    def execute(self):
        try:
            while True:
                # Acquire shared memory semaphore
                self.shm_buffer.frame_available.acquire()

                # Get the latest frame
                current_index = self.shm_buffer.latest_index.value
                frame = self.shm_buffer.buffer[current_index]

                preproc_img = self.client.preprocess(frame, target_dtype=np.float32)
                boxes, scores, classes = self.client.predict(preproc_img)

                dets = sv.Detections(xyxy=boxes, confidence=scores, class_id=classes)
                box_ann = sv.BoxAnnotator()

                annotated = box_ann.annotate(frame, dets)
                cv2.imshow("Det", annotated)
                cv2.imwrite("test.png", annotated)

                # Mark frame as consumed and release space
                self.shm_buffer.index_buffer[current_index] = -1
                self.shm_buffer.space_available.release()

        finally:
            # Release resources and close OpenCV windows
            cv2.destroyAllWindows()
