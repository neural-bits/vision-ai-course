import os
import cv2
import numpy as np
import supervision as sv
from core.buffer import SharedBuffer
from core.inference_client import TritonServerClient
from omegaconf import OmegaConf
from utils.logger import get_logger
import time

logger = get_logger("InferenceHandler")


class InferenceHandler:
    def __init__(self, shared_buffer: SharedBuffer = None):
        self.shm_buffer = shared_buffer
        infer_config_path = os.path.join("configs", "triton.yaml")

        try:
            # Load inference configuration
            self.infer_config = OmegaConf.load(infer_config_path)["inference"]
            self.det_classes= np.asarray(self.infer_config["model_classes"], dtype="<U14")
            self.vr = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"MJPG"), 30, (1920, 1080))
            logger.info(f"Loaded inference configuration from {infer_config_path}.")
            
        except Exception as e:
            logger.error(f"Error loading inference config from {infer_config_path}: {e}")
            raise

    def draw_on_frame(self, frame, boxes, scores, classes):
        """Draw detections on the frame."""
        dets = sv.Detections(xyxy=boxes, confidence=scores, class_id=classes)
        box_ann = sv.BoxAnnotator()
        annotated_frame = box_ann.annotate(frame, dets)
        lab_ann = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)
        labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence
                in zip(self.det_classes[dets.class_id], dets.confidence)
            ]
        annotated_frame = lab_ann.annotate(scene=annotated_frame, detections=dets, labels=labels)
        return annotated_frame

    def execute(self):
        """Main loop for processing frames and running inference."""
        try:
            # Initialize the Triton client
            self.client = TritonServerClient(
                config_dict={
                    "server_url": self.infer_config["triton_url"],
                    "model_name": self.infer_config["model_name"],
                    "model_version": self.infer_config.get("model_version", "1"),
                    "infer_model_classes": self.infer_config["model_classes"],
                }
            )
            logger.info("TritonServerClient initialized.")
            frames_since_inference = 0
            inference_interval = 2
            while True:
                # Wait for a frame to become available in shared memory
                self.shm_buffer.frame_available.acquire()

                # Get the latest frame from the shared buffer
                current_index = self.shm_buffer.latest_index.value
                frame = self.shm_buffer.buffer[current_index]

                if frame is None:
                    logger.warning(f"Frame at index {current_index} is None. Skipping.")
                    self._release_buffer(current_index)
                    continue

                # Perform inference every `inference_interval` frames
                if frames_since_inference >= inference_interval:
                    preproc_img = self.client.preprocess(frame, target_dtype=np.float16)

                    try:
                        boxes, scores, classes = self.client.predict(preproc_img)
                    except Exception as e:
                        logger.error(f"Inference error: {e}")
                        self._release_buffer(current_index)
                        continue

                    # Check for empty detection results
                    if boxes.size == 0 or scores.size == 0 or classes.size == 0:
                        logger.info(f"No detections for frame at index {current_index}.")
                        self._release_buffer(current_index)
                        continue

                    annotated_frame = self.draw_on_frame(frame, boxes, scores, classes)
                    cv2.imwrite(f"annotated_frame.png", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                    
                    frames_since_inference = 0
                else:
                    frames_since_inference += 1

                self._release_buffer(current_index)

        except Exception as e:
            logger.error(f"Unexpected error during execution: {e}")

        finally:
            # Cleanup resources
            logger.info("Cleaning up resources.")
            self.vr.release()
            cv2.destroyAllWindows()
            self._cleanup_shared_buffer()

    def _release_buffer(self, current_index):
        """Helper function to release the shared buffer space."""
        self.shm_buffer.index_buffer[current_index] = -1
        self.shm_buffer.space_available.release()
        logger.info(f"Released buffer space for frame at index {current_index}.")

    def _cleanup_shared_buffer(self):
        """Ensures shared buffer is cleaned up after process termination."""
        if self.shm_buffer:
            try:
                self.shm_buffer.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up shared buffer: {e}")
