import json
import time
import numpy as np
import triton_python_backend_utils as pb_utils
from tools import xywh2xyxy, non_max_suppression, upscale_bounding_boxes
from typing import Tuple
from numpy.typing import NDArray


class TritonPythonModel:
    """NMS Python model for Triton"""

    def initialize(self, args):
        """Called once when the model is loaded to initialize state."""
        self.model_config = json.loads(args["model_config"])
        self.logger = pb_utils.Logger
        self.model_name = self.model_config["name"]

        # Get the dimensions and data types of input and output
        input_config = pb_utils.get_input_config_by_name(self.model_config, "output0")
        self.dimension = input_config["dims"][0]

        output_config_0 = pb_utils.get_output_config_by_name(self.model_config, "boxes")
        output_config_1 = pb_utils.get_output_config_by_name(self.model_config, "classes")
        output_config_2 = pb_utils.get_output_config_by_name(self.model_config, "scores")

        self.out0_dtype = pb_utils.triton_string_to_numpy(output_config_0["data_type"])
        self.out1_dtype = pb_utils.triton_string_to_numpy(output_config_1["data_type"])
        self.out2_dtype = pb_utils.triton_string_to_numpy(output_config_2["data_type"])

        # Parameters from model configuration
        self.max_dets = int(self.model_config["parameters"]["max_det"]["string_value"])
        self.conf_threshold = float(self.model_config["parameters"]["conf_threshold"]["string_value"])
        self.nms_iou_th = float(self.model_config["parameters"]["nms_iou_th"]["string_value"])
        
        orig_imgsz = self.model_config["parameters"]["orig_imgsz"]["string_value"].split(',')
        tgt_imgsz = self.model_config["parameters"]["tgt_imgsz"]["string_value"].split(',')
        self.orig_imgsz = tuple(map(int, orig_imgsz))
        self.tgt_imgsz = tuple(map(int, tgt_imgsz))

        self.logger.log_info(f"Initialized NMS Model: {self.model_name}")

    def execute(self, requests):
        """Performs NMS on input tensors without batching."""
        responses = []

        for request in requests:
            try:
                # Extract input tensor
                input_tensor = pb_utils.get_input_tensor_by_name(request, "output0")
                inputs = input_tensor.as_numpy()
                
                # Perform NMS and postprocess the input tensor
                boxes, scores, classes = self.postprocess(inputs)

                # Log empty detection case
                if boxes.size == 0:
                    self.logger.log_info("No valid detections after NMS.")
                
                # Create output tensors and response
                out_tensor_0 = pb_utils.Tensor("boxes", boxes.astype(self.out0_dtype))
                out_tensor_1 = pb_utils.Tensor("classes", classes.astype(self.out1_dtype))
                out_tensor_2 = pb_utils.Tensor("scores", scores.astype(self.out2_dtype))

                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[out_tensor_0, out_tensor_1, out_tensor_2]
                )
                responses.append(inference_response)

            except Exception as e:
                self.logger.log_error(f"Error during NMS execution: {e}")
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(f"Error: {e}")
                )
                responses.append(inference_response)

        return responses

    def postprocess(self, raw_dets) -> Tuple[NDArray, NDArray, NDArray]:
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

        try:
            if raw_dets is not None and raw_dets.shape[0] > 0:
                predictions = raw_dets.T
                # self.logger.log_info(f"Predictions shape: {predictions.shape}")

                # Check if there are enough columns for class scores
                if predictions.shape[1] <= 4:
                    raise ValueError(f"Not enough columns in predictions: {predictions.shape}")

                scores = np.max(predictions[:, 4:], axis=1)
                valid_scores_mask = scores > float(self.conf_threshold)
                predictions = predictions[valid_scores_mask, :]
                scores = scores[valid_scores_mask]

                # Log filtered predictions and scores
                # self.logger.log_info(f"Filtered predictions shape: {predictions.shape}")
                # self.logger.log_info(f"Filtered scores: {scores}")

                if len(scores) == 0:
                    return boxes, scores, classes

                classes = np.argmax(predictions[:, 4:], axis=1)
                boxes = xywh2xyxy(predictions[:, :4])

                # Perform NMS
                i = non_max_suppression(boxes, scores, iou_thres=self.nms_iou_th)

                if i.shape[0] > self.max_dets:
                    i = i[:self.max_dets]

                # Check if i has valid indices
                if np.max(i) >= len(boxes):
                    raise IndexError(f"Index {np.max(i)} is out of bounds for boxes size {len(boxes)}")
                
                boxes = upscale_bounding_boxes(boxes, self.orig_imgsz, self.tgt_imgsz)
                return boxes[i], scores[i], classes[i]
        except Exception as e:
            self.logger.log_error(f"Error during postprocessing: {e}")

        return boxes, scores, classes
    def finalize(self):
        """Called once when the model is unloaded (optional)."""
        self.logger.log_info("Finalizing NMS model")

