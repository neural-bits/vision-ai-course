import json
import time

import numpy as np
import triton_python_backend_utils as pb_utils
from tools import postprocess


class TritonPythonModel:
    """Every Python model that is created must have "TritonPythonModel" as the class name"""

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.logger = pb_utils.Logger
        self.model_config = model_config = json.loads(args["model_config"])
        self.model_name = model_config["name"]
        self.model_vers = model_config["version_policy"]["latest"]["num_versions"]
        self.logger.log_info(
            f"[{self.model_name}]-[{self.model_vers}][STAGE][NMS] Model Config: {model_config}"
        )
        input0_config = pb_utils.get_input_config_by_name(model_config, "output0")
        self.dimension = input0_config["dims"][0]

        output0_config = pb_utils.get_output_config_by_name(model_config, "boxes")
        output1_config = pb_utils.get_output_config_by_name(model_config, "classes")
        output2_config = pb_utils.get_output_config_by_name(model_config, "scores")
        self.out0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])
        self.out1_dtype = pb_utils.triton_string_to_numpy(output1_config["data_type"])
        self.out2_dtype = pb_utils.triton_string_to_numpy(output2_config["data_type"])
        self.logger.log_info(
            f"[{self.model_name}]-[{self.model_vers}][STAGE][NMS] Output Config: {output0_config}, {output1_config}, {output2_config}"
        )
        self.inp0_dtype = pb_utils.triton_string_to_numpy(input0_config["data_type"])

        self.max_dets = int(self.model_config["parameters"]["max_det"]["string_value"])
        self.nms_threshold = float(
            self.model_config["parameters"]["nms_th"]["string_value"]
        )
        self.nms_iou_th = float(
            self.model_config["parameters"]["nms_iou_th"]["string_value"]
        )
        self.logger.log_info(
            f"[{self.model_name}]-[{self.model_vers}][STAGE][NMS] Unpacked Params: {self.max_dets=}, {self.nms_threshold=}, {self.nms_iou_th=}"
        )

        # COMPOSE CUSTOM METRIC
        metrics_prefix = "custom_metric"
        custom_metric_name = f"{metrics_prefix}_{self.model_name}".replace("-", "_")
        self.metric_family = pb_utils.MetricFamily(
            name=custom_metric_name,
            description="Cumulative time spent processing NMS requests",
            kind=pb_utils.MetricFamily.COUNTER,
        )
        self.metric = self.metric_family.Metric(
            labels={"model": self.model_name, "version": str(self.model_vers)}
        )

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        start_ns = time.time_ns()
        try:
            for request in requests:
                # Get INPUTS
                input_tensor = pb_utils.get_input_tensor_by_name(request, "output0")
                inputs = input_tensor.as_numpy()
                self.logger.log_info(
                    f"[{self.model_name}]-[{self.model_vers}][STAGE][NMS] Inputs: {inputs.shape=}"
                )

                batched_class_ids = []
                batched_scores = []
                batched_bboxes = []

                for inp in inputs:
                    boxes, scores, classes = postprocess(
                        inp,
                        nms_th=self.nms_threshold,
                        nms_iou_th=self.nms_iou_th,
                        max_det=self.max_dets,
                    )
                    self.logger.log_info(
                        f"[{self.model_name}]-[{self.model_vers}][STAGE][NMS] Postprocessed: {boxes}, {scores}, {classes}"
                    )

                    if boxes.size == 0 or scores.size == 0 or classes.size == 0:
                        self.logger.log_info(
                            f"[{self.model_name}]-[{self.model_vers}][STAGE][NMS] Skipping input as postprocessed arrays are empty: {boxes.shape=}, {scores.shape=}, {classes.shape=}"
                        )
                        continue
                    if np.shape(classes)[0] > self.max_dets:
                        self.max_dets = np.shape(classes)[0]
                    self.logger.log_info(
                        f"[{self.model_name}]-[{self.model_vers}][STAGE][NMS] Postprocessed After Unpack: {boxes.shape=}, {scores.shape=}, {classes.shape=}"
                    )

                    batched_class_ids.append(classes)
                    batched_scores.append(scores)
                    batched_bboxes.append(boxes)

                max_detections = max(self.dimension, self.max_dets)

                padded_batched_class_ids = []
                padded_batched_scores = []
                padded_batched_bboxes = []

                for classes, scores, boxes in zip(
                    batched_class_ids, batched_scores, batched_bboxes
                ):
                    padd_dim = max_detections - len(classes)
                    if padd_dim < 0:
                        raise ValueError(
                            "ERROR: Dimensions of output tensors are not valid."
                        )

                    # Pad the arrays if necessary
                    if len(classes) > 0:
                        classes = np.concatenate(
                            [classes, -1 * np.ones(shape=(padd_dim,))], axis=0
                        )
                        scores = np.concatenate(
                            [scores, -1 * np.ones(shape=(padd_dim,))], axis=0
                        )
                        boxes = np.concatenate(
                            [boxes, -1 * np.ones(shape=(padd_dim, 4))], axis=0
                        )
                    else:
                        classes = -1 * np.ones(shape=(padd_dim,))
                        scores = -1 * np.ones(shape=(padd_dim,))
                        boxes = -1 * np.ones(shape=(padd_dim, 4))

                    padded_batched_class_ids.append(classes)
                    padded_batched_scores.append(scores)
                    padded_batched_bboxes.append(boxes)

                # Convert lists to numpy arrays
                batched_class_ids = np.array(padded_batched_class_ids)
                batched_scores = np.array(padded_batched_scores)
                batched_bboxes = np.array(padded_batched_bboxes)

                self.logger.log_info(
                    f"[{self.model_name}]-[{self.model_vers}][STAGE][NMS] Batched After Padding: {batched_class_ids.shape=}, {batched_scores.shape=}, {batched_bboxes.shape=}"
                )

                # Create output tensors
                out_tensor_0 = pb_utils.Tensor(
                    "boxes", batched_bboxes.astype(self.out0_dtype)
                )
                out_tensor_1 = pb_utils.Tensor(
                    "classes", batched_class_ids.astype(self.out1_dtype)
                )
                out_tensor_2 = pb_utils.Tensor(
                    "scores", batched_scores.astype(self.out2_dtype)
                )

                self.logger.log_info(
                    f"[{self.model_name}]-[{self.model_vers}][STAGE][NMS] Batched: {batched_class_ids.shape=}, {batched_scores.shape=}, {batched_bboxes.shape=}"
                )

                # Create and append the inference response
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[out_tensor_0, out_tensor_1, out_tensor_2]
                )
                responses.append(inference_response)

        except IndexError as e:
            self.logger.log_error(f"IndexError: Invalid input tensor shape {e}")
        except Exception as e:
            self.logger.log_error(f"Exception: {e}")
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[], error=pb_utils.TritonError(f"An error occurred: {e}")
            )
            responses.append(inference_response)
        end_ns = time.time_ns()
        self.metric.increment(end_ns - start_ns)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        del self.metric
        del self.metric_family
        print("Finished postprocessing")
