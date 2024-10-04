import os
import sys
from typing import Dict, Tuple

import cv2
import numpy as np
import supervision as sv
import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
from numpy.typing import NDArray
from tritonclient.utils import (
    InferenceServerException,
    np_to_triton_dtype,
    triton_to_np_dtype,
)
from utils.logger import get_logger


class TritonServerClient:
    def __init__(self, config_dict: Dict[str, str]):
        """
        Initialize a client that makes image requests to a Triton Inference
        Server using gRPC and CUDA shared memory.

        Parameters
        ----------
        config_dict: config_dict: Dict[str, str]
            Contains necesary elements for establishing a connection, making
            requests and understanding the responses. Should contain:
            - `infer_server_url`      : Triton Server URL.
            - `infer_server_classes`  : Subscribed model label classes.
            - `infer_model_name`      : Subscribed model name
            - `infer_model_vers`      : Subscribed model version
            class only.
        Returns
        -------
        None
        """

        self._inputs = []
        self._outputs = []
        self._config = config_dict

        self.logger = get_logger(self.__class__.__name__)

        # == Connect ==
        self._create_server_context()
        self._health_check()

        # == Fetch Triton Model Config ==
        self._model_config = self.get_model_info()
        self.input_names = [self._model_config["input_name"]]
        self.output_names = self._model_config["output_names"]

        # == Create placeholder tensors ==
        self._create_in_out()

        self.logger.info(
            f"Instantiated Triton {self.__class__.__name__} with GPU support."
        )

    @property
    def model_config(self):
        return self._model_config

    def _create_server_context(self) -> None:
        """
        Establish a connection to the Triton Inference Server using the
        initialized attributes.
        """

        try:
            self.triton_client = grpcclient.InferenceServerClient(
                url=self._config["server_url"],
                verbose=False,
            )
        except InferenceServerException as e:
            self.logger.error(f"Connection to server failed: {str(e)}")
            sys.exit(1)

    def _create_in_out(self) -> None:
        """
        Create attributes for model inputs and outputs from the Triton
        Inference Server configuration. This will help with understanding
        the server's response. For outputs, set the parameters to use data
        from the shared memory.
        """
        self._inputs.append(
            grpcclient.InferInput(
                self.input_names[0],
                [1, *self._model_config["input_shape"]],
                np_to_triton_dtype(self._model_config["input_dtype"]),
            )
        )
        self._outputs = [grpcclient.InferRequestedOutput(oname) for oname in self.output_names]

    def get_model_info(
        self,
    ) -> Dict[str, str]:

        try:
            metadata = self.triton_client.get_model_metadata(
                model_name=self._config["model_name"],
                model_version=self._config["model_version"],
            )
        except InferenceServerException as ex:
            if "Request for unknown model" not in ex.message():
                self.logger.error(f"FAILED to get model metadata. Got {ex.message()}")
                sys.exit(1)
            else:
                self.logger.error("FAILED to get model metadata")
                sys.exit(1)
        try:
            config = self.triton_client.get_model_config(
                model_name=self._config["model_name"],
                model_version=self._config["model_version"],
            )
            if not (config.config.name == self._config["model_name"]):
                self.logger.error(
                    f"FAILED on model_fetching. Asked for{self._config['model_name']} got {config.config.name}."
                )
                sys.exit(1)
        except InferenceServerException as ex:
            self.logger.error(f"FAILED to fetch model_config.Got {ex.message()}")
            sys.exit(1)

        input_metadata = metadata.inputs[0]
        output_metadata = metadata.outputs
        input_config = config.config.input[0]

        input_batch_dim = config.config.max_batch_size > 0
        if isinstance(input_config.format, str):
            FORMAT_ENUM_TO_INT = dict(mc.ModelInput.Format.items())
            input_config.format = FORMAT_ENUM_TO_INT[input_config.format]

        if input_config.format not in (
            mc.ModelInput.FORMAT_NCHW,
            mc.ModelInput.FORMAT_NHWC,
        ):
            got_format = f"{mc.ModelInput.Format.Name(input_config.format)}"
            expected_formats = (
                f"{mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW)} "
                f"or {mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC)}"
            )
            error_message = (
                f"Unexpected input format {got_format}, expecting {expected_formats}"
            )
            self.logger.error(error_message)

        if input_config.format == mc.ModelInput.FORMAT_NHWC:
            h, w, c = (
                input_metadata.shape[1:] if input_batch_dim else input_metadata.shape
            )
        else:
            c, h, w = (
                input_metadata.shape[1:]
                if input_batch_dim
                else input_metadata.shape[:0:-1]
            )

        return {
            "model_name": self._config["model_name"],
            "model_version": self._config["model_version"],
            "input_name": input_metadata.name,
            "output_names": [out.name for out in output_metadata],
            "input_shape": (c, h, w),
            "input_tensor_format": input_config.format,
            "input_dtype": triton_to_np_dtype(input_metadata.datatype),
            "output_dtypes": [
                triton_to_np_dtype(out.datatype) for out in output_metadata
            ],
            "output_shapes": [out.shape for out in output_metadata],
        }

    def _health_check(self) -> None:
        """
        Check if the connection to the Triton Inference Server is ready to be
        used. This method will end the process if the server is not found, or
        the server is not ready, or the model is not ready.
        """

        if not self.triton_client.is_server_live():
            self.logger.error("Connection FAILED: server not live")
            sys.exit(1)

        if not self.triton_client.is_server_ready():
            self.logger.error("Connection FAILED: server not ready")
            sys.exit(1)

        self.logger.debug(f"Model name: {self._config['model_name']}")

        if not self.triton_client.is_model_ready(
            model_name=self._config["model_name"],
            model_version=self._config["model_version"],
        ):
            self.logger.error(
                "Connection FAILED: model not ready. Bad model name/version is in config"
            )
            sys.exit(1)

    def predict(self, input_image: NDArray) -> Tuple[NDArray]:
        # == Set input ==
        self._inputs[-1].set_data_from_numpy(input_image)

        # == Inference request ==
        infer_results = self.triton_client.infer(
            model_name=self._config["model_name"],
            inputs=self._inputs,
            outputs=self._outputs,
        )

        # == Unpack and postprocess ==
        boxes, scores, classes = self.postprocess(results=infer_results)
        return boxes, scores, classes

    def preprocess(
    self,
    input_image: NDArray,
    target_dtype: np.dtype = np.float16,
    ) -> np.ndarray:
        """
        Preprocess the input image with letterbox resizing to maintain aspect ratio.

        Parameters
        ----------
        input_image : NDArray
            The original input image in BGR format.
        target_dtype : np.dtype
            The target datatype (e.g., np.float32) for the model input.

        Returns
        -------
        np.ndarray
            Preprocessed image ready for model inference.
        """
        # Model input dimensions (assuming [batch_size, channels, height, width])
        _, target_h, target_w = self._model_config["input_shape"]

        # Apply letterbox resize to the input image
        # padded_image = self.letterbox_resize(input_image, (target_h, target_w))
        image_data = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        image_data = cv2.resize(image_data, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        image_data = np.array(image_data) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
        image_data = np.expand_dims(image_data, axis=0).astype(target_dtype)
        
        return image_data

    def letterbox_resize(
        self,
        input_image: NDArray,
        target_size: Tuple[int, int],
        color: Tuple[int, int, int] = (128, 128, 128)
    ) -> np.ndarray:
        """
        Resizes the input image using the letterbox method to preserve aspect ratio.

        Parameters
        ----------
        input_image : NDArray
            The original input image in BGR format (from OpenCV).
        target_size : Tuple[int, int]
            The target size (height, width) for the resized image.
        color : Tuple[int, int, int]
            Padding color, default is gray (128, 128, 128).

        Returns
        -------
        np.ndarray
            The resized image with letterbox padding applied.
        """
        img_h, img_w = 1080, 1920
        target_h, target_w = 640,640

        # Calculate scaling factors and the new width and height
        scale = min(target_w / img_w, target_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        # Resize the image with the calculated width and height
        resized_image = cv2.resize(input_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create a new image with the target size and fill with the padding color
        padded_image = np.full((target_h, target_w, 3), color, dtype=np.uint8)

        # Compute top-left corner where the resized image will be placed
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2

        # Place the resized image into the padded image
        padded_image[top:top + new_h, left:left + new_w] = resized_image

        return padded_image

    def postprocess(self, results: grpcclient.InferResult) -> Tuple[np.ndarray]:
        boxes, scores, classes = [results.as_numpy(oname) for oname in self.output_names]
        return boxes, scores, classes

if __name__ == "__main__":
    config_dict = {
        cc.INFER_SERVER_URL: "172.18.0.3:8001",
        "model_name": "yolov8_1.0.0",
        "model_version": "1",
        cc.INFER_MODEL_CLASSES: [
            "person",
            "cart",
            "product",
            "trash",
            "pallet",
            "pallet_truck",
        ],
    }
    image_path = "image.png"
    if not os.path.exists(image_path):
        video_frame = av.VideoFrame(width=1920, height=1080, format="yuv420p")
    else:
        container = av.open(image_path)
        frame = next(container.decode(video=0))
        yuv_data = frame.to_ndarray(format="yuv420p")
        video_frame = av.VideoFrame.from_ndarray(yuv_data, format="yuv420p")

    tis_client = TISClient(config_dict=config_dict)
    boxes, scores, classes = tis_client.predict(video_frame)

    # [DEPRECATED] TEST DRAWING
    from src.ingestion.conversions import yuv2rgb_cv2

    rgb_frame = yuv2rgb_cv2(video_frame)
    for b in boxes:
        cv2.rectangle(
            rgb_frame,
            (int(b[0]), int(b[1])),
            (int(b[2]), int(b[3])),
            color=(255, 0, 0),
            thickness=1,
            lineType=cv2.LINE_4,
        )

    cv2.imwrite("temp_result.png", rgb_frame)
