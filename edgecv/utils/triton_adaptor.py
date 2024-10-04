import os
import shutil
from pathlib import Path

from omegaconf import OmegaConf
from utils.logger import get_logger
from utils.template_filler import TemplateFiller

logger = get_logger("TritonAdapter")
export_conf = OmegaConf.load(os.path.join("configs", "export.yaml"))
triton_conf = OmegaConf.load(os.path.join("configs", "triton.yaml"))


def create_model_repository() -> None:
    logger.info("Creating Triton Server Model Repository")
    trt_model_path = os.path.join(
        triton_conf["model_repository"], triton_conf["trt_name"]
    )
    trt_version_path = os.path.join(trt_model_path, triton_conf["model_version"])
    os.makedirs(trt_version_path, exist_ok=True)
    logger.info(f"Step 1 : Created TensorRT Triton model : {trt_model_path}")
    logger.info(f"-> Copy TensorRT model : {trt_model_path}")
    shutil.copy(
        Path().resolve / "pretrained" / "model.plan",
        Path(trt_version_path) / "model.plan",
    )

    nms_path = os.path.join(
        triton_conf["model_repository"], triton_conf["nms_plugin_name"]
    )
    nms_version_path = os.path.join(nms_path, triton_conf["model_version"])
    os.makedirs(nms_version_path, exist_ok=True)
    logger.info("Step 2")
    logger.info(f"-> Created NMS Triton model : {nms_version_path}")
    logger.info(f"-> Copying Plugin Files : {nms_path}")
    shutil.copy(
        Path(triton_conf["plugin_path"]) / "config.pbtxt",
        Path(nms_path) / "config.pbtxt",
    )
    logger.info(f"-> config.pbtxt -> {nms_path}/config.pbtxt")
    shutil.copy(
        Path(triton_conf["plugin_path"]) / "model.py",
        Path(nms_version_path) / "model.py",
    )
    logger.info(f"-> model.py -> {nms_version_path}/model.py")
    shutil.copy(
        Path(triton_conf["plugin_path"]) / "tools.py", Path(nms_path) / "tools.py"
    )
    logger.info(f"-> tools.py -> {nms_path}/tools.py")
    logger.info("Done")


def generate_ensemble() -> None:
    """Will generate needed files for the model_repository ensemble Yolov11 + NMS"""
    trt_meta = {}
    nms_meta = {}


def write_triton_config() -> None:

    model_name = normalized_path.split(os.sep)[-2]
    max_batch_size = self.full_cfg.export.max_batch_size
    default_model_filename = "model.plan"
    dynamic_batching_enabled = self.full_cfg.export.dynamic
    instance_groups = [
        {
            "kind": "KIND_GPU",
            "count": 1,
            "gpus": [0],
        }
    ]
    inputs = [
        {
            "name": inp.name,
            "data_type": inp.dtype,
            "format": "FORMAT_NCHW",
            "dims": inp.shape,
        }
        for inp in list(self.full_cfg.export.trt_inputs)
    ]
    outputs = [
        {"name": out.name, "data_type": out.dtype, "dims": out.shape}
        for out in list(self.full_cfg.export.trt_outputs)
    ]

    output = TemplateFiller.fill_triton_config(
        {
            "model_name": model_name,
            "max_batch_size": max_batch_size,
            "default_model_filename": default_model_filename,
            "inputs": inputs,
            "outputs": outputs,
            "dynamic_batching_enabled": dynamic_batching_enabled,
            "instance_groups": instance_groups,
        }
    )
    with open(self.triton_config_location, "w") as f:
        f.write(output)
    logger.info(f"Wrote Triton config file: {output}")
