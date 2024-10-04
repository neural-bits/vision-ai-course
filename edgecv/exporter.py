import os
from pathlib import Path

import docker
import fire
import ultralytics as ul
from omegaconf import DictConfig, OmegaConf
from utils.logger import get_logger

logger = get_logger("ONNXExporter")
export_config_path = os.path.join("configs", "export.yaml")
conf = OmegaConf.load(export_config_path)


def pt2onnx() -> None:
    """
    Converts the YOLOv11 model from .PyTorch format into the unified .ONNX format.
    We're using ultralytics package to handle the conversion and OmegaConf to manipulate
    and update the export.yaml config file.
    """
    onnx_cfg = conf.get("onnx")
    model = ul.YOLO(model=onnx_cfg["weights_path"])
    m_path = model.export(
        format="onnx",
        device=onnx_cfg["device"],
        imgsz=onnx_cfg["image_size"],
        nms=onnx_cfg["with_nms"],
        half=onnx_cfg["is_half"],
        simplify=onnx_cfg["is_simplified"],
        dynamic=onnx_cfg["is_dynamic"],
    )
    logger.info(f"Exported {conf['onnx']['weights_path']} to ONNX format")
    onnx_updated = OmegaConf.merge(
        onnx_cfg,
        DictConfig({"onnx_path": m_path}),
    )
    merged = OmegaConf.merge(
        conf,
        DictConfig({"onnx": onnx_updated}),
    )
    
    
    OmegaConf.save(merged, export_config_path)

    logger.info("Added ONNX export outputs details to conf")


def onnx2trt() -> None:
    """
    Handle conversion from ONNX to TensorRT format for super-fast inference speeds.
    """
    # Fetch configuration
    onnx_cfg = conf.get("onnx")
    trt_cfg = conf.get("tensorrt")

    if not onnx_cfg or not trt_cfg:
        raise ValueError(
            "ONNX or TensorRT configuration is missing. Check your configuration file."
        )

    # Log starting the process
    logger.info("Starting ONNX to TensorRT conversion.")

    # Container Warmup
    gpu_devices = [trt_cfg["device"]]
    gpu_config = {
        "device_requests": [{"count": len(gpu_devices), "capabilities": [["gpu"]]}],
        "devices": [f"/dev/nvidia{n}" for n in gpu_devices],
    }

    client = docker.from_env()
    volume_mapping = {
        f"{Path().resolve()/ 'pretrained'}": {"bind": "/workspace", "mode": "rw"}
    }

    try:
        logger.info(
            f"Starting container {trt_cfg['image']} with GPU support and volume mapping."
        )

        container = client.containers.run(
            trt_cfg["image"],
            command='sh -c "while true; do sleep 3600; done"',
            detach=True,
            stdout=True,
            stderr=True,
            remove=True,
            volumes=volume_mapping,
            **gpu_config,
            name="onnx2tensorrt-container",
        )
        logger.info("Docker container started successfully.")

        # Export command
        try:
            _exec = "trtexec"
            _onnx_path = Path(onnx_cfg["onnx_path"]).stem
            _o2t = f" --onnx=/workspace/{_onnx_path}.onnx --saveEngine=/workspace/model.plan --{trt_cfg['dtype'].lower()}"
            _shapes = f" --minShapes={trt_cfg['minShapes']} --optShapes={trt_cfg['optShapes']} --maxShapes={trt_cfg['maxShapes']}"
            command = _exec + _o2t + _shapes

            logger.info(f"Running TensorRT conversion command: {command}")

            exec_result = container.exec_run(command, detach=False)

            if exec_result.exit_code != 0:
                raise RuntimeError(
                    f"Error during conversion: {exec_result.output.decode('utf-8')}"
                )
            else:
                logger.info("Conversion successful.")
                logger.info(exec_result.output.decode("utf-8"))

        except Exception as e:
            logger.error(f"Error executing TensorRT conversion command: {e}")
            raise RuntimeError(f"Error executing TensorRT conversion command: {e}")

    except docker.errors.DockerException as docker_error:
        logger.error(f"Docker error: {docker_error}")
        raise RuntimeError(f"Error starting Docker container: {docker_error}")

    finally:
        # Clean up: stop and remove the container after use
        if container:
            logger.info("Stopping and removing Docker container.")
            container.stop()
            logger.info("Docker container removed.")


def pipeline():
    # pt2onnx()
    onnx2trt()


if __name__ == "__main__":
    fire.Fire(pipeline)
