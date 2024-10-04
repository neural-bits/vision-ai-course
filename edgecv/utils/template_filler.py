from enum import Enum
from typing import Dict, Union

import jinja2
from utils.logger import get_logger

logger = get_logger("TemplateFiller")


class TemplateType(Enum):
    """Enum for the template types"""

    TENSORRT = "tensorrt"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"


# Defining the registry of templates
templates_registry: Dict[TemplateType, str] = {
    TemplateType.TENSORRT: "templates/T_TritonModelConfig.jinja",
    TemplateType.ENSEMBLE: "templates/T_TritonEnsembleConfig.jinja",
    TemplateType.CUSTOM: "templates/T_TritonPythonConfig.jinja",
}


def get_template_path(template_type: TemplateType) -> str:
    return templates_registry[template_type]


class TemplateFiller:
    @staticmethod
    def fill_tensorrt_config(
        fields: Dict[str, Union[str, int]],
    ) -> str:
        """
        Given a dictionary of data, fill the triton config template.

        Parameters
        ----------
        data : Dict[str, Union[str, int]]
            Dictionary of data to fill the template, containing the following:"
            * model_name: str
                Name of the model.
            * max_batch_size: int
                Maximum batch size.
            * default_model_filename: str
                Default model filename.
            * inputs: List[Dict[str, str]]
                * List of dictionaries containing the input data.
                * [{name: str, data_type: str, format: str, dims: List[int]}]
            * outputs: List[Dict[str, str]]
                * List of dictionaries containing the output data.
                * [{name: str, data_type: str, dims: List[int]}]
            * dynamic_batching_enabled: bool
                Flag indicating if dynamic batching is enabled.
            * instance_groups: List[Dict[str, Union[str, int]]]
                List of dictionaries containing the instance groups.
        """
        template_file = open(get_template_path(TemplateType.TENSORRT), "r").read()
        template = jinja2.Template(template_file)

        config_text = template.render(
            model_name=fields["model_name"],
            platform="tensorrt_plan",
            max_batch_size=fields["max_batch_size"],
            default_model_filename=fields["default_name"],
            inputs=fields["inputs"],
            outputs=fields["outputs"],
            dynamic_batching_enabled=fields["dynamic_batching_enabled"],
            instance_groups=fields["instance_groups"],
        )

        text_lines = config_text.strip().splitlines()
        text_clean = "\n".join(line for line in text_lines if line.strip())

        return text_clean

    def fill_plugin(self, fields: Dict[str, Union[str, int]]):
        template_file = open(get_template_path(TemplateType.CUSTOM), "r").read()
        template = jinja2.Template(template_file)

        config_text = template.render(
            model_name=fields["name"],
            default_model_name="model.py",
            platform="python",
            max_batch_size=fields["max_batch_size"],
            input_name=fields["input_name"],
            input_data_type=fields["input_dtype"],
            input_dims=fields["input_shape"],
            preferred_batch_sizes=fields["preferred_batch_sizes"],
            max_det=fields["max_det"],
            nms_th=fields["nms_th"],
            nms_iou_th=fields["nms_iou_th"],
        )

        text_lines = config_text.strip().splitlines()
        text_clean = "\n".join(line for line in text_lines if line.strip())

        return text_clean

    def fill_ensemble(self, fields: Dict[str, Union[str, int]]):
        template_file = open(get_template_path(TemplateType.ENSEMBLE), "r").read()
        template = jinja2.Template(template_file)

        config_text = template.render(
            ensemble_name=fields["ensemble"],
            max_batch_size=fields["max_batch_size"],
            input_name=fields["input_name"],
            input_data_type=fields["input_dtype"],
            input_dims=fields["input_shape"],
            model_1=fields["model_1_name"],
            model_2=fields["model_2_name"],
        )

        text_lines = config_text.strip().splitlines()
        text_clean = "\n".join(line for line in text_lines if line.strip())

        return text_clean

    def fill_nms_plugin(self, fields: Dict[str, Union[str, int]]):
        template_file = open(get_template_path(TemplateType.CUSTOM), "r").read()
        template = jinja2.Template(template_file)

        config_text = template.render(
            model_name=fields["name"],
            default_model_name="model.py",
            platform="python",
            max_batch_size=fields["max_batch_size"],
            input_name=fields["input_name"],
            input_data_type=fields["input_dtype"],
            input_dims=fields["input_shape"],
            preferred_batch_sizes=fields["preferred_batch_sizes"],
            max_det=fields["max_det"],
            nms_th=fields["nms_th"],
            nms_iou_th=fields["nms_iou_th"],
        )

        text_lines = config_text.strip().splitlines()
        text_clean = "\n".join(line for line in text_lines if line.strip())

        return text_clean