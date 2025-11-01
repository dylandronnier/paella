import inspect
import typing as tp
from dataclasses import dataclass, field, make_dataclass

from omegaconf import MISSING

from models import *


def make_config_dataclass(
    fun: tp.Callable,
    target_name: tp.Optional[str] = None,
    config_name: tp.Optional[str] = None,
    bases=(),
    skip_args: list[str] = [],
) -> type[object]:
    mod = inspect.getmodule(fun)
    if mod is None:
        raise TypeError
    module_name = mod.__name__

    if config_name is None:
        config_name = "Config" + fun.__name__

    if target_name is None:
        target_name = fun.__name__

    new_fields = [("_target_", str, module_name + "." + target_name)]
    for f in inspect.signature(fun).parameters.values():
        if f.name in skip_args:
            continue
        if f.default == inspect._empty:
            new_fields.append((f.name, f.annotation, field(default=MISSING)))
        else:
            new_fields.append((f.name, f.annotation, f.default))
    # Dataclass here ensures runtime type safety
    return make_dataclass(
        cls_name=config_name,
        fields=new_fields,
        bases=bases,
    )


def make_config_dictionary(
    fun: tp.Callable,
    skip_args: list[str] = [],
) -> dict[str, tp.Any]:
    mod = inspect.getmodule(fun)
    if mod is None:
        raise TypeError
    module_name = mod.__name__
    res = {"_target_": module_name + "." + fun.__name__}
    for f in inspect.signature(fun).parameters.values():
        if f.name in skip_args:
            continue
        if f.default == inspect._empty:
            res[f.name] = MISSING
        else:
            res[f.name] = f.default
    return res


@dataclass
class DatasetConfig:
    """Configuration of the dataset."""

    hf_id: str = MISSING
    images_width: int = MISSING
    images_height: int = MISSING
    channels: int = MISSING
    num_classes: int = MISSING
    image_column_name: str = "image"


@dataclass
class EarlyStopConfig:
    min_delta: float = 0
    patiente: int = MISSING


@dataclass
class GradientDescentConfig:
    """Class that defines the parameters for the gradient descent."""

    # Number of epochs
    epochs: int = MISSING

    # Batch size
    batch_size: int = MISSING

    # Early stopping
    early_stop: tp.Optional[EarlyStopConfig] = None

    # Optimizer
    optimizer: tp.Any = MISSING

    _partial_: bool = True


@dataclass
class TrackingConfig:
    """Configuration of the logging."""

    frequency: int = MISSING
    sample_prediction_size: int = MISSING


@dataclass
class GlobalConfig:
    """Configuration of the experiment."""

    gradient_descent: GradientDescentConfig = MISSING
    dataset: DatasetConfig = MISSING
    tracking: TrackingConfig = MISSING
    seed: int = 42


# def prepare_configuration_store(cs: ConfigStore):
#     cs.store(name="base_config", node=GlobalConfig)
#
#     # Training config
#     cs.store(
#         group="training_hp", name="base_trainingconfig", node=GradientDescentConfig
#     )
#     for f in [sgd, adam, adamw]:
#         dict_args = make_config_dictionary(f)
#         cs.store(
#             name="base_" + f.__name__, group="training_hp/optimizer", node=dict_args
#         )
#
#     # Database config
#     cs.store(group="dataset", name="base_datasetconfig", node=DatasetConfig)
#
#     # Model config
#     for cls in [SimpleCNN, ResNet, DenseNet, ViT]:
#         dc_cls = make_config_dataclass(
#             cls.__init__,
#             target_name=cls.__name__,
#             config_name="Config" + cls.__name__,
#             bases=(ModelConfig,),
#             skip_args=["self", "channels", "num_classes", "rngs"],
#         )
#         cs.store(name="base_" + cls.__name__, group="model", node=dc_cls)
