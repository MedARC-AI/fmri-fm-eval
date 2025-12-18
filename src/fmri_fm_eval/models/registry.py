import importlib
import logging
import pkgutil
from typing import Callable

import fmri_fm_eval.models
from fmri_fm_eval.models.base import ModelTransformPair, ModelFn

_logger = logging.getLogger(__package__)

_MODEL_REGISTRY: dict[str, Callable[..., ModelTransformPair]] = {}


def register_model(name_or_func: str | ModelFn | None = None):
    """Register a model and optional tranform.

    ```
    @register_model
    def new_model(**kwargs):
        ...
        return transform, model
    ```
    """

    def _decorator(func: Callable):
        name = name_or_func if isinstance(name_or_func, str) else func.__name__
        if name in _MODEL_REGISTRY:
            _logger.warning(f"Model {name} already registered; overwriting.")
        _MODEL_REGISTRY[name] = func
        return func

    if isinstance(name_or_func, Callable):
        return _decorator(name_or_func)
    return _decorator


def create_model(name: str, **kwargs) -> ModelTransformPair:
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Model {name} not registered")
    model_pair = _MODEL_REGISTRY[name](**kwargs)

    if not isinstance(model_pair, tuple):
        transform, model = None, model_pair
    else:
        transform, model = model_pair
    return transform, model


def list_models() -> list[str]:
    return list(_MODEL_REGISTRY)


def import_model_plugins():
    """Finds and imports all plugins registering new models."""
    # https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-namespace-packages
    plugins = {}
    for finder, name, ispkg in pkgutil.iter_modules(fmri_fm_eval.models.__path__):
        if not (name in {"base", "registry", "template"} or name.startswith("test_")):
            try:
                plugins[name] = importlib.import_module(f"fmri_fm_eval.models.{name}")
            except Exception as exc:
                _logger.warning(f"Import model plugin {name} failed: {exc}", exc_info=True)
    return plugins


# import all discovered plugins to register
_MODEL_PLUGINS = import_model_plugins()
