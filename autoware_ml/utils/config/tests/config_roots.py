"""Config roots of the tests.

The bundled configs hold no T4Dataset experiment. The test external root is laid out like the
configs of a private repository and binds the bundled T4Dataset bases to a test database, a test
taxonomy with synthetic labels and a test ego, so the tests compose every bundled base.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from autoware_ml.utils.config.resolvers import register_config_resolvers
from autoware_ml.utils.config.settings import CONFIGS_ROOT, Settings

EXTERNAL_CONFIGS_ROOT = Path(__file__).parent / "external_configs"
SEARCHPATH_OVERRIDE = f"hydra.searchpath=[file://{EXTERNAL_CONFIGS_ROOT}]"


def task_config_names() -> list[str]:
    """
    Name of every task config of both roots, relative to the config roots.

    Returns:
      list[str]: Config names such as ``tasks/detection3d/ptv3/base``, sorted.
    """
    return sorted(
        path.relative_to(root).with_suffix("").as_posix()
        for root in (CONFIGS_ROOT, EXTERNAL_CONFIGS_ROOT)
        for path in (root / "tasks").rglob("*.yaml")
    )


def compose_task(config_name: str, overrides: Sequence[str] = ()) -> DictConfig:
    """
    Compose a config of either root with the run context the entrypoints provide.

    Args:
      config_name: Config name relative to the config roots.
      overrides: Hydra overrides applied on top.

    Returns:
      DictConfig: The composed config.
    """
    # Like the CLI, the root holding the config is opened and the other one is searched
    external = (EXTERNAL_CONFIGS_ROOT / f"{config_name}.yaml").is_file()
    root, searched = (
        (EXTERNAL_CONFIGS_ROOT, CONFIGS_ROOT) if external else (CONFIGS_ROOT, EXTERNAL_CONFIGS_ROOT)
    )
    register_config_resolvers(Settings())
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        cfg = compose(
            config_name=config_name,
            overrides=[f"hydra.searchpath=[file://{searched}]", *overrides],
            return_hydra_config=True,
        )
    HydraConfig.instance().set_config(cfg)
    return cfg
