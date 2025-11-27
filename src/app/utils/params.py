from __future__ import annotations
from typing import Any, Dict
import dvc.api


def load_params(section: str) -> Dict[str, Any]:
    all_params: Dict[str, Any] = dvc.api.params_show()  # type: ignore
    return all_params.get(section, {})
