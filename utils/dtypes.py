from pathlib import Path
from typing import Annotated, Any, Dict, TypeVar, Union

import numpy as np
from PIL import Image
from pydantic import StringConstraints
from pydantic.functional_validators import AfterValidator

T = TypeVar("T")
UNSET = object()  # Sentinel for unset values

# Type for file or directory path
Url = str
PathLike = Union[Path, str]
ImgLike = Image.Image | np.ndarray | Url | PathLike

# Type for dictionary with string as key
SDict = Dict[str, T]

JsonObject = Any
YamlObject = Any
NonEmptyStr = Annotated[str, StringConstraints(strict=True, min_length=1)]


def non_zero(v: int) -> int:
    if v == 0:
        raise ValueError("Input should be a non-zero integer")
    return v


NonZeroInt = Annotated[int, AfterValidator(non_zero)]
