import base64
import datetime as dt
import os
import re
import sys
import time
import warnings
from contextlib import nullcontext
from typing import Iterable, Optional

from pydantic import validate_call
from typing_extensions import Self

from .anybase import AnyBase
from .dtypes import SDict

try:
    import torch  # type: ignore
except ImportError:
    torch = None


class Timer(AnyBase):
    def __init__(self):
        self._name: Optional[str] = None
        self._start_time: Optional[float] = None
        self._sessions: SDict[float] = {}

    def __call__(self, name: str) -> Self:
        if name in self._sessions:
            raise ValueError(f"Timer '{name}' already exists")
        self._name = name
        return self

    def __enter__(self) -> Self:
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        duration = time.perf_counter() - self._start_time
        self._sessions[self._name] = duration
        self._name = self._start_time = None

    def __repr__(self) -> str:
        data = [f"{k}={round(v, 2)}" for k, v in self._sessions.items()]
        return f"{self.cname_}({', '.join(data)})"

    def __str__(self) -> str:
        return self.__repr__()

    def reset(self) -> None:
        self._sessions.clear()

    @property
    def sessions(self) -> SDict[float]:
        return self._sessions.copy()


def get_date(date_format: str = r"%Y-%m-%d") -> str:
    if date_format == "":
        raise ValueError("`date_format` must not be empty")
    return dt.datetime.now(dt.timezone.utc).strftime(date_format)


def get_time(time_format: str = r"%H:%M:%S") -> str:
    if time_format == "":
        raise ValueError("`time_format` must not be empty")
    return dt.datetime.now(dt.timezone.utc).strftime(time_format)


def get_datetime(
    date_format: str = r"%Y%m%d",
    time_format: str = r"%H%M%S",
    sep: str = "-",
    date_first: bool = True,
) -> str:
    """
    Get current date and time.

    Args:
        date_format (str, optional): Format string for date. Defaults to r"%Y%m%d".
        time_format (str, optional): Format string for time. Defaults to r"%H%M%S".
        sep (str, optional): Separator between date and time. Defaults to "-".
        date_first (bool, optional): Put date before time. Defaults to True.

    Returns:
        str: Current date and time
    """

    now = dt.datetime.now(dt.timezone.utc)
    if date_first:
        return now.strftime(f"{date_format}{sep}{time_format}")
    else:
        return now.strftime(f"{time_format}{sep}{date_format}")


def bytes_to_base64(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")


def base64_to_bytes(s: str) -> bytes:
    return base64.b64decode(s)


def format_error(error: BaseException) -> tuple[str, str]:
    import traceback

    return (
        f"{error.__class__.__module__}.{error.__class__.__name__}: {str(error)}",
        traceback.format_exc(),
    )


def generate_random_hash(max_length: Optional[int] = None) -> str:
    import hashlib

    return hashlib.sha3_256(os.urandom(256)).hexdigest()[:max_length]


@validate_call
def is_cuda(device: str) -> bool:
    return device.lower().startswith("cuda")


@validate_call
def has_cuda() -> bool:
    return False if torch is None else torch.cuda.is_available()


@validate_call
def has_mps() -> bool:
    return False if torch is None else torch.backends.mps.is_available()


@validate_call
def get_device(mps_ok: bool = False) -> tuple[str, Optional[str]]:
    if has_cuda():
        try:
            return "cuda", torch.cuda.get_device_name()
        except Exception:
            return "cuda", None
    if mps_ok and has_mps():
        return "mps", None
    return "cpu", None


@validate_call
def finalize_device(device: Optional[str]) -> str:
    if device is None:
        device, _ = get_device()
    elif is_cuda(device) and not has_cuda():
        warnings.warn(f"'{device}' is not available, using CPU instead")
        device = "cpu"
    return device


def poly2bbox(poly: Iterable[float]) -> tuple[float, float, float, float]:
    poly = list(poly)
    xs, ys = poly[::2], poly[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def replace_non_alphanumeric(text: str, new: str = "_") -> str:
    return re.sub(r"[^a-zA-Z0-9]", new, text.strip())


def tsprint(
    *values: object, sep: Optional[str] = " ", end: Optional[str] = "\n"
) -> None:
    from tqdm import tqdm

    with tqdm.external_write_mode(nolock=False):
        sys.stdout.write(sep.join(map(str, values)))
        sys.stdout.write(end)
        sys.stdout.flush()


@validate_call
def loading(desc: str = "Loading...", disable: bool = False):
    from . import console

    return console.status(desc) if not disable else nullcontext()
