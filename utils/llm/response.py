from typing import Annotated, Optional, get_args

from pydantic import BaseModel, Field, validate_call
from typing_extensions import Self

ZERO_SHOT_COT_REASONING = '"Let\'s think step by step. <<FILL_IN>>"'


class ResponseModelParsingError(RuntimeError):
    pass


class ResponseModel(BaseModel, validate_assignment=True, strict=True):
    @classmethod
    def to_str(cls) -> str:
        raise NotImplementedError()

    @classmethod
    def from_str(cls, text: str) -> Self:
        raise NotImplementedError()


class JsonResponseModel(ResponseModel):
    @classmethod
    @validate_call
    def to_str(
        cls,
        _indent: Annotated[int, Field(ge=0)] = 2,
        _depth: Annotated[int, Field(ge=1)] = 1,
        **substitutions: str,
    ) -> str:
        def get_fill(ann, depth: int, name: Optional[str] = None) -> str:
            if name is not None and name in substitutions:
                return substitutions[name]
            if issubclass(ann, dict):
                raise RuntimeError("Use JsonResponseModel for dict fields")
            args = get_args(ann)
            if len(args) > 0:
                fill = "[\n"
                for i, arg in enumerate(args):
                    fill += " " * (_indent * (depth + 1))
                    fill += get_fill(arg, depth + 1)
                    fill += f"{',' if len(args) == 1 or i < len(args) - 1 else ''}\n"
                if len(args) == 1:
                    fill += " " * (_indent * (depth + 1)) + "...\n"
                fill += " " * (_indent * depth) + "]"
                return fill
            elif issubclass(ann, JsonResponseModel):
                return ann.to_str(
                    _depth=depth + 1, _indent=_indent, **substitutions
                )
            elif issubclass(ann, bool):
                return "true/false"
            elif issubclass(ann, int):
                return "<<FILL_IN_INTEGER>>"
            elif issubclass(ann, float):
                return "<<FILL_IN_FLOAT>>"
            elif issubclass(ann, str):
                return '"<<FILL_IN_STRING>>"'
            else:
                raise RuntimeError(f"Unsupported data type: {ann}")

        output = "{\n"
        for i, (name, field) in enumerate(cls.model_fields.items()):
            ann = field.annotation
            args = get_args(ann)
            if len(args) == 2:
                if issubclass(args[0], type(None)):
                    ann = args[1]
                elif issubclass(args[1], type(None)):
                    ann = args[0]
            output += " " * (_indent * _depth)
            output += f'"{name}": '
            output += get_fill(ann, _depth, name=name)
            output += f"{',' if i < len(cls.model_fields) - 1 else ''}\n"
        output += " " * (_indent * (_depth - 1)) + "}"
        return output

    @classmethod
    @validate_call
    def from_str(cls, text: str) -> Self:
        from .parser import JsonParser

        parsed = JsonParser()(text)
        if parsed is None:
            raise ResponseModelParsingError(
                "No JSON object could be extracted"
            )
        try:
            return cls.model_validate(parsed)
        except Exception as e:
            raise ResponseModelParsingError(
                f"Failed to parse response into {cls.__name__}: {str(e)}"
            ) from e
