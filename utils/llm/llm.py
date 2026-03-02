import logging
import random
import time
from copy import deepcopy
from datetime import datetime, timezone
from typing import (
    Annotated,
    Any,
    Generic,
    Literal,
    Optional,
    Sequence,
    TypeVar,
    Union,
    overload,
)
from uuid import uuid4

import numpy as np
import torch
from openai import (
    DEFAULT_MAX_RETRIES,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionTokenLogprob
from pydantic import BaseModel, Field, HttpUrl, validate_call
from rich.panel import Panel
from typing_extensions import Self

from .. import console
from ..anybase import AnyBase
from ..dtypes import UNSET, ImgLike, NonEmptyStr
from ..misc import format_error
from .msg import Messages
from .response import ResponseModel, ResponseModelParsingError
from .template import PromptTemplate

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

ResponseType = TypeVar("ResponseType", bound=Union[ResponseModel, str])

logging.getLogger("httpx").setLevel(logging.WARNING)


class ChatParameters(BaseModel, validate_assignment=True, strict=True):
    model: NonEmptyStr
    max_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    timeout: Optional[Annotated[float, Field(gt=0.0)]] = None
    max_retries: Optional[Annotated[int, Field(ge=0)]] = None
    base_url: Optional[HttpUrl] = None
    temperature: Optional[Annotated[float, Field(ge=0.0, le=2.0)]] = None
    image_detail: Optional[Literal["auto", "low", "high"]] = None
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[Annotated[int, Field(ge=0, le=5)]] = None
    extra_body: Optional[dict[str, Any]] = None

    def to_api_params(self) -> dict[str, Any]:
        return self.model_dump(
            exclude={"timeout", "max_retries", "base_url", "image_detail"},
            exclude_none=True,
        )


class LlmOutput(
    BaseModel, Generic[ResponseType], validate_assignment=True, strict=True
):
    id: NonEmptyStr = Field(default_factory=lambda: uuid4().hex, frozen=True)
    created_at: NonEmptyStr = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            r"%Y/%m/%d %H:%M:%S"
        ),
        frozen=True,
    )
    parameters: ChatParameters
    response: Optional[ResponseType] = None
    detail: Optional[NonEmptyStr] = None
    duration: Optional[Annotated[float, Field(gt=0.0)]] = None
    finish_reason: Optional[NonEmptyStr] = None
    prompt_tokens: Optional[Annotated[int, Field(ge=0)]] = None
    completion_tokens: Optional[Annotated[int, Field(ge=0)]] = None
    cost: Optional[Annotated[float, Field(ge=0.0)]] = None
    logprobs: Optional[list[ChatCompletionTokenLogprob]] = None
    messages: Optional[Messages] = None


class LlmChatError(RuntimeError):
    pass


class OpenAiApi(AnyBase):
    @validate_call
    def __init__(
        self,
        model: NonEmptyStr,
        *,
        timeout: Optional[Annotated[float, Field(gt=0.0)]] = None,
        max_retries: Optional[Annotated[int, Field(ge=0)]] = None,
        api_key: Optional[NonEmptyStr] = None,
        base_url: Optional[NonEmptyStr] = None,
        input_cost: Optional[float] = None,
        output_cost: Optional[float] = None,
    ) -> None:
        self._model = model
        self._timeout = timeout
        self._max_retries = (
            DEFAULT_MAX_RETRIES if max_retries is None else max_retries
        )
        self._api_key = api_key
        self._base_url = base_url
        self._input_cost = input_cost
        self._output_cost = output_cost
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=self._max_retries,
        )
        self._input_cost_per_token = self._normalize_cost(input_cost)
        self._output_cost_per_token = self._normalize_cost(output_cost)
        self._cost = 0.0

    def __call__(self):
        raise NotImplementedError()

    @property
    def cost(self) -> float:
        return self._cost

    def _normalize_cost(self, cost: Optional[float]) -> Optional[float]:
        if cost is None:
            return None
        return cost / 1_000_000


class Llm(OpenAiApi):
    @validate_call
    def __init__(
        self,
        model: str,
        *,
        max_tokens: Annotated[int, Field(ge=1)] = 4096,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        input_cost: Optional[float] = None,
        output_cost: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            api_key=api_key,
            base_url=base_url,
            input_cost=input_cost,
            output_cost=output_cost,
        )
        self._max_tokens = max_tokens
        self._kwargs = kwargs
        self._messages = Messages()
        self._history: list[LlmOutput] = []
        self._last_sys_pmt: Union[Optional[NonEmptyStr], object] = UNSET

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __call__(
        self,
        prompt: Union[NonEmptyStr, PromptTemplate],
        response_type: type[ResponseType] = str,
        *,
        sys_prompt: Optional[NonEmptyStr] = DEFAULT_SYSTEM_PROMPT,
        temperature: Optional[Annotated[float, Field(ge=0.0, le=2.0)]] = None,
        images: Optional[Sequence[ImgLike]] = None,
        image_detail: Literal["auto", "low", "high"] = "auto",
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[Annotated[int, Field(ge=0, le=5)]] = None,
        extra_body: Optional[dict[str, Any]] = None,
        base_delay: Annotated[float, Field(ge=1.0)] = 1.5,
        verbose: bool = True,
    ) -> LlmOutput[ResponseType]:
        rea_eff = (
            self._kwargs.get("reasoning_effort", None)
            if reasoning_effort is None
            else reasoning_effort
        )
        chat_parameters = ChatParameters(
            model=self._model,
            max_tokens=self._max_tokens,
            timeout=self._timeout,
            max_retries=self._max_retries,
            base_url=self._base_url,
            temperature=temperature,
            image_detail=image_detail,
            reasoning_effort=rea_eff,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            extra_body=extra_body,
        )

        # Dynamic system prompt
        if sys_prompt is not None:
            self._messages.set_system(sys_prompt)
        else:
            self._messages.remove_system()
        if self._last_sys_pmt is UNSET or sys_prompt != self._last_sys_pmt:
            self._last_sys_pmt = sys_prompt
            if verbose:
                console.print(
                    Panel(
                        "None" if sys_prompt is None else sys_prompt,
                        title="[not italic]System Prompt[/]",
                        title_align="left",
                        style="italic" if sys_prompt is None else "none",
                        border_style="grey53",
                    )
                )

        # User prompt
        final_pmt = prompt() if isinstance(prompt, PromptTemplate) else prompt
        if verbose:
            console.print(
                Panel(
                    final_pmt,
                    title="User Prompt",
                    title_align="left",
                    border_style="blue",
                )
            )
        self._messages.add_user(
            final_pmt, images=images, image_detail=image_detail
        )

        llm_output = None
        for r in range(self._max_retries + 1):
            start_time = time.perf_counter()
            llm_output = LlmOutput[ResponseType](parameters=chat_parameters)
            self._history.append(llm_output)
            delay = 0.0
            try:
                with console.status("Generating response..."):
                    completion: ChatCompletion = (
                        self._client.chat.completions.create(
                            messages=self._messages.to_api_format(),
                            stream=False,
                            **chat_parameters.to_api_params(),
                        )
                    )
                if len(completion.choices) == 0:
                    raise LlmChatError("LLM returned 0 choice")
                choice = completion.choices[0]
                asst_msg = choice.message.content
                if asst_msg is None:
                    raise LlmChatError("LLM response is None")
                response = (
                    response_type.from_str(asst_msg)
                    if issubclass(response_type, ResponseModel)
                    else asst_msg
                )
                if logprobs and choice.logprobs.content is None:
                    raise LlmChatError("LLM logprobs is None")
            except (
                InternalServerError,
                LlmChatError,
                RateLimitError,
                ResponseModelParsingError,
            ) as e:
                self._print_error(e)
                llm_output.detail = format_error(e)[0]
                if r == self._max_retries:
                    self._messages.remove_last()
                    raise
                delay = base_delay**r + random.random()
                time.sleep(delay)
                continue
            except Exception as e:
                self._print_error(e)
                llm_output.detail = format_error(e)[0]
                self._messages.remove_last()
                raise
            else:
                if verbose:
                    console.print(
                        Panel(
                            asst_msg,
                            title="LLM Response",
                            title_align="left",
                            border_style="green",
                        )
                    )
                self._messages.add_assistant(asst_msg)
                llm_output.response = response
                llm_output.finish_reason = choice.finish_reason
                assert completion.usage is not None
                llm_output.prompt_tokens = completion.usage.prompt_tokens
                llm_output.completion_tokens = (
                    completion.usage.completion_tokens
                )

                # Calculate cost
                cost = 0.0
                if (
                    self._input_cost_per_token is not None
                    and llm_output.prompt_tokens is not None
                ):
                    cost += (
                        llm_output.prompt_tokens * self._input_cost_per_token
                    )
                if (
                    self._output_cost_per_token is not None
                    and llm_output.completion_tokens is not None
                ):
                    cost += (
                        llm_output.completion_tokens
                        * self._output_cost_per_token
                    )
                llm_output.cost = cost
                self._cost += cost

                llm_output.logprobs = (
                    choice.logprobs.content
                    if choice.logprobs is not None
                    else None
                )
                llm_output.messages = deepcopy(self._messages)
                break
            finally:
                llm_output.duration = time.perf_counter() - start_time - delay

        assert llm_output is not None
        return llm_output

    @property
    def max_retries(self) -> int:
        return self._max_retries

    @property
    def history(self) -> list[LlmOutput]:
        return deepcopy(self._history)

    def clear_context(self) -> None:
        self._messages.clear()

    def replicate(self) -> Self:
        return Llm(
            model=self._model,
            max_tokens=self._max_tokens,
            timeout=self._timeout,
            max_retries=self._max_retries,
            api_key=self._api_key,
            base_url=self._base_url,
            input_cost=self._input_cost,
            output_cost=self._output_cost,
            **self._kwargs,
        )

    def _print_error(self, err: BaseException) -> None:
        console.print(
            Panel(
                format_error(err)[0],
                title="Error",
                title_align="left",
                border_style="red",
            )
        )


class TextEmbedder(OpenAiApi):
    def __init__(
        self,
        model: str,
        *,
        max_retries: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        input_cost: Optional[float] = None,
    ) -> None:
        super().__init__(
            model=model,
            max_retries=max_retries,
            api_key=api_key,
            base_url=base_url,
            input_cost=input_cost,
        )

    @overload
    def __call__(
        self, text: str, response_type: type[torch.Tensor]
    ) -> torch.Tensor: ...

    @overload
    def __call__(
        self, text: str, response_type: type[np.ndarray]
    ) -> np.ndarray: ...

    @overload
    def __call__(
        self, text: str, response_type: type[list]
    ) -> list[float]: ...

    @overload
    def __call__(
        self, text: str, response_type: None = None
    ) -> list[float]: ...

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __call__(
        self,
        text: str,
        response_type: Optional[
            Union[type[torch.Tensor], type[np.ndarray], type[list]]
        ] = None,
    ) -> Union[torch.Tensor, np.ndarray, list]:
        response = self._client.embeddings.create(
            model=self._model, input=text
        )
        if self._input_cost_per_token is not None:
            self._cost += (
                response.usage.total_tokens * self._input_cost_per_token
            )
        assert len(response.data) > 0
        emb = response.data[0].embedding
        if response_type is None or response_type is list:
            return emb
        elif response_type is torch.Tensor:
            return torch.tensor(emb)
        elif response_type is np.ndarray:
            return np.array(emb)
        else:
            raise ValueError(f"Unsupported response_type: {response_type}")


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("-t", "--temperature", type=float, default=None)
    parser.add_argument("--img", action="append", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    args = parser.parse_args()
    llm = Llm(args.model, api_key=args.api_key, base_url=args.base_url)
    llm_output = llm(
        args.prompt, temperature=args.temperature, images=args.img
    )
    print(llm_output.response)
