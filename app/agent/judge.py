from typing import Literal

from pydantic import BaseModel


class JudgeVerdict(BaseModel):
    verdict: Literal["pass", "fail"]
    reason: str
