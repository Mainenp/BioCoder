from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from biocoder.state import utc_now


class FeedbackType(StrEnum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"
    TEXT = "text_feedback"
    CORRECTION = "corrected_answer"


class FeedbackRequest(BaseModel):
    task_id: str = Field(min_length=1, max_length=100)
    feedback_type: FeedbackType
    rating: int | None = Field(default=None, ge=1, le=5)
    text_feedback: str | None = Field(default=None, max_length=8000)
    corrected_answer: str | None = Field(default=None, max_length=30000)

    @model_validator(mode="after")
    def validate_payload(self) -> FeedbackRequest:
        if self.feedback_type == FeedbackType.RATING and self.rating is None:
            raise ValueError("rating is required for rating feedback")
        if self.feedback_type == FeedbackType.TEXT and not self.text_feedback:
            raise ValueError("text_feedback is required for text feedback")
        if self.feedback_type == FeedbackType.CORRECTION and not self.corrected_answer:
            raise ValueError("corrected_answer is required for correction feedback")
        return self


class FeedbackRecord(FeedbackRequest):
    feedback_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def is_negative(self) -> bool:
        return self.feedback_type == FeedbackType.THUMBS_DOWN or (
            self.feedback_type == FeedbackType.RATING and self.rating is not None and self.rating <= 2
        )

    @property
    def is_positive(self) -> bool:
        return self.feedback_type == FeedbackType.THUMBS_UP or (
            self.feedback_type == FeedbackType.RATING and self.rating is not None and self.rating >= 4
        )

    @property
    def is_score(self) -> bool:
        return self.feedback_type in {
            FeedbackType.THUMBS_UP,
            FeedbackType.THUMBS_DOWN,
            FeedbackType.RATING,
        }
