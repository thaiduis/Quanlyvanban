from __future__ import annotations

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class DocumentBase(BaseModel):
	title: str
	tags: Optional[str] = None


class DocumentCreate(DocumentBase):
	pass


class DocumentOut(DocumentBase):
	id: int
	filename: str
	mime_type: Optional[str] = None
	filesize: Optional[int] = None
	created_at: datetime
	updated_at: datetime

	class Config:
		from_attributes = True


class SearchQuery(BaseModel):
	q: Optional[str] = None
	tags: Optional[str] = None
	title: Optional[str] = None
	page: int = 1
	limit: int = 10
