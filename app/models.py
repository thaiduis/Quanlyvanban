from __future__ import annotations

from sqlalchemy import Column, Integer, String, Text, DateTime, func
from .database import Base


class Document(Base):
	__tablename__ = "documents"

	id = Column(Integer, primary_key=True, index=True)
	title = Column(String(255), nullable=False, index=True)
	filename = Column(String(512), nullable=False)
	content = Column(Text, nullable=True)
	tags = Column(String(255), nullable=True, index=True)
	mime_type = Column(String(100), nullable=True)
	filesize = Column(Integer, nullable=True)
	created_at = Column(DateTime, server_default=func.now(), nullable=False)
	updated_at = Column(
		DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
	)
