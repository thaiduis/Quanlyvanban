from __future__ import annotations

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from dotenv import load_dotenv

load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_USER = os.getenv("MYSQL_USER", "docuser")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "StrongPass123!")
MYSQL_DB = os.getenv("MYSQL_DB", "docmgr")

DATABASE_URL = (
	f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
	f"?charset=utf8mb4"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
	pass


def get_db():
	db = SessionLocal()
	try:
		yield db
	except Exception:
		db.rollback()
		raise
	finally:
		db.close()


def init_db() -> None:
	from . import models  # noqa: F401
	Base.metadata.create_all(bind=engine)


def ensure_fulltext_indexes() -> None:
	# MySQL 8 InnoDB FULLTEXT on title, content, tags
	ddl = text(
		"""
		ALTER TABLE documents
		ADD FULLTEXT INDEX ft_title_content_tags (title, content, tags);
		"""
	)
	with engine.connect() as conn:
		try:
			conn.execute(ddl)
			conn.commit()
		except Exception:
			# Index có thể đã tồn tại -> bỏ qua
			conn.rollback()
