from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import DATABASE_URL, DATABASE_URL_IA


engine = create_async_engine(DATABASE_URL, echo=False, future=True)
engine_ia = create_async_engine(DATABASE_URL_IA, echo=False, future=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
async_session_ia = sessionmaker(engine_ia, class_=AsyncSession, expire_on_commit=False)

@asynccontextmanager
async def get_session(database_url: str = DATABASE_URL) -> AsyncSession:
    if database_url == DATABASE_URL:
        async with async_session() as session:
            yield session
    elif database_url == DATABASE_URL_IA:
        async with async_session_ia() as session:
            yield session