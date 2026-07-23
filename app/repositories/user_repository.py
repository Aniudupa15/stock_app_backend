import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities import User
from app.domain.ports import UserRepositoryPort
from app.models.user import UserModel


def _to_entity(row: UserModel) -> User:
    return User(
        id=row.id,
        email=row.email,
        display_name=row.display_name,
        password_hash=row.password_hash,
        created_at=row.created_at,
    )


class SqlAlchemyUserRepository(UserRepositoryPort):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_email(self, email: str) -> User | None:
        stmt = select(UserModel).where(UserModel.email == email.strip().lower())
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        return _to_entity(row) if row else None

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        stmt = select(UserModel).where(UserModel.id == user_id)
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        return _to_entity(row) if row else None

    async def create(self, email: str, password_hash: str, display_name: str) -> User:
        model = UserModel(email=email.strip().lower(), password_hash=password_hash, display_name=display_name)
        self._session.add(model)
        await self._session.commit()
        await self._session.refresh(model)
        return _to_entity(model)

    async def update(self, user_id: uuid.UUID, display_name: str | None, email: str | None) -> User:
        stmt = select(UserModel).where(UserModel.id == user_id)
        result = await self._session.execute(stmt)
        model = result.scalar_one()

        if display_name is not None:
            model.display_name = display_name
        if email is not None:
            model.email = email.strip().lower()

        await self._session.commit()
        await self._session.refresh(model)
        return _to_entity(model)
