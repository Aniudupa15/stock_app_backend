from app.repositories.user_repository import SqlAlchemyUserRepository
from tests.conftest import requires_docker

pytestmark = requires_docker


async def test_create_and_get_by_email(db_session):
    repo = SqlAlchemyUserRepository(db_session)

    created = await repo.create("New@Example.com", "hashed", "New User")

    assert created.email == "new@example.com"  # normalized to lowercase
    fetched = await repo.get_by_email("new@example.com")
    assert fetched is not None
    assert fetched.id == created.id


async def test_get_by_id(db_session):
    repo = SqlAlchemyUserRepository(db_session)
    created = await repo.create("user@example.com", "hashed", "User")

    fetched = await repo.get_by_id(created.id)

    assert fetched is not None
    assert fetched.email == "user@example.com"


async def test_get_by_email_unknown_returns_none(db_session):
    repo = SqlAlchemyUserRepository(db_session)

    assert await repo.get_by_email("nobody@example.com") is None
