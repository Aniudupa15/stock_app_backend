from app.core.auth import DEFAULT_USER_ID
from app.repositories.search_history_repository import SqlAlchemySearchHistoryRepository
from tests.conftest import requires_docker

pytestmark = requires_docker


async def test_log_and_list_returns_most_recent_first(db_session):
    repo = SqlAlchemySearchHistoryRepository(db_session)

    await repo.log(DEFAULT_USER_ID, "RELIANCE")
    await repo.log(DEFAULT_USER_ID, "TCS")

    entries = await repo.list_for_user(DEFAULT_USER_ID, limit=10, offset=0)

    assert [e.query for e in entries] == ["TCS", "RELIANCE"]


async def test_list_respects_limit_and_offset(db_session):
    repo = SqlAlchemySearchHistoryRepository(db_session)
    for symbol in ("A", "B", "C"):
        await repo.log(DEFAULT_USER_ID, symbol)

    page = await repo.list_for_user(DEFAULT_USER_ID, limit=1, offset=1)

    assert [e.query for e in page] == ["B"]


async def test_clear_for_user_deletes_all_and_returns_count(db_session):
    repo = SqlAlchemySearchHistoryRepository(db_session)
    await repo.log(DEFAULT_USER_ID, "RELIANCE")
    await repo.log(DEFAULT_USER_ID, "TCS")

    cleared = await repo.clear_for_user(DEFAULT_USER_ID)

    assert cleared == 2
    assert await repo.list_for_user(DEFAULT_USER_ID, limit=10, offset=0) == []
