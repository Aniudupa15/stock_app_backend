import uuid

from app.services.search_history_service import SearchHistoryService
from tests.conftest import FakeSearchHistoryRepository

USER_ID = uuid.uuid4()
OTHER_USER_ID = uuid.uuid4()


async def test_log_best_effort_records_query():
    repo = FakeSearchHistoryRepository()
    service = SearchHistoryService(repo)

    await service.log_best_effort(USER_ID, "RELIANCE")

    entries = await service.list(USER_ID, limit=10, offset=0)
    assert len(entries) == 1
    assert entries[0].query == "RELIANCE"


async def test_log_best_effort_swallows_repository_errors():
    class BrokenRepository(FakeSearchHistoryRepository):
        async def log(self, user_id, query):
            raise RuntimeError("db exploded")

    service = SearchHistoryService(BrokenRepository())

    # Must not raise - a logging failure can never surface to the caller.
    await service.log_best_effort(USER_ID, "RELIANCE")


async def test_list_only_returns_own_history():
    repo = FakeSearchHistoryRepository()
    service = SearchHistoryService(repo)
    await service.log_best_effort(USER_ID, "RELIANCE")
    await service.log_best_effort(OTHER_USER_ID, "TCS")

    entries = await service.list(USER_ID, limit=10, offset=0)

    assert [e.query for e in entries] == ["RELIANCE"]


async def test_clear_removes_all_entries_for_user():
    repo = FakeSearchHistoryRepository()
    service = SearchHistoryService(repo)
    await service.log_best_effort(USER_ID, "RELIANCE")
    await service.log_best_effort(USER_ID, "TCS")

    cleared = await service.clear(USER_ID)

    assert cleared == 2
    assert await service.list(USER_ID, limit=10, offset=0) == []
