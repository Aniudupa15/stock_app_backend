"""The seeded default-user UUID (migration 0006). Real per-user auth landed
in Phase 5 (see app/core/security.py, app/services/auth_service.py) - this
constant now exists only as a stable, real FK target for repository-level
tests that don't go through the API/auth layer at all.
"""

import uuid

DEFAULT_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
