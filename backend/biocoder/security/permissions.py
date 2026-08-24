from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ToolPermission(StrEnum):
    READ_ONLY = "READ_ONLY"
    WRITE = "WRITE"
    DANGEROUS = "DANGEROUS"


@dataclass(frozen=True, slots=True)
class PermissionGate:
    allow_write: bool = False
    allow_dangerous: bool = False

    def check(self, permission: ToolPermission) -> None:
        if permission == ToolPermission.WRITE and not self.allow_write:
            raise PermissionError("WRITE tool execution requires explicit permission")
        if permission == ToolPermission.DANGEROUS and not self.allow_dangerous:
            raise PermissionError("DANGEROUS tool execution requires explicit permission")
