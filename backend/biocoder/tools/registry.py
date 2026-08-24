from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from biocoder.security.permissions import PermissionGate
from biocoder.tools.schema import ToolMetadata


class ToolRegistry:
    def __init__(self, permission_gate: PermissionGate | None = None) -> None:
        self.permission_gate = permission_gate or PermissionGate()
        self._tools: dict[str, BaseTool] = {}
        self._metadata: dict[str, ToolMetadata] = {}

    def register(self, tool: BaseTool, metadata: ToolMetadata) -> None:
        if tool.name != metadata.name:
            raise ValueError(f"Tool name mismatch: {tool.name} != {metadata.name}")
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        self._metadata[tool.name] = metadata

    def get(self, name: str) -> BaseTool:
        try:
            metadata = self._metadata[name]
            tool = self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc
        self.permission_gate.check(metadata.permission)
        return tool

    def metadata(self, name: str) -> ToolMetadata:
        if name not in self._metadata:
            raise KeyError(f"Unknown tool: {name}")
        return self._metadata[name]

    def tools(self) -> list[BaseTool]:
        return [self.get(name) for name in self._tools]

    def schemas(self) -> list[dict[str, Any]]:
        return [metadata.model_dump(mode="json") for metadata in self._metadata.values()]
