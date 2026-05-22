from __future__ import annotations


class ContextPacker:
    _SECTION_ORDER = (
        "developer_contract",
        "task_spec",
        "phase_tool_policy",
        "compact_state",
        "retrieved_memory",
        "current_request",
    )

    def __init__(self, max_chars: int = 4000) -> None:
        self.max_chars = max_chars

    def build(
        self,
        *,
        developer_contract: str,
        task_spec: str,
        phase_tool_policy: str,
        compact_state: str,
        retrieved_memory: str,
        current_request: str,
    ) -> list[dict[str, str]]:
        raw_sections = {
            "developer_contract": developer_contract,
            "task_spec": task_spec,
            "phase_tool_policy": phase_tool_policy,
            "compact_state": compact_state,
            "retrieved_memory": retrieved_memory,
            "current_request": current_request,
        }
        remaining = max(self.max_chars, 0)
        packed: list[dict[str, str]] = []

        for section_name in self._SECTION_ORDER:
            sections_left = len(self._SECTION_ORDER) - len(packed)
            content = raw_sections[section_name]
            if remaining <= 0:
                packed.append({"name": section_name, "content": ""})
                continue

            allocation = max(1, remaining // sections_left)
            truncated = content[:allocation]
            packed.append({"name": section_name, "content": truncated})
            remaining -= len(truncated)

        return packed

    def render_text(self, sections: list[dict[str, str]]) -> str:
        rendered: list[str] = []
        for section in sections[: len(self._SECTION_ORDER)]:
            name = str(section.get("name", "")).strip()
            content = str(section.get("content", ""))
            rendered.append(f"[{name}]\n{content}")
        return "\n\n".join(rendered)
