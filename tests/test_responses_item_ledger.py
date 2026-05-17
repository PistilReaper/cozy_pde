from __future__ import annotations

import json

from agent_runner.responses_items import (
    ResponsesFunctionCall,
    assistant_text,
    extract_function_calls,
    extract_output_text,
    function_call_output,
    response_to_ledger_items,
    system_text,
    user_text,
)


def test_responses_items_round_trip_messages_and_function_calls():
    response = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "planning complete"}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "write_file",
                "arguments": json.dumps(
                    {
                        "path": "submission/code/hello.py",
                        "content": "print('hello')\n",
                    }
                ),
            },
        ]
    }

    assert system_text("sys")["content"][0]["type"] == "input_text"
    assert user_text("user")["content"][0]["type"] == "input_text"
    assert assistant_text("assistant")["content"][0]["type"] == "output_text"
    assert extract_output_text(response) == "planning complete"
    assert extract_function_calls(response) == [
        ResponsesFunctionCall(
            name="write_file",
            arguments={
                "path": "submission/code/hello.py",
                "content": "print('hello')\n",
            },
            call_id="call_1",
        )
    ]

    ledger_items = response_to_ledger_items(response)
    assert [item["type"] for item in ledger_items] == ["message", "function_call"]
    assert ledger_items[1]["call_id"] == "call_1"

    result_item = function_call_output("call_1", {"ok": True, "path": "submission/code/hello.py"})
    assert result_item["type"] == "function_call_output"
    assert json.loads(result_item["output"])["ok"] is True
