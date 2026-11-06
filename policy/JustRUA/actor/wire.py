"""Translate the Direct EEF chat contract to incremental Codex response items."""
import base64
import hashlib
import json
import re

from XPolicyLab.utils.openai_responses import chat_messages_to_responses_input

from .runtime import MODEL

ROBOT_TOOLS = {"move_eef", "give_up"}
INSTRUCTIONS = """Use native coding tools to inspect the supplied observations, calculate geometry and
keep notes in your case scratch directory. Observations are read-only copies of exactly the text
and images supplied in this conversation. Use only these observations, your scratch work and
installed generic libraries. Web, network, other cases, simulator internals and other agents are
out of bounds. Use login=false when running shell tools. Files persist between decisions. To
retain live Python or Node variables, keep a process running and use write_stdin in that session.
There is no decision-round limit. The simulator and its physical action limits govern termination.
Coding tools cannot move the robot. To request the next action return one JSON object with content
(text or null) and tool_calls (the ordered robot calls, each with name and JSON-encoded arguments).
Only the robot controller executes these calls. Use its measured feedback to assess the result.
Robot tool definitions:
"""


def digest(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def robot_contract(messages, tools):
    systems = [message["content"] for message in messages if message.get("role") == "system"]
    if not systems or any(not isinstance(text, str) for text in systems):
        raise ValueError("The Direct EEF system instructions are required")
    names = [tool["function"]["name"] for tool in tools if tool["type"] == "function"]
    if len(names) != len(tools) or len(names) != 2 or set(names) != ROBOT_TOOLS:
        raise ValueError("Expected the absolute move_eef and give_up robot tools")
    return "\n\n".join(systems), INSTRUCTIONS + json.dumps(tools, ensure_ascii=False)


def output_schema():
    string = {"type": "string"}
    call = {"type": "object", "additionalProperties": False, "required": ["name", "arguments"],
            "properties": {"name": {"type": "string", "enum": sorted(ROBOT_TOOLS)}, "arguments": string}}
    return {"type": "object", "additionalProperties": False, "required": ["content", "tool_calls"],
            "properties": {"content": {"type": ["string", "null"]}, "tool_calls": {"type": "array", "items": call}}}


def decode_answer(text, turn_id):
    answer = json.loads(text)
    if set(answer) != {"content", "tool_calls"} or not isinstance(answer["tool_calls"], list):
        raise ValueError("Malformed robot action envelope")
    if answer["content"] is not None and not isinstance(answer["content"], str):
        raise ValueError("Malformed assistant text")
    message = {"role": "assistant", "content": answer["content"]}
    calls = []
    for index, call in enumerate(answer["tool_calls"]):
        if (set(call) != {"name", "arguments"} or call["name"] not in ROBOT_TOOLS
                or not isinstance(call["arguments"], str)):
            raise ValueError("Malformed robot call")
        # The EEF controller owns argument validation, bimanual merging and repair.
        calls.append({"id": f"call_{turn_id}_{index}", "type": "function", "function": call})
    if calls:
        message["tool_calls"] = calls
    return {"id": turn_id, "model": MODEL, "choices": [{
        "finish_reason": "tool_calls" if calls else "stop", "message": message,
    }]}


def incremental_items(messages):
    fresh = []
    for message in messages:
        role = message["role"]
        if role == "system":
            continue
        if role not in ("user", "assistant", "tool"):
            raise ValueError(f"Unexpected policy message role: {role}")
        # Native assistant prose and opaque reasoning already belong to the thread.
        # Only the controller's actual function aliases are needed to attach tool feedback.
        fresh.append({**message, "content": None} if role == "assistant" else message)
    items = chat_messages_to_responses_input(fresh)
    for item in items:
        if "role" in item:
            item["type"] = "message"
            if isinstance(item["content"], str):
                item["content"] = [{"type": "input_text", "text": item["content"]}]
    return items


def write_inputs(directory, messages, tools):
    """Expose the same supplied bytes to native tools without giving them policy or simulator files."""
    directory.mkdir(mode=0o700)
    files = []

    def project(value):
        if isinstance(value, list):
            return [project(item) for item in value]
        if isinstance(value, dict):
            if value.get("type") == "image_url":
                image = value["image_url"]
                url = image["url"] if isinstance(image, dict) else image
                match = re.fullmatch(r"data:image/(png|jpe?g|webp);base64,(.+)", url, re.DOTALL)
                if not match:
                    raise ValueError("Expected an inline camera image from this observation")
                raw = base64.b64decode(match[2], validate=True)
                path = directory / f"image-{len(files):03d}.{match[1]}"
                path.write_bytes(raw)
                files.append(path)
                return {"type": "supplied_image", "path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
            return {key: project(item) for key, item in value.items()}
        return value

    record = {"messages": project(messages), "robot_tools": tools}
    path = directory / "observations.json"
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    for file in [*files, path]:
        file.chmod(0o400)
    return path
