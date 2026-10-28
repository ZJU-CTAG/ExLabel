"""
Launcher for the external coding agent CLIs (claude code / codex) used by the
coding agent baseline.

Each call starts one fresh non-interactive agent session rooted in a
per-instance playground worktree, feeds the task prompt via stdin (prompts
embed full patches, which can exceed the Windows command-line length limit),
and returns the agent's final message for JSON parsing by the labeller.

Only read-only exploration is requested: claude runs with `--restricted`
(command-running and file-editing tools removed), codex runs with its default
exec policy — no sandbox override — and the disposable worktree contains any
accidents either way.
"""

import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _resolve_cli(name: str) -> List[str]:
    """
    Resolve a CLI name to an argv prefix.

    Windows npm shims are .cmd files that CreateProcess cannot execute
    directly, so they are routed through `cmd /c`.
    """
    resolved = shutil.which(name)
    if resolved is None:
        raise RuntimeError(f"CLI not found on PATH: {name}")
    if os.name == "nt" and resolved.lower().endswith((".cmd", ".bat")):
        return ["cmd", "/c", resolved]
    return [resolved]


def _write_output_schema(path: Path, label_field: str):
    """Write the JSON Schema constraining codex's final response."""
    schema = {
        "type": "object",
        "properties": {
            label_field: {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": [label_field, "reason"],
        "additionalProperties": False,
    }
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")


def build_coding_agent_command(
    agent: str,
    workdir: str,
    label_field: str,
    model: Optional[str] = None,
    out_dir: Optional[str] = None,
    instance_id: str = "instance",
) -> List[str]:
    """
    Build the command line for one non-interactive agent session.

    The task prompt itself is always passed via stdin, never as an argument.
    """
    if agent == "claude":
        cmd = [
            *_resolve_cli("claude"),
            "-p",
            "--output-format",
            "json",
            "--restricted",
            # Prevent Claude Code from spawning any subagent.
            "--disallowedTools",
            "Agent",
        ]
        effort = os.environ.get("CLAUDE_EFFORT", "xhigh")
        if effort:
            cmd += ["--effort", effort]
        if model:
            cmd += ["--model", model]
    elif agent == "codex":
        out_path = Path(out_dir or ".").resolve()
        base_url = os.environ.get("CODEX_BASE_URL", "http://localhost:8888/v1")
        reasoning_effort = os.environ.get("CODEX_REASONING_EFFORT", "xhigh")
        cmd = [
            *_resolve_cli("codex"), "exec", "--skip-git-repo-check",
            "-c", 'model_provider="vllm"',
            "-c", 'model_providers.vllm.name="vllm"',
            "-c", f'model_providers.vllm.base_url="{base_url}"',
            "-c", 'model_providers.vllm.wire_api="responses"',
            "-c", "model_providers.vllm.requires_openai_auth=false",
            "-c", 'model_providers.vllm.env_key="VLLM_API_KEY"',
            "-c", f'model_reasoning_effort="{reasoning_effort}"',
            "-C", workdir,
            "-o", str(out_path / f"{instance_id}.last_message.txt"),
            "--json",
            "-",
        ]
        if model:
            cmd += ["-m", model]
    else:
        raise ValueError(f"Unknown agent: {agent!r} (expected claude or codex)")
    return cmd


def _kill_process_tree(proc: asyncio.subprocess.Process):
    """Best-effort kill of the agent process and its entire process group.

    The agent CLI (e.g. codex) spawns its own children; killing only the
    direct child leaks orphans that hang forever on a disconnected stream.
    Kill the whole process group (requires start_new_session=True).
    """
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
            )
        except Exception:
            pass
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _extract_claude_result(stdout_text: str, instance_id: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Extract the final message and token usage from the `claude -p --output-format json` envelope."""
    envelope = None
    try:
        envelope = json.loads(stdout_text)
    except json.JSONDecodeError:
        # Tolerate stray non-JSON lines around the envelope.
        for line in reversed(stdout_text.splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and "result" in candidate:
                envelope = candidate
                break
    if not isinstance(envelope, dict):
        raise RuntimeError(f"[{instance_id}] Could not parse the claude output envelope")
    if envelope.get("is_error"):
        raise RuntimeError(
            f"[{instance_id}] claude agent reported an error: {str(envelope.get('result'))[:500]}"
        )
    result = envelope.get("result")
    if not isinstance(result, str) or not result.strip():
        raise RuntimeError(
            f"[{instance_id}] claude agent returned no result (subtype={envelope.get('subtype')!r})"
        )
    usage = envelope.get("usage")
    return result, (usage if isinstance(usage, dict) else None)


def _extract_codex_message(stdout_text: str, out_dir: Path, instance_id: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Extract codex's final message and token usage from the `-o` file / event stream."""
    # Walk the --json event stream once: collect agent_message texts and the
    # turn.completed usage summary.
    messages = []
    usage = None
    for line in stdout_text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "turn.completed":
            u = event.get("usage")
            if isinstance(u, dict):
                usage = u
        item = event.get("item") or {}
        if event.get("type") == "item.completed" and item.get("type") == "agent_message":
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                messages.append(text)

    # Preferred source: the file written via -o/--output-last-message.
    last_msg_file = out_dir / f"{instance_id}.last_message.txt"
    if last_msg_file.exists():
        text = last_msg_file.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            return text, usage
    if not messages:
        raise RuntimeError(
            f"[{instance_id}] Could not extract the codex final message "
            f"(no last-message file and no agent_message events in stdout)"
        )
    return messages[-1], usage


async def run_coding_agent(
    agent: str,
    label_field: str,
    prompt: str,
    workdir: str,
    instance_id: str,
    out_dir: Optional[str] = None,
    model: Optional[str] = None,
    timeout_seconds: float = 900.0,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Run one non-interactive coding agent session and return its final message.

    Args:
        agent: 'claude' or 'codex'
        label_field: Task label ('underspecified' or 'reasonable'); used by the
            codex output schema
        prompt: Full task prompt (passed via stdin)
        workdir: Working directory for the agent (a playground worktree)
        instance_id: Sample identifier for logging and output artifacts
        out_dir: Directory for codex last-message/schema artifacts
        model: Optional model override for the agent CLI
        timeout_seconds: Kill the agent if it runs longer than this

    Raises:
        RuntimeError: On timeout, non-zero exit, or missing output, so the
            caller's retry flow can re-run the instance.
    """
    out_path = Path(out_dir or ".")
    out_path.mkdir(parents=True, exist_ok=True)

    cmd = build_coding_agent_command(
        agent,
        workdir=str(workdir),
        label_field=label_field,
        model=model,
        out_dir=str(out_path),
        instance_id=instance_id,
    )

    logging.info(f"[{instance_id}] Launching {agent} agent (model={model or 'CLI default'}, timeout={timeout_seconds:.0f}s)")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(workdir),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(prompt.encode("utf-8")),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        _kill_process_tree(proc)
        await proc.wait()
        raise RuntimeError(
            f"[{instance_id}] {agent} agent timed out after {timeout_seconds:.0f}s"
        )

    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")

    if proc.returncode != 0:
        raise RuntimeError(
            f"[{instance_id}] {agent} agent exited with code {proc.returncode}: "
            f"{stderr_text.strip()[-2000:]}"
        )

    if agent == "claude":
        return _extract_claude_result(stdout_text, instance_id)
    return _extract_codex_message(stdout_text, out_path, instance_id)
