"""
generate_and_run.py
--------------------
Reads every TODO comment from a ClinicalIQ starter session folder,
uses the Groq LLM to generate the missing implementation, writes the
implementations back into the files, sets up any required databases /
vector stores, then launches the agent's interactive terminal loop.

Usage
-----
    # Auto-detect the lowest unimplemented session and run it
    python generate_and_run.py

    # Target a specific session (folder must contain a starter/ sub-folder)
    python generate_and_run.py --session s01
    python generate_and_run.py --session s02
    python generate_and_run.py --session s03
    python generate_and_run.py --session s04
    python generate_and_run.py --session s05

    # Only generate code, skip launching the agent
    python generate_and_run.py --session s01 --no-run

    # Regenerate even if the TODO is already resolved
    python generate_and_run.py --session s01 --force

How it works
------------
1. Discovers all Python files in sXX/starter/clinicaliq/ (or sXX/clinicaliq/).
2. For each file that still contains 'raise NotImplementedError' or a placeholder
   that matches a known TODO pattern, it sends the ENTIRE file to the LLM with an
   instruction to fill in only the missing implementation and return the complete
   file.
3. Writes the LLM response back to the file (after basic sanity checks).
4. Runs data/seed.py and data/ingest.py if the session requires them.
5. Launches the agent via `python -m clinicaliq.agent` in the session's starter dir.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Bootstrap: load .env so GROQ_API_KEY is available before we import langchain
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[generate_and_run] python-dotenv not installed. Run: pip install python-dotenv")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).parent
DATA_DIR      = ROOT / "data"
SEED_SCRIPT   = DATA_DIR / "seed.py"
INGEST_SCRIPT = DATA_DIR / "ingest.py"

# Sessions that need the SQLite clinic database seeded
NEEDS_SQLITE = {"s04", "s05"}
# Sessions that need the ChromaDB vector store built
NEEDS_CHROMA = {"s04", "s05"}

# Ordered list of session directories
ALL_SESSIONS = ["s01", "s02", "s03", "s04", "s05"]

# Pattern that marks a file as having unimplemented code
TODO_MARKERS = [
    r"raise NotImplementedError",
    r"TODO\s*\d+\s*of\s*\d+",
    r'pass\s*#\s*TODO',
    r'SYSTEM_PROMPT\s*=\s*"""\s*\nTODO',
]
TODO_RE = re.compile("|".join(TODO_MARKERS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# LLM client (Groq — same provider as the agents themselves)
# ---------------------------------------------------------------------------

# Ordered list of models to try for code generation.
# The script will test each in order at startup and use the first one that
# responds successfully, so it degrades gracefully as Groq's model catalogue
# changes over time.
_CODEGEN_MODEL_CANDIDATES = [
    "meta-llama/llama-4-scout-17b-16e-instruct",  # preferred — high quality
    "llama-3.3-70b-versatile",                     # strong fallback
    "qwen/qwen3.8-27b",                            # available on restricted keys
    "groq/compound",                               # broad availability fallback
    "qwen/qwen3.6-27b",                            # additional fallback
]


def _probe_model(api_key: str, model_id: str) -> bool:
    """Return True if `model_id` accepts a minimal chat completion request."""
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=3,
        )
        return True
    except Exception:
        return False


def _get_llm():
    """Return a ChatGroq instance wired to the best available code-generation model."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print(
            "[generate_and_run] GROQ_API_KEY not found in environment.\n"
            "  Make sure your .env file is populated and in the project root."
        )
        sys.exit(1)

    try:
        from langchain_groq import ChatGroq
    except ImportError:
        print("[generate_and_run] langchain-groq not installed. Run: pip install langchain-groq")
        sys.exit(1)

    print("  [model-probe] Finding best available Groq model for code generation ...")
    selected = None
    for candidate in _CODEGEN_MODEL_CANDIDATES:
        if _probe_model(api_key, candidate):
            selected = candidate
            print(f"  [model-probe] Using: {selected}")
            break

    if selected is None:
        print(
            "[generate_and_run] None of the candidate models responded successfully.\n"
            "  Models tried:\n"
            + "\n".join(f"    • {m}" for m in _CODEGEN_MODEL_CANDIDATES)
            + "\n  Check your GROQ_API_KEY and network connectivity."
        )
        sys.exit(1)

    return ChatGroq(
        api_key=api_key,
        model=selected,
        temperature=0.0,        # deterministic — we want exact code, not creativity
        max_tokens=4096,
    )


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------

def find_session_dir(session: str) -> Optional[Path]:
    """Return the clinicaliq package directory for a given session key."""
    base = ROOT / session
    if not base.exists():
        return None

    # Check starter/ sub-folder first (s02-s05 pattern)
    starter = base / "starter" / "clinicaliq"
    if starter.exists():
        return starter

    # Bare clinicaliq/ in session root (s01 pattern)
    bare = base / "clinicaliq"
    if bare.exists():
        return bare

    return None


def find_first_unimplemented_session() -> Optional[str]:
    """Walk sessions in order; return the first one that has a TODO stub."""
    for session in ALL_SESSIONS:
        pkg_dir = find_session_dir(session)
        if pkg_dir is None:
            continue
        for py_file in sorted(pkg_dir.glob("*.py")):
            if has_todos(py_file):
                return session
    return None


def has_todos(path: Path) -> bool:
    """Return True if the file still contains unimplemented stubs."""
    try:
        content = path.read_text(encoding="utf-8")
        return bool(TODO_RE.search(content))
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

GENERATION_SYSTEM = textwrap.dedent("""\
    You are an expert Python developer working on ClinicalIQ, a LangGraph-based
    healthcare chatbot built on top of the Groq API.

    You will receive a Python source file that has TODO comment blocks describing
    missing implementations. Some functions raise NotImplementedError or have
    placeholder bodies that need to be replaced.

    Your task:
    - Read every TODO comment block carefully.
    - Implement EXACTLY what the comment specifies — variable names, return shapes,
      error handling, and method calls must match the instructions precisely.
    - Do NOT remove any existing code, imports, docstrings, or comments.
    - Do NOT add new imports that aren't already present or clearly required by the
      TODO instructions.
    - Keep the file structure identical — only fill in the missing function bodies
      and string values.
    - The TODO comment blocks themselves should be kept in the output (don't strip them).
    - Return ONLY the complete Python file content — no markdown fences, no explanation,
      no prose before or after the code. Start your response with the first line of the
      Python file (usually a triple-quoted docstring or an import).
""")


def generate_implementation(llm, file_path: Path, session: str) -> str:
    """Send a stub file to the LLM and get back the completed implementation."""
    import time
    from langchain_core.messages import HumanMessage, SystemMessage

    source = file_path.read_text(encoding="utf-8")

    user_prompt = textwrap.dedent(f"""\
        Session: {session}
        File:    {file_path.name}

        Here is the complete source file. Please fill in all TODO implementations
        and return the complete file:

        ```python
        {source}
        ```
    """)

    print(f"    [LLM] Generating implementation for {file_path.name} ...")

    messages = [
        SystemMessage(content=GENERATION_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    # Retry up to 5 times with exponential backoff for rate limit errors
    max_retries = 5
    for attempt in range(max_retries):
        try:
            result = llm.invoke(messages)
            break
        except Exception as e:
            err_str = str(e)
            # Groq rate limit (429) — back off and retry
            if "429" in err_str or "rate_limit" in err_str.lower():
                # Parse the suggested wait time from the error message if present
                wait_match = re.search(r"try again in (\d+\.?\d*)s", err_str, re.IGNORECASE)
                wait_secs  = float(wait_match.group(1)) if wait_match else (2 ** (attempt + 1))
                wait_secs  = min(wait_secs + 2, 120)   # add 2s buffer; cap at 2 min
                print(f"    [rate-limit] Waiting {wait_secs:.0f}s before retry "
                      f"(attempt {attempt + 1}/{max_retries}) ...")
                time.sleep(wait_secs)
                if attempt == max_retries - 1:
                    raise
            else:
                raise

    raw = result.content.strip()

    # Strip any markdown code fence or <think> block the model may have added
    raw = _strip_code_fence(raw)

    return raw


def _strip_code_fence(text: str) -> str:
    """Remove ```python ... ``` or ``` ... ``` wrapping if present."""
    # Some models (e.g. Qwen in chain-of-thought mode) emit <think>...</think>
    # blocks before the actual output. Strip them.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Multi-line fence at start
    fence_start = re.match(r"^```(?:python)?\s*\n", text)
    if fence_start:
        text = text[fence_start.end():]
        # Remove trailing fence
        text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def sanity_check(original: str, generated: str, file_name: str) -> bool:
    """
    Basic sanity checks before overwriting the file.
    Returns True if the generated code looks safe to write.
    """
    # Must be non-trivially longer than a blank file
    if len(generated.strip()) < 50:
        print(f"    [WARN] {file_name}: Generated code is suspiciously short — skipping.")
        return False

    # Must contain at least some Python keywords
    keywords = {"def ", "class ", "return ", "import "}
    if not any(kw in generated for kw in keywords):
        print(f"    [WARN] {file_name}: Generated text doesn't look like Python — skipping.")
        return False

    # Warn if raise NotImplementedError is still present, but only for cases
    # where it appears as a *function body* (not as a runtime guard or test).
    # Heuristic: if the raise is preceded by a TODO comment it's an unfinished body.
    stub_pattern = re.compile(
        r"#\s*TODO.*\n.*raise NotImplementedError",
        re.DOTALL,
    )
    if stub_pattern.search(generated) and file_name not in ("tools.py",):
        print(f"    [WARN] {file_name}: Generated code may still have unimplemented stubs.")
        print("           The LLM may not have completed the implementation.")
        print("           Writing anyway — inspect the file if the agent crashes.")

    return True


# ---------------------------------------------------------------------------
# File-level orchestration
# ---------------------------------------------------------------------------

def process_file(llm, file_path: Path, session: str, force: bool) -> bool:
    """
    Generate + write implementation for a single file if it has TODOs.
    Returns True if the file was modified.
    """
    if not force and not has_todos(file_path):
        print(f"  ✓ {file_path.name:20s} — already implemented, skipping.")
        return False

    print(f"  ✎ {file_path.name:20s} — TODO markers found, generating ...")

    original  = file_path.read_text(encoding="utf-8")
    generated = generate_implementation(llm, file_path, session)

    if not sanity_check(original, generated, file_path.name):
        return False

    # Back up the original file next to itself before overwriting
    backup = file_path.with_suffix(".py.bak")
    backup.write_text(original, encoding="utf-8")
    print(f"    [backup] Original saved to {backup.name}")

    file_path.write_text(generated, encoding="utf-8")
    print(f"    [OK]    {file_path.name} updated.")
    return True


def process_session(session: str, force: bool) -> list[Path]:
    """
    Discover and fill in all stub files for a session.
    Returns the list of files that were modified.
    """
    pkg_dir = find_session_dir(session)
    if pkg_dir is None:
        print(f"[ERROR] Could not find a clinicaliq package directory for '{session}'.")
        print(f"        Expected: {ROOT / session / 'starter' / 'clinicaliq'}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Session : {session}")
    print(f"  Package : {pkg_dir}")
    print(f"{'='*60}\n")

    # Process in dependency order: __init__ → config → state → tools → nodes → agent
    file_order = ["__init__.py", "config.py", "state.py", "tools.py", "nodes.py", "agent.py"]
    all_py     = {p.name: p for p in pkg_dir.glob("*.py")}

    ordered_paths = []
    for name in file_order:
        if name in all_py:
            ordered_paths.append(all_py[name])
    # Any extra files not in the ordered list
    for name, path in sorted(all_py.items()):
        if name not in file_order:
            ordered_paths.append(path)

    # Build the LLM client once per session (model probe runs only once)
    llm      = _get_llm()
    modified = []

    for file_path in ordered_paths:
        changed = process_file(llm, file_path, session, force)
        if changed:
            modified.append(file_path)

    return modified


# ---------------------------------------------------------------------------
# Database / Vector store setup
# ---------------------------------------------------------------------------

def run_script(script: Path, label: str) -> bool:
    """Run a Python script as a subprocess and stream its output."""
    if not script.exists():
        print(f"  [SKIP] {label}: script not found at {script}")
        return True

    print(f"\n  ▶ Running {label} ({script.name}) ...")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        capture_output=False,   # let output stream to terminal
    )
    if result.returncode != 0:
        print(f"  [ERROR] {label} exited with code {result.returncode}.")
        return False
    print(f"  [OK] {label} completed successfully.")
    return True


def setup_data(session: str) -> bool:
    """Seed database and / or build vector store as required by the session."""
    print(f"\n{'='*60}")
    print("  Data Setup")
    print(f"{'='*60}")

    ok = True

    # Always seed SQLite for sessions that need it (idempotent)
    if session in NEEDS_SQLITE:
        db_path = DATA_DIR / "clinic_data.db"
        if not db_path.exists():
            ok = run_script(SEED_SCRIPT, "seed.py (SQLite database)") and ok
        else:
            print(f"  ✓ SQLite database already exists at data/clinic_data.db")

    # Build ChromaDB vector store (also idempotent but slow — skip if already present)
    if session in NEEDS_CHROMA:
        vector_dir = DATA_DIR / "vectorstore"
        if not vector_dir.exists() or not any(vector_dir.iterdir()):
            ok = run_script(INGEST_SCRIPT, "ingest.py (ChromaDB vector store)") and ok
        else:
            print(f"  ✓ ChromaDB vector store already exists at data/vectorstore/")

    return ok


# ---------------------------------------------------------------------------
# Agent launcher
# ---------------------------------------------------------------------------

def run_agent(session: str) -> None:
    """
    Launch the agent's interactive terminal loop.
    This replaces the current process so Ctrl-C / 'quit' work naturally.
    """
    pkg_dir = find_session_dir(session)
    # The working directory must be the starter/ (or bare session) folder
    # so that `python -m clinicaliq.agent` finds the package correctly.
    work_dir = pkg_dir.parent

    print(f"\n{'='*60}")
    print(f"  Launching ClinicalIQ — {session}")
    print(f"  Working directory: {work_dir}")
    print(f"{'='*60}\n")

    cmd = [sys.executable, "-m", "clinicaliq.agent"]
    # Replace the current process — keeps stdin/stdout as-is for the REPL
    if sys.platform == "win32":
        # os.execv doesn't work well on Windows; use subprocess instead
        result = subprocess.run(cmd, cwd=str(work_dir))
        sys.exit(result.returncode)
    else:
        os.execv(sys.executable, cmd)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate implementations from TODO comments and run the ClinicalIQ agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python generate_and_run.py                  # auto-detect first unimplemented session
              python generate_and_run.py --session s01    # target s01 specifically
              python generate_and_run.py --session s04 --no-run   # generate only
              python generate_and_run.py --session s01 --force    # regenerate even if already done
              python generate_and_run.py --list           # list sessions and their status
        """),
    )
    p.add_argument(
        "--session", "-s",
        choices=ALL_SESSIONS,
        default=None,
        help="Session to process (e.g. s01, s04). Auto-detects if omitted.",
    )
    p.add_argument(
        "--no-run",
        action="store_true",
        default=False,
        help="Generate implementations but do not launch the agent.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-generate even if the file no longer contains TODO markers.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        default=False,
        help="List all sessions and whether they have unimplemented stubs.",
    )
    p.add_argument(
        "--setup-only",
        action="store_true",
        default=False,
        help="Only run seed.py and ingest.py for the target session; skip code generation.",
    )
    return p


def list_sessions() -> None:
    """Print a status table for all known sessions."""
    print(f"\n{'Session':<10} {'Package dir':<50} {'Status'}")
    print("-" * 80)
    for session in ALL_SESSIONS:
        pkg_dir = find_session_dir(session)
        if pkg_dir is None:
            print(f"{session:<10} {'(not found)':<50} —")
            continue

        todo_files = [p.name for p in sorted(pkg_dir.glob("*.py")) if has_todos(p)]
        if todo_files:
            status = f"⚠  TODOs in: {', '.join(todo_files)}"
        else:
            status = "✓  Fully implemented"

        rel = pkg_dir.relative_to(ROOT)
        print(f"{session:<10} {str(rel):<50} {status}")
    print()


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.list:
        list_sessions()
        return

    # ----- Determine target session ----------------------------------------
    if args.session:
        session = args.session
    else:
        session = find_first_unimplemented_session()
        if session is None:
            print("\n✓ All known sessions are fully implemented — nothing to do.")
            print("  Use --force to regenerate a specific session anyway:")
            print("    python generate_and_run.py --session s01 --force")
            return
        print(f"\n[auto-detect] First unimplemented session: {session}")

    # ----- Code generation ---------------------------------------------------
    if not args.setup_only:
        modified = process_session(session, force=args.force)
        if modified:
            print(f"\n  ✓ {len(modified)} file(s) updated:")
            for p in modified:
                print(f"    • {p.relative_to(ROOT)}")
        else:
            print("\n  No files needed updating.")

    # ----- Data setup --------------------------------------------------------
    if session in NEEDS_SQLITE or session in NEEDS_CHROMA:
        ok = setup_data(session)
        if not ok:
            print("\n[ERROR] Data setup failed. The agent may not work correctly.")
            if not args.no_run:
                answer = input("  Continue anyway? [y/N] ").strip().lower()
                if answer != "y":
                    sys.exit(1)

    # ----- Launch agent ------------------------------------------------------
    if args.no_run:
        print("\n  --no-run specified. Done — agent not launched.")
        return

    run_agent(session)


if __name__ == "__main__":
    main()
