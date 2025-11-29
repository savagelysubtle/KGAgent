#!/usr/bin/env python
"""Run linting and formatting on multi-agent module.

Usage:
    python scripts/lint_multi_agent.py        # Full lint + format
    python scripts/lint_multi_agent.py check  # Check only (no fixes)
"""

import subprocess
import sys

sys.stdout.reconfigure(line_buffering=True)

PATHS = [
    "src/kg_agent/agent/multi/",
    "src/kg_agent/api/routes/multi_agent.py",
    "src/kg_agent/api/routes/agui.py",
    "tests/test_multi_agent*.py",
]


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'=' * 60}")
    print(f"🔧 {description}")
    print(f"{'=' * 60}")
    print(f"$ {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    check_only = "check" in sys.argv

    # Determine paths (expand glob patterns)
    import glob

    expanded_paths = []
    for path in PATHS:
        if "*" in path:
            expanded_paths.extend(glob.glob(path))
        else:
            expanded_paths.append(path)

    print("📁 Files to process:")
    for p in expanded_paths:
        print(f"   - {p}")

    all_passed = True

    # 1. Format (or check format)
    if check_only:
        cmd = ["ruff", "format", "--check"] + expanded_paths
        all_passed &= run_command(cmd, "Checking format...")
    else:
        cmd = ["ruff", "format"] + expanded_paths
        all_passed &= run_command(cmd, "Formatting...")

    # 2. Lint (with or without fix)
    if check_only:
        cmd = ["ruff", "check"] + expanded_paths
        all_passed &= run_command(cmd, "Checking lint...")
    else:
        cmd = ["ruff", "check", "--fix"] + expanded_paths
        all_passed &= run_command(cmd, "Linting (with auto-fix)...")

    # 3. Type check
    cmd = ["python", "-m", "ty", "check"] + [
        p for p in expanded_paths if not p.startswith("tests/")
    ]
    # Note: ty may not be installed, so we handle that gracefully
    try:
        all_passed &= run_command(cmd, "Type checking...")
    except FileNotFoundError:
        print("⚠️  ty not found, skipping type check")
        print("   Install with: pip install ty")

    # Summary
    print(f"\n{'=' * 60}")
    if all_passed:
        print("✅ All checks passed!")
    else:
        print("❌ Some checks failed")
    print(f"{'=' * 60}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

