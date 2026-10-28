"""
Playground: repository provisioning and per-instance worktrees for the
coding agent baseline.

Repo provisioning mirrors SPICEBench (swebench_qa/core.py `setup_repo`):
clone `https://github.com/{repo}.git` once per repo, then check out
`base_commit`. Instead of re-checking out a single shared tree (which forces
instances of the same repo to run sequentially), each instance gets its own
detached git worktree, so instances run in parallel without checkout
conflicts, and any file an agent leaves behind disappears with the worktree.

Layout:
    <root>/repos/<owner>/<name>/                       # clone cache
    <root>/instances/<experiment_id>/<instance_id>/    # per-instance worktree
"""

import asyncio
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional


def _run_git(args, cwd=None, check=True):
    """Run a git command and return the completed process."""
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result


class Playground:
    """Clone cache + per-instance worktrees for coding agent runs."""

    def __init__(self, root_dir: str, experiment_id: str):
        """
        Args:
            root_dir: Playground root (default: 'playground' under the repo root)
            experiment_id: Experiment identifier, namespaces the instance worktrees
        """
        self.root = Path(root_dir)
        self.experiment_id = experiment_id
        self.repos_dir = self.root / "repos"
        self.instances_dir = self.root / "instances" / experiment_id
        self._repo_locks: Dict[str, asyncio.Lock] = {}
        self._repo_paths: Dict[str, Path] = {}

    # ------------------------------------------------------------------ #
    # Internal helpers (sync; called via asyncio.to_thread)               #
    # ------------------------------------------------------------------ #

    def _repo_lock(self, repo: str) -> asyncio.Lock:
        if repo not in self._repo_locks:
            self._repo_locks[repo] = asyncio.Lock()
        return self._repo_locks[repo]

    def _repo_cache_path(self, repo: str) -> Path:
        # "owner/name" -> nested dirs, same layout as SPICEBench's workspace
        return self.repos_dir / repo

    def _instance_path(self, instance_id: str) -> Path:
        return self.instances_dir / instance_id

    def _ensure_repo_sync(self, repo: str) -> Path:
        """Clone the repo once (SPICEBench style); return the clone path."""
        repo_path = self._repo_cache_path(repo)
        self._repo_paths.setdefault(repo, repo_path)
        if (repo_path / ".git").exists():
            return repo_path

        clone_url = f"https://github.com/{repo}.git"
        last_error: Optional[RuntimeError] = None
        for attempt in range(1, 4):  # initial try + 2 retries
            logging.info(f"Cloning {clone_url} to {repo_path} (attempt {attempt}/3)")
            try:
                _run_git(["clone", clone_url, str(repo_path)])
                return repo_path
            except RuntimeError as e:
                last_error = e
                shutil.rmtree(repo_path, ignore_errors=True)
        raise last_error

    def _create_worktree_sync(self, repo: str, base_commit: str, instance_id: str) -> Path:
        repo_path = self._ensure_repo_sync(repo)
        try:
            _run_git(["worktree", "prune"], cwd=repo_path)
        except RuntimeError:
            pass
        worktree_path = self._instance_path(instance_id).resolve()

        if worktree_path.exists():
            # Leftover from a previous crashed run: remove and recreate.
            self._remove_worktree_sync(repo_path, worktree_path)

        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"[{instance_id}] Creating worktree for {repo} at {base_commit}: {worktree_path}")
        _run_git(
            ["worktree", "add", "--detach", str(worktree_path), base_commit],
            cwd=repo_path,
        )
        return worktree_path

    def _remove_worktree_sync(self, repo_path: Path, worktree_path: Path):
        if not worktree_path.exists():
            return
        try:
            _run_git(
                ["worktree", "remove", "--force", str(worktree_path)],
                cwd=repo_path,
            )
        except RuntimeError:
            # Corrupted worktree metadata: delete manually, then prune.
            shutil.rmtree(worktree_path, ignore_errors=True)
            try:
                _run_git(["worktree", "prune"], cwd=repo_path)
            except RuntimeError:
                pass

    # ------------------------------------------------------------------ #
    # Public async API                                                    #
    # ------------------------------------------------------------------ #

    async def create_worktree(self, repo: str, base_commit: str, instance_id: str) -> Path:
        """Create a detached worktree for one instance (per-repo serialized)."""
        async with self._repo_lock(repo):
            return await asyncio.to_thread(
                self._create_worktree_sync, repo, base_commit, instance_id
            )

    async def discard_worktree(self, repo: str, instance_id: str):
        """Remove one instance's worktree (no-op if it is already gone)."""
        async with self._repo_lock(repo):
            repo_path = self._repo_paths.get(repo) or self._repo_cache_path(repo)
            await asyncio.to_thread(
                self._remove_worktree_sync, repo_path, self._instance_path(instance_id).resolve()
            )

    async def cleanup(self):
        """Prune stale worktree metadata in every cached repo."""
        for repo, repo_path in list(self._repo_paths.items()):
            try:
                await asyncio.to_thread(_run_git, ["worktree", "prune"], repo_path)
            except RuntimeError:
                pass
