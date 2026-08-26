import os
import subprocess
from datetime import datetime
from pathlib import Path

from config import REPO_ROOT, LOCAL_TZ, get_logger

logger = get_logger(__name__)

# Project-local credential file so we never depend on ~/.git-credentials.
# Create it with `https://<user>:<token>@github.com` if you want the
# scheduler to push back to a private mirror; otherwise the helper is
# silently disabled when REPO_ROOT is not a git work tree.
_CRED_FILE = Path(REPO_ROOT) / ".git-credentials"
_CRED_HELPER = f"store --file={_CRED_FILE}"

# Branch to push the data_light/ snapshots to.
PUSH_BRANCH = os.getenv("LIVE_NOWCAST_PUSH_BRANCH", "main")


def _run_git(cmd, *, needs_auth=False):
    """Run a git command in REPO_ROOT. If *needs_auth* is True, inject
    the project-local credential helper so the push/pull never falls
    back to the global credential store."""
    if needs_auth:
        # Empty string clears all previously-defined helpers (global, system),
        # then we set the only one we want.  Without the empty-string reset,
        # git still consults ~/.git-credentials via the global 'store' helper
        # and may pick up credentials written by other projects on this server.
        cmd = [
            "git",
            "-c", "credential.helper=",
            "-c", f"credential.helper={_CRED_HELPER}",
        ] + cmd[1:]
    logger.info("Running git command: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(
            "git command failed (%s): %s\nstdout: %s\nstderr: %s",
            " ".join(cmd), result.returncode,
            result.stdout.strip(), result.stderr.strip())
    else:
        logger.info("git command OK (%s): %s", " ".join(cmd),
                     result.stdout.strip())
    return result


def git_commit_and_push():
    """
    git add data_light -> git commit -> git push origin PUSH_BRANCH.
    Pushes only the light data (no raw outputs).
    No-op when REPO_ROOT is not a git work tree.
    """
    if not (Path(REPO_ROOT) / ".git").exists():
        return

    ts = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    cmds = [
        (["git", "add", "data_light"], False),
        (["git", "commit", "-m", f"update predictions {ts}"], False),
        (["git", "push", "origin", PUSH_BRANCH], True),
    ]

    for cmd, needs_auth in cmds:
        result = _run_git(cmd, needs_auth=needs_auth)
        if result.returncode != 0:
            break


def git_pull_rebase_main():
    """git pull --rebase origin PUSH_BRANCH. No-op outside a git work tree."""
    if not (Path(REPO_ROOT) / ".git").exists():
        return
    _run_git(["git", "pull", "--rebase", "origin", PUSH_BRANCH], needs_auth=True)
