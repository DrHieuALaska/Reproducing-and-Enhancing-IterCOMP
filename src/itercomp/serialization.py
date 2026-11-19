"""Fsynced JSONL checkpointing, tail repair, metadata validation and merge support."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from .config import canonical_hash


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def append_jsonl(path, value):
    payload = (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    with Path(path).open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def recover_jsonl(path, repair=False, collect=True):
    """Repair only a trailing incomplete record. Interior corruption is fatal."""
    path = Path(path)
    if not path.exists():
        return []
    rows, seen, good_end = [], set(), 0
    with path.open("rb") as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                is_tail = not handle.read(1)
                if not is_tail or not repair or line.endswith(b"\n"):
                    raise ValueError(f"Invalid JSONL at byte {start}: {path}") from exc
                tail = line
                # Preserve the damaged bytes for debugging before truncating them.
                backup = path.with_name(path.name + f".incomplete-{time.time_ns()}")
                backup.write_bytes(tail)
                with path.open("r+b") as writable:
                    writable.truncate(good_end)
                    writable.flush()
                    os.fsync(writable.fileno())
                break
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] in seen:
                raise ValueError(f"Invalid or duplicate JSONL ID in {path}")
            seen.add(row["id"])
            rows.append(row if collect else {"id": row["id"], "run_identity_sha256": row.get("run_identity_sha256"),
                                             "has_error": bool(row.get("error"))})
            good_end = handle.tell()
            if not line.endswith(b"\n"):
                if not repair:
                    # A complete JSON object without newline is valid input to readers.
                    break
                with path.open("ab") as writable:
                    writable.write(b"\n")
                    writable.flush()
                    os.fsync(writable.fileno())
                break
    return rows


class RunStore:
    """One writer per output directory; lock is released even after interruption."""
    def __init__(self, directory, filename, identity, details, resume=False):
        self.directory = Path(directory)
        self.path = self.directory / filename
        self.metadata_path = self.directory / "run_metadata.json"
        self.identity = identity
        self.details = details
        self.resume = resume
        self.lock_handle = None
        self.rows = []
        self.ids = set()

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        lock = self.directory / ".run.lock"
        # OS-level advisory lock is released on process death (also on Windows).
        self.lock_handle = lock.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self.lock_handle.seek(0)
                self.lock_handle.write(b"0")
                self.lock_handle.flush()
                self.lock_handle.seek(0)
                msvcrt.locking(self.lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.metadata_path.exists():
                if not self.resume:
                    raise FileExistsError(f"Run exists: {self.directory}; select another output directory or --resume")
                self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                if self.metadata.get("identity") != self.identity:
                    raise ValueError("Run metadata differs; cannot resume with different data/configuration/subset")
            else:
                allowed = {".run.lock"}
                if self.identity["stage"] == "read":
                    allowed.add("compression")  # combined mode's separate compression stage
                if self.path.exists() or any(p.name not in allowed for p in self.directory.iterdir()):
                    raise FileExistsError("Refusing to adopt an existing output directory without matching metadata")
                self.metadata = {"identity": self.identity, "details": self.details, "created_at": utc_now(),
                                 "sessions": [], "serialization_seconds": 0.0}
            self.rows = recover_jsonl(self.path, repair=self.resume, collect=False)
            self.ids = {r["id"] for r in self.rows}
            expected = set(self.identity["selected_ids"])
            if not self.ids <= expected:
                raise ValueError("Checkpoint contains IDs outside this run")
            for row in self.rows:
                if row.get("run_identity_sha256") != canonical_hash(self.identity):
                    raise ValueError("Checkpoint record identity mismatch")
            self.failed_ids = {r["id"] for r in self.rows if r["has_error"]}
            errors_path = self.directory / "errors.jsonl"
            error_rows = recover_jsonl(errors_path, repair=self.resume, collect=False)
            logged_errors = {r["id"] for r in error_rows}
            if not logged_errors <= self.failed_ids:
                raise ValueError("Error log has IDs without corresponding failed output records")
            # A process can stop between the main record and its separate error log.
            missing_errors = self.failed_ids - logged_errors
            if missing_errors:
                from .dataset import read_jsonl
                for row in read_jsonl(self.path):
                    if row["id"] in missing_errors:
                        append_jsonl(errors_path, {"id": row["id"], "time": utc_now(), "recovered": True,
                                                   "error": row["error"], "debug": row.get("debug"),
                                                   "controller_calls": row.get("controller_calls"),
                                                   "reader_calls": row.get("reader_calls")})
            self.metadata["failed_sample_ids"] = sorted(self.failed_ids)
            self.metadata["sessions"].append({"started_at": utc_now()})
            self.metadata["status"] = "running"
            self.save_metadata()
            return self
        except BaseException:
            self._unlock()
            raise

    def _unlock(self):
        if self.lock_handle:
            self.lock_handle.close()
            self.lock_handle = None

    def save_metadata(self):
        self.metadata["updated_at"] = utc_now()
        self.metadata["completed_count"] = len(self.ids)
        self.metadata["output_sha256"] = file_hash(self.path) if self.path.exists() else None
        session = self.metadata["sessions"][-1]
        session["status"] = self.metadata["status"]
        session["completed_count"] = len(self.ids)
        atomic_json(self.directory / "logs" / f"session-{len(self.metadata['sessions']):04d}.json", session)
        atomic_json(self.metadata_path, self.metadata)

    def append(self, row):
        if row["id"] in self.ids or row["id"] not in self.identity["selected_ids"]:
            raise ValueError("Duplicate or unexpected output ID")
        start = time.perf_counter()
        row["run_identity_sha256"] = canonical_hash(self.identity)
        append_jsonl(self.path, row)
        self.ids.add(row["id"])
        if row.get("error"):
            self.failed_ids.add(row["id"])
            append_jsonl(self.directory / "errors.jsonl", {"id": row["id"], "time": utc_now(),
                                                        "error": row["error"], "debug": row.get("debug"),
                                                        "controller_calls": row.get("controller_calls"),
                                                        "reader_calls": row.get("reader_calls")})
        # Avoid hashing a growing full-ranking artifact after every sample (quadratic IO).
        self.metadata["completed_count"] = len(self.ids)
        self.metadata["failed_sample_ids"] = sorted(self.failed_ids)
        self.metadata["updated_at"] = utc_now()
        atomic_json(self.metadata_path, self.metadata)
        # Includes record/error writes and metadata replacement; persisted on the next
        # checkpoint or final metadata save to avoid timing the timestamp itself.
        self.metadata["serialization_seconds"] += time.perf_counter() - start

    def __exit__(self, exc_type, exc_value, exc_tb):
        try:
            selected_done = len(self.ids) == len(self.identity["selected_ids"])
            coverage_complete = len(self.ids) == self.identity["dataset_count"]
            self.metadata["status"] = "interrupted" if exc_type else "complete" if coverage_complete else "partial"
            self.metadata["selected_subset_complete"] = selected_done
            self.metadata["complete_split"] = coverage_complete
            self.metadata["sessions"][-1]["ended_at"] = utc_now()
            self.save_metadata()
        finally:
            self._unlock()
