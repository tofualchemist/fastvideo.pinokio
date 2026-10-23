"""Projects and turns, persisted as one JSON file per project under data/projects/<id>/."""
from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
from pathlib import Path

ACTIVE = ("queued", "preparing", "running")


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(4)}"


def _atomic_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def safe_name(name: str) -> str:
    name = re.sub(r"[^\w.\-]+", "_", name).strip("._")
    return name[:80] or "file"


class ProjectStore:

    def __init__(self, data_dir: Path):
        self.root = Path(data_dir) / "projects"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self._cache: dict[str, dict] = {}
        for folder in self.root.iterdir():
            meta = folder / "project.json"
            if meta.is_file():
                try:
                    self._cache[folder.name] = json.loads(meta.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    pass

    # -- paths ----------------------------------------------------------------------------

    def dir(self, pid: str) -> Path:
        return self.root / pid

    def clip_path(self, pid: str, tid: str) -> Path:
        return self.dir(pid) / "clips" / f"{tid}.mp4"

    def refs_dir(self, pid: str) -> Path:
        d = self.dir(pid) / "refs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def rel(self, pid: str, path: Path) -> str:
        return str(Path(path).relative_to(self.dir(pid)))

    # -- projects -------------------------------------------------------------------------

    def _save(self, project: dict) -> None:
        project["updated"] = time.time()
        d = self.dir(project["id"])
        d.mkdir(parents=True, exist_ok=True)
        _atomic_write(d / "project.json", project)

    def list(self) -> list[dict]:
        with self.lock:
            items = []
            for p in self._cache.values():
                turns = p.get("turns", [])
                items.append({
                    "id": p["id"],
                    "name": p["name"],
                    "created": p["created"],
                    "updated": p.get("updated", p["created"]),
                    "turn_count": len(turns),
                    "active": sum(1 for t in turns if t["status"] in ACTIVE),
                    "unseen": sum(1 for t in turns if t["status"] == "done" and not t.get("seen")),
                    "last_model": next((t["model"] for t in reversed(turns)), None),
                })
            items.sort(key=lambda x: x["updated"], reverse=True)
            return items

    def create(self, name: str) -> dict:
        with self.lock:
            project = {"id": new_id("p"), "name": name.strip() or "Untitled", "created": time.time(), "turns": []}
            self._cache[project["id"]] = project
            self._save(project)
            return project

    def get(self, pid: str) -> dict:
        with self.lock:
            project = self._cache.get(pid)
            if project is None:
                raise KeyError(pid)
            return project

    def rename(self, pid: str, name: str) -> dict:
        with self.lock:
            project = self.get(pid)
            project["name"] = name.strip() or project["name"]
            self._save(project)
            return project

    def delete(self, pid: str) -> None:
        with self.lock:
            self._cache.pop(pid, None)
            d = self.dir(pid)
            if d.exists():
                import shutil
                shutil.rmtree(d, ignore_errors=True)

    # -- turns ----------------------------------------------------------------------------

    def add_turn(self, pid: str, turn: dict) -> dict:
        with self.lock:
            project = self.get(pid)
            turn.setdefault("id", new_id("t"))
            turn["project_id"] = pid
            turn.setdefault("created", time.time())
            turn.setdefault("status", "queued")
            turn.setdefault("progress", {})
            turn["index"] = len(project["turns"]) + 1
            project["turns"].append(turn)
            self._save(project)
            return turn

    def get_turn(self, pid: str, tid: str) -> dict:
        with self.lock:
            for turn in self.get(pid)["turns"]:
                if turn["id"] == tid:
                    return turn
            raise KeyError(tid)

    def update_turn(self, pid: str, tid: str, **fields) -> dict:
        with self.lock:
            project = self.get(pid)
            turn = self.get_turn(pid, tid)
            turn.update(fields)
            self._save(project)
            return turn

    def mark_seen(self, pid: str) -> None:
        with self.lock:
            project = self.get(pid)
            changed = False
            for turn in project["turns"]:
                if turn["status"] == "done" and not turn.get("seen"):
                    turn["seen"] = True
                    changed = True
            if changed:
                self._save(project)

    def delete_turn(self, pid: str, tid: str) -> None:
        with self.lock:
            project = self.get(pid)
            project["turns"] = [t for t in project["turns"] if t["id"] != tid]
            self._save(project)
            clip = self.clip_path(pid, tid)
            if clip.exists():
                clip.unlink()

    def active_turns(self) -> list[tuple[str, str]]:
        with self.lock:
            return [(p["id"], t["id"]) for p in self._cache.values() for t in p["turns"] if t["status"] in ACTIVE]

    def save_upload(self, pid: str, filename: str, data: bytes) -> str:
        name = f"{new_id('u')}_{safe_name(filename)}"
        path = self.refs_dir(pid) / name
        path.write_bytes(data)
        return self.rel(pid, path)
