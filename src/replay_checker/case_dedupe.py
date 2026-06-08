"""Content-aware duplicate detection for replay cases.

Builds fingerprints from case metadata and reference evidence, then
compares them to find exact and likely duplicates across an existing
cases root.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .case_paths import iter_case_dirs
from .core import stable_hash
from .yaml_lite import parse_simple_yaml


# Chunk size for streaming large diff.patch files (64 KiB).
_DIFF_CHUNK_SIZE = 65536

# Similarity thresholds.
_TASK_SIMILARITY_THRESHOLD = 0.85
_FILE_OVERLAP_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseFingerprint:
    case_id: str
    project_key: str
    source_type: str
    base_commit: str
    source_key: str
    task_key: str
    reference_key: str
    changed_files_key: str
    fingerprint: str


@dataclass(frozen=True)
class DuplicateMatch:
    case_id: str
    relation: str  # "exact" | "likely_duplicate"
    confidence: str  # "high" | "medium" | "low"
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class DuplicateReport:
    candidate_id: str
    matches: tuple[DuplicateMatch, ...]


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _normalize_text(text: str, max_length: int = 2000) -> str:
    """Lowercase, collapse whitespace, trim punctuation, bound length."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_length]


def _text_similarity(a: str, b: str) -> float:
    """Compute sequence-matcher similarity between two normalized strings."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _file_overlap(files_a: set[str], files_b: set[str]) -> float:
    """Compute Jaccard overlap between two file path sets."""
    if not files_a and not files_b:
        return 1.0
    if not files_a or not files_b:
        return 0.0
    intersection = files_a & files_b
    union = files_a | files_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Diff / reference helpers
# ---------------------------------------------------------------------------


def _hash_diff_patch_chunked(path: Path) -> str:
    """Stream-hash a diff.patch file in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_DIFF_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]


def _extract_changed_files_from_diff(diff_path: Path) -> set[str]:
    """Extract changed file paths from unified diff headers.

    Only reads the header lines (``---``/``+++`` or ``diff --git``) to avoid
    loading the entire diff into memory.
    """
    files: set[str] = set()
    with open(diff_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("diff --git "):
                # ``diff --git a/path b/path``
                parts = line.split()
                if len(parts) >= 4:
                    b_path = parts[3]
                    if b_path.startswith("b/"):
                        b_path = b_path[2:]
                    files.add(b_path)
            elif line.startswith("+++ b/"):
                files.add(line[6:].strip())
    return files


# ---------------------------------------------------------------------------
# Case reading helpers
# ---------------------------------------------------------------------------


def _read_case_metadata(case_dir: Path) -> dict[str, Any]:
    """Read case.yaml and return a plain dict."""
    data = parse_simple_yaml(case_dir / "case.yaml")
    return data if isinstance(data, dict) else {}


def _read_reference_metadata(case_dir: Path) -> dict[str, Any]:
    """Read _reference/reference_metadata.yaml if present."""
    meta_path = case_dir / "_reference" / "reference_metadata.yaml"
    if not meta_path.exists():
        return {}
    data = parse_simple_yaml(meta_path)
    return data if isinstance(data, dict) else {}


def _read_task_text(case_dir: Path) -> str:
    """Read the first 4000 chars of task.md for fingerprinting."""
    task_path = case_dir / "task.md"
    if not task_path.exists():
        return ""
    return task_path.read_text(encoding="utf-8", errors="ignore")[:4000]


def _read_changed_files_from_reference(case_dir: Path) -> set[str]:
    """Get changed files from reference metadata or diff headers."""
    ref_meta = _read_reference_metadata(case_dir)
    meta_files = ref_meta.get("changed_files")
    if isinstance(meta_files, list) and meta_files:
        return {str(f) for f in meta_files}

    diff_path = case_dir / "_reference" / "diff.patch"
    if diff_path.exists():
        return _extract_changed_files_from_diff(diff_path)

    return set()


# ---------------------------------------------------------------------------
# Fingerprint builder
# ---------------------------------------------------------------------------


def build_case_fingerprint(case_dir: Path) -> CaseFingerprint:
    """Build a content-aware fingerprint for a case directory."""
    meta = _read_case_metadata(case_dir)
    case_id = str(meta.get("id", case_dir.name))
    project_name = Path(str(meta.get("project_path", ""))).name
    source_type = str(meta.get("source_type", ""))
    base_commit = str(meta.get("base_commit", "")).strip()
    skill_name = str(meta.get("skill_name", ""))
    eval_id = str(meta.get("eval_id", ""))

    source_key = _build_source_key(meta, case_dir)
    task_key = _build_task_key(case_dir)
    reference_key = _build_reference_key(case_dir, base_commit)
    changed_files_key = _build_changed_files_key(case_dir)

    # Composite fingerprint
    composite = stable_hash(
        f"{project_name}:{base_commit}:{reference_key}:{task_key}:{changed_files_key}"
    )

    return CaseFingerprint(
        case_id=case_id,
        project_key=stable_hash(project_name),
        source_type=source_type,
        base_commit=base_commit,
        source_key=source_key,
        task_key=task_key,
        reference_key=reference_key,
        changed_files_key=changed_files_key,
        fingerprint=composite,
    )


def _build_source_key(meta: dict[str, Any], case_dir: Path) -> str:
    """Hash source path and title for source-level identity."""
    source_path = str(meta.get("source_path", ""))
    title = ""
    task_path = case_dir / "task.md"
    if task_path.exists():
        for line in task_path.read_text(encoding="utf-8", errors="ignore").splitlines()[:5]:
            if line.startswith("# "):
                title = line[2:].strip()
                break
    return stable_hash(source_path + title)


def _build_task_key(case_dir: Path) -> str:
    """Normalized hash of task text content."""
    task_text = _read_task_text(case_dir)
    normalized = _normalize_text(task_text)
    return stable_hash(normalized) if normalized else ""


def _build_reference_key(case_dir: Path, base_commit: str) -> str:
    """Hash of reference diff content (chunked) or base commit."""
    diff_path = case_dir / "_reference" / "diff.patch"
    if diff_path.exists() and diff_path.stat().st_size > 0:
        return _hash_diff_patch_chunked(diff_path)
    if base_commit:
        return stable_hash(base_commit)
    return ""


def _build_changed_files_key(case_dir: Path) -> str:
    """Sorted, deduplicated hash of changed file paths."""
    files = _read_changed_files_from_reference(case_dir)
    if not files:
        return ""
    return stable_hash("\n".join(sorted(files)))


# ---------------------------------------------------------------------------
# Duplicate finder
# ---------------------------------------------------------------------------


def find_duplicates(
    candidate_dir: Path,
    existing_cases: dict[str, CaseFingerprint],
    *,
    existing_case_dirs: dict[str, Path] | None = None,
) -> DuplicateReport:
    """Compare a candidate case against all existing cases.

    Returns a ``DuplicateReport`` with exact and likely matches.
    For likely-duplicate detection that requires task text comparison,
    pass ``existing_case_dirs`` mapping case_id to its directory.
    """
    candidate_fp = build_case_fingerprint(candidate_dir)
    matches: list[DuplicateMatch] = []

    candidate_meta = _read_case_metadata(candidate_dir)
    candidate_task_text = _normalize_text(_read_task_text(candidate_dir))
    candidate_files = _read_changed_files_from_reference(candidate_dir)
    candidate_skill_prefix = _skill_eval_prefix(str(candidate_meta.get("id", "")))

    for existing_id, existing_fp in existing_cases.items():
        if existing_id == candidate_fp.case_id:
            continue

        existing_dir = (existing_case_dirs or {}).get(existing_id)
        match = _compare_fingerprints(
            candidate_fp,
            existing_fp,
            candidate_task_text=candidate_task_text,
            candidate_files=candidate_files,
            candidate_skill_prefix=candidate_skill_prefix,
            existing_case_dir=existing_dir,
        )
        if match:
            matches.append(match)

    return DuplicateReport(candidate_id=candidate_fp.case_id, matches=tuple(matches))


def _compare_fingerprints(
    candidate: CaseFingerprint,
    existing: CaseFingerprint,
    *,
    candidate_task_text: str,
    candidate_files: set[str],
    candidate_skill_prefix: str,
    existing_case_dir: Path | None = None,
) -> DuplicateMatch | None:
    """Compare two fingerprints and return a match or None."""
    reasons: list[str] = []

    # Exact: same reference diff content
    if (candidate.reference_key
            and candidate.reference_key == existing.reference_key
            and candidate.project_key == existing.project_key):
        reasons.append("same_reference_diff")

    # Exact: same base commit + project + files
    if (candidate.base_commit
            and candidate.base_commit == existing.base_commit
            and candidate.project_key == existing.project_key
            and candidate.changed_files_key == existing.changed_files_key):
        if "same_reference_diff" not in reasons:
            reasons.append("same_base_commit_and_files")

    # Exact: same skill eval prefix
    existing_prefix = _skill_eval_prefix(existing.case_id)
    if (candidate_skill_prefix
            and candidate_skill_prefix == existing_prefix):
        reasons.append("same_skill_eval_prefix")

    if reasons:
        return DuplicateMatch(
            case_id=existing.case_id,
            relation="exact",
            confidence="high",
            reasons=tuple(reasons),
        )

    # Likely duplicate: task similarity + file overlap within same project
    if candidate.project_key != existing.project_key:
        return None

    # Use task_key equality for fast check; fall back to full text comparison
    task_sim = 0.0
    if candidate.task_key and candidate.task_key == existing.task_key:
        task_sim = 1.0
    elif candidate_task_text and existing_case_dir:
        existing_task = _normalize_text(_read_task_text(existing_case_dir))
        if existing_task:
            task_sim = _text_similarity(candidate_task_text, existing_task)

    file_overlap = 0.0
    if candidate.changed_files_key and candidate.changed_files_key == existing.changed_files_key:
        file_overlap = 1.0
    elif candidate_files and existing_case_dir:
        existing_files = _read_changed_files_from_reference(existing_case_dir)
        if existing_files:
            file_overlap = _file_overlap(candidate_files, existing_files)

    likely_reasons: list[str] = []
    if task_sim >= _TASK_SIMILARITY_THRESHOLD:
        likely_reasons.append(f"task_similarity={task_sim:.2f}")
    if file_overlap >= _FILE_OVERLAP_THRESHOLD:
        likely_reasons.append(f"file_overlap={file_overlap:.2f}")

    if len(likely_reasons) >= 2:
        return DuplicateMatch(
            case_id=existing.case_id,
            relation="likely_duplicate",
            confidence="medium",
            reasons=tuple(likely_reasons),
        )

    return None


def _skill_eval_prefix(case_id: str) -> str:
    """Extract the ``{skill}-eval{N}`` prefix from a case ID, if any."""
    parts = case_id.rsplit("-", 1)
    if len(parts) != 2:
        return ""
    prefix, suffix = parts
    if len(suffix) < 4 or not all(c in "0123456789abcdef" for c in suffix):
        return ""
    if "-eval" not in prefix:
        return ""
    return prefix


# ---------------------------------------------------------------------------
# Bulk scan for inspect-duplicates CLI
# ---------------------------------------------------------------------------


def scan_all_fingerprints(cases_root: Path) -> dict[str, CaseFingerprint]:
    """Build fingerprints for every case directory under *cases_root*."""
    result: dict[str, CaseFingerprint] = {}
    for child in iter_case_dirs(cases_root):
        fp = build_case_fingerprint(child)
        result[fp.case_id] = fp
    return result


def _build_case_dir_map(cases_root: Path) -> dict[str, Path]:
    """Map case_id to its directory for text-based comparison."""
    result: dict[str, Path] = {}
    for child in iter_case_dirs(cases_root):
        meta = _read_case_metadata(child)
        case_id = str(meta.get("id", child.name))
        result[case_id] = child
    return result


def find_all_duplicates(cases_root: Path) -> tuple[
    list[tuple[str, str, tuple[str, ...]]],
    list[tuple[str, str, float, tuple[str, ...]]],
]:
    """Scan all cases and return (exact_clusters, likely_pairs).

    exact_clusters: list of (representative_id, duplicate_id, reasons)
    likely_pairs: list of (id_a, id_b, similarity, reasons)
    """
    fingerprints = scan_all_fingerprints(cases_root)
    if len(fingerprints) < 2:
        return [], []

    # Group by fingerprint hash for exact detection
    by_fingerprint: dict[str, list[str]] = defaultdict(list)
    by_ref_key: dict[str, list[str]] = defaultdict(list)
    by_base_files: dict[str, list[str]] = defaultdict(list)
    by_skill_prefix: dict[str, list[str]] = defaultdict(list)

    for case_id, fp in fingerprints.items():
        by_fingerprint[fp.fingerprint].append(case_id)
        if fp.reference_key:
            by_ref_key[fp.reference_key].append(case_id)
        combined = f"{fp.project_key}:{fp.base_commit}:{fp.changed_files_key}"
        if fp.base_commit:
            by_base_files[combined].append(case_id)
        sp = _skill_eval_prefix(case_id)
        if sp:
            by_skill_prefix[sp].append(case_id)

    exact_pairs: list[tuple[str, str, tuple[str, ...]]] = []
    seen_exact: set[tuple[str, str]] = set()

    # Same composite fingerprint
    for group in by_fingerprint.values():
        if len(group) < 2:
            continue
        rep = group[0]
        for dup in group[1:]:
            pair = tuple(sorted([rep, dup]))
            if pair not in seen_exact:
                seen_exact.add(pair)
                exact_pairs.append((rep, dup, ("same_fingerprint",)))

    # Same reference diff
    for group in by_ref_key.values():
        if len(group) < 2:
            continue
        rep = group[0]
        for dup in group[1:]:
            pair = tuple(sorted([rep, dup]))
            if pair not in seen_exact:
                seen_exact.add(pair)
                exact_pairs.append((rep, dup, ("same_reference_diff",)))

    # Same base commit + files
    for group in by_base_files.values():
        if len(group) < 2:
            continue
        rep = group[0]
        for dup in group[1:]:
            pair = tuple(sorted([rep, dup]))
            if pair not in seen_exact:
                seen_exact.add(pair)
                exact_pairs.append((rep, dup, ("same_base_commit_and_files",)))

    # Same skill eval prefix
    for group in by_skill_prefix.values():
        if len(group) < 2:
            continue
        rep = group[0]
        for dup in group[1:]:
            pair = tuple(sorted([rep, dup]))
            if pair not in seen_exact:
                seen_exact.add(pair)
                exact_pairs.append((rep, dup, ("same_skill_eval_prefix",)))

    # Likely duplicates: task text similarity + file overlap
    likely_pairs: list[tuple[str, str, float, tuple[str, ...]]] = []
    fp_list = list(fingerprints.items())
    dir_map = _build_case_dir_map(cases_root)

    for i in range(len(fp_list)):
        id_a, fp_a = fp_list[i]
        for j in range(i + 1, len(fp_list)):
            id_b, fp_b = fp_list[j]
            if fp_a.project_key != fp_b.project_key:
                continue

            pair = tuple(sorted([id_a, id_b]))
            if pair in seen_exact:
                continue

            # Task similarity
            task_sim = 0.0
            if fp_a.task_key and fp_a.task_key == fp_b.task_key:
                task_sim = 1.0
            else:
                dir_a = dir_map.get(id_a)
                dir_b = dir_map.get(id_b)
                if dir_a and dir_b:
                    text_a = _normalize_text(_read_task_text(dir_a))
                    text_b = _normalize_text(_read_task_text(dir_b))
                    if text_a and text_b:
                        task_sim = _text_similarity(text_a, text_b)

            # File overlap
            file_overlap = 0.0
            if fp_a.changed_files_key and fp_a.changed_files_key == fp_b.changed_files_key:
                file_overlap = 1.0
            else:
                dir_a = dir_map.get(id_a)
                dir_b = dir_map.get(id_b)
                if dir_a and dir_b:
                    files_a = _read_changed_files_from_reference(dir_a)
                    files_b = _read_changed_files_from_reference(dir_b)
                    if files_a and files_b:
                        file_overlap = _file_overlap(files_a, files_b)

            reasons: list[str] = []
            if task_sim >= _TASK_SIMILARITY_THRESHOLD:
                reasons.append(f"task_similarity={task_sim:.2f}")
            if file_overlap >= _FILE_OVERLAP_THRESHOLD:
                reasons.append(f"file_overlap={file_overlap:.2f}")

            if len(reasons) >= 2:
                likely_pairs.append((id_a, id_b, task_sim, tuple(reasons)))

    return exact_pairs, likely_pairs


def inspect_duplicates(cases_root: Path) -> dict[str, Any]:
    """Full audit of duplicates for CLI reporting."""
    exact_clusters, likely_pairs = find_all_duplicates(cases_root)
    return {
        "exact_clusters": exact_clusters,
        "likely_pairs": likely_pairs,
        "total_cases": len(scan_all_fingerprints(cases_root)),
    }
