from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

ScalarParser = Callable[[str, tuple[Any, ...]], Any]


def identity_scalar(value: str, path: tuple[Any, ...] = ()) -> str:
    return value


def coerce_bool_int(value: str, path: tuple[Any, ...] = ()) -> Any:
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        return int(value)
    except ValueError:
        return value


def coerce_nested_int(value: str, path: tuple[Any, ...] = ()) -> Any:
    if len(path) >= 2 and not any(isinstance(part, int) for part in path):
        try:
            return int(value)
        except ValueError:
            return value
    return value


def parse_yaml_file(
    path: str | Path,
    *,
    scalar_parser: ScalarParser = identity_scalar,
    parse_list_item_dicts: bool = True,
    parse_inline_lists: bool = False,
) -> Any:
    return parse_yaml_text(
        Path(path).read_text(encoding="utf-8"),
        scalar_parser=scalar_parser,
        parse_list_item_dicts=parse_list_item_dicts,
        parse_inline_lists=parse_inline_lists,
    )


def parse_yaml_text(
    text: str,
    *,
    scalar_parser: ScalarParser = identity_scalar,
    parse_list_item_dicts: bool = True,
    parse_inline_lists: bool = False,
) -> Any:
    rows = _rows(text)
    if not rows:
        return {}
    value, _ = _parse_block(
        rows,
        0,
        rows[0][0],
        scalar_parser,
        (),
        parse_list_item_dicts,
        parse_inline_lists,
    )
    return value


def emit_yaml(data: Any, indent: int = 0) -> str:
    lines: list[str] = []
    pfx = "  " * indent

    if isinstance(data, dict):
        for key, value in data.items():
            _emit_key_value(lines, str(key), value, indent, bullet=False)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                _emit_dict_as_list_item(lines, item, indent)
            else:
                lines.append(f"{pfx}- {item}")
    else:
        lines.append(f"{pfx}{data}")

    return "\n".join(lines)


def parse_list_of_dicts(text: str) -> list[dict[str, str]]:
    parsed = parse_yaml_text(text)
    if isinstance(parsed, list):
        return [
            {str(key): str(value) for key, value in item.items()}
            for item in parsed
            if isinstance(item, dict)
        ]
    if isinstance(parsed, dict) and isinstance(parsed.get("_top_list_"), list):
        return [
            {str(key): str(value) for key, value in item.items()}
            for item in parsed["_top_list_"]
            if isinstance(item, dict)
        ]
    return []


def _rows(text: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for raw_line in text.rstrip("\n").splitlines():
        line = raw_line.rstrip()
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append((len(line) - len(stripped), stripped))
    return rows


def _parse_block(
    rows: list[tuple[int, str]],
    index: int,
    indent: int,
    scalar_parser: ScalarParser,
    path: tuple[Any, ...],
    parse_list_item_dicts: bool,
    parse_inline_lists: bool,
) -> tuple[Any, int]:
    if index >= len(rows):
        return {}, index
    if rows[index][1].startswith("- "):
        return _parse_list(
            rows, index, indent, scalar_parser, path, parse_list_item_dicts, parse_inline_lists
        )
    return _parse_dict(
        rows, index, indent, scalar_parser, path, parse_list_item_dicts, parse_inline_lists
    )


def _parse_list(
    rows: list[tuple[int, str]],
    index: int,
    indent: int,
    scalar_parser: ScalarParser,
    path: tuple[Any, ...],
    parse_list_item_dicts: bool,
    parse_inline_lists: bool,
) -> tuple[list[Any], int]:
    result: list[Any] = []
    while index < len(rows):
        row_indent, stripped = rows[index]
        if row_indent < indent:
            break
        if row_indent != indent or not stripped.startswith("- "):
            break

        item_path = (*path, len(result))
        content = stripped[2:].strip()
        if parse_list_item_dicts and ":" in content:
            key, value = _split_key_value(content)
            item: dict[str, Any] = {}
            index += 1
            if value == "":
                if index < len(rows) and rows[index][0] > row_indent:
                    child, index = _parse_block(
                        rows,
                        index,
                        rows[index][0],
                        scalar_parser,
                        (*item_path, key),
                        parse_list_item_dicts,
                        parse_inline_lists,
                    )
                    item[key] = child
                else:
                    item[key] = []
            else:
                item[key] = _parse_value(value, scalar_parser, (*item_path, key), parse_inline_lists)

            while index < len(rows) and rows[index][0] > row_indent:
                next_indent, next_stripped = rows[index]
                if next_indent != row_indent + 2 or next_stripped.startswith("- ") or ":" not in next_stripped:
                    break
                child_key, child_value = _split_key_value(next_stripped)
                index += 1
                if child_value == "":
                    if index < len(rows) and rows[index][0] > next_indent:
                        child, index = _parse_block(
                            rows,
                            index,
                            rows[index][0],
                            scalar_parser,
                            (*item_path, child_key),
                            parse_list_item_dicts,
                            parse_inline_lists,
                        )
                        item[child_key] = child
                    else:
                        item[child_key] = []
                else:
                    item[child_key] = _parse_value(
                        child_value, scalar_parser, (*item_path, child_key), parse_inline_lists
                    )
            result.append(item)
        else:
            result.append(_parse_value(content, scalar_parser, item_path, parse_inline_lists))
            index += 1
    return result, index


def _parse_dict(
    rows: list[tuple[int, str]],
    index: int,
    indent: int,
    scalar_parser: ScalarParser,
    path: tuple[Any, ...],
    parse_list_item_dicts: bool,
    parse_inline_lists: bool,
) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    while index < len(rows):
        row_indent, stripped = rows[index]
        if row_indent < indent:
            break
        if row_indent != indent or stripped.startswith("- ") or ":" not in stripped:
            break

        key, value = _split_key_value(stripped)
        key_path = (*path, key)
        index += 1
        if value == "":
            if index < len(rows) and rows[index][0] > row_indent:
                child, index = _parse_block(
                    rows,
                    index,
                    rows[index][0],
                    scalar_parser,
                    key_path,
                    parse_list_item_dicts,
                    parse_inline_lists,
                )
                result[key] = child
            else:
                result[key] = []
        elif value == "[]":
            result[key] = []
        else:
            result[key] = _parse_value(value, scalar_parser, key_path, parse_inline_lists)
    return result, index


def _split_key_value(text: str) -> tuple[str, str]:
    key, value = text.split(":", 1)
    return key.strip(), value.strip()


def _parse_scalar(value: str, scalar_parser: ScalarParser, path: tuple[Any, ...]) -> Any:
    return scalar_parser(value, path)


def _parse_value(
    value: str,
    scalar_parser: ScalarParser,
    path: tuple[Any, ...],
    parse_inline_lists: bool,
) -> Any:
    if parse_inline_lists and value.startswith("[") and value.endswith("]"):
        raw_items = value[1:-1].split(",")
        return [
            _parse_scalar(item.strip().strip('"').strip("'"), scalar_parser, (*path, index))
            for index, item in enumerate(raw_items)
            if item.strip()
        ]
    return _parse_scalar(value, scalar_parser, path)


def _emit_key_value(
    lines: list[str],
    key: str,
    value: Any,
    indent: int,
    *,
    bullet: bool = False,
) -> None:
    pfx = "  " * indent
    marker = f"{pfx}- " if bullet else f"{pfx}"

    if isinstance(value, dict):
        lines.append(f"{marker}{key}:")
        for k, v in value.items():
            _emit_key_value(lines, str(k), v, indent + 1)
    elif isinstance(value, list):
        if not value:
            lines.append(f"{marker}{key}: []")
        elif all(isinstance(v, dict) for v in value):
            lines.append(f"{marker}{key}:")
            for item in value:
                _emit_dict_as_list_item(lines, item, indent + 1)
        else:
            lines.append(f"{marker}{key}:")
            for item in value:
                lines.append(f"{'  ' * (indent + 1)}- {item}")
    elif isinstance(value, bool):
        lines.append(f"{marker}{key}: {str(value).lower()}")
    elif isinstance(value, float):
        lines.append(f"{marker}{key}: {value!r}")
    else:
        lines.append(f"{marker}{key}: {value}")


def parse_simple_yaml(path: str | Path) -> dict[str, object]:
    """Parse the tiny YAML subset this project writes: scalar keys, string lists, nested blocks."""
    data = parse_yaml_file(
        path,
        scalar_parser=coerce_nested_int,
        parse_list_item_dicts=False,
    )
    return data if isinstance(data, dict) else {}


def _emit_dict_as_list_item(
    lines: list[str],
    item: dict[str, Any],
    indent: int,
) -> None:
    pfx = "  " * indent
    for index, (key, value) in enumerate(item.items()):
        prefix = f"{pfx}- " if index == 0 else f"{pfx}  "
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            for k, v in value.items():
                _emit_key_value(lines, str(k), v, indent + 2)
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            for v in value:
                if isinstance(v, dict):
                    _emit_dict_as_list_item(lines, v, indent + 2)
                else:
                    lines.append(f"{'  ' * (indent + 2)}- {v}")
        elif isinstance(value, bool):
            lines.append(f"{prefix}{key}: {str(value).lower()}")
        elif isinstance(value, float):
            lines.append(f"{prefix}{key}: {value!r}")
        else:
            lines.append(f"{prefix}{key}: {value}")
