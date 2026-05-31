from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from pathlib import Path

from .core import atomic_write_text


@dataclass(frozen=True)
class ReviewItem:
    title: str
    media_url: str
    kind: str = "image"


@dataclass(frozen=True)
class ReviewPage:
    title: str
    items: tuple[ReviewItem, ...] = ()
    notes: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


def render_review_html(page: ReviewPage) -> str:
    cards = []
    for item in page.items:
        media = (
            f'<img src="{escape(item.media_url)}" alt="{escape(item.title)}">'
            if item.kind == "image"
            else f'<a href="{escape(item.media_url)}">{escape(item.media_url)}</a>'
        )
        cards.append(f"""
        <section class="item">
          <h2>{escape(item.title)}</h2>
          {media}
          <textarea placeholder="Review notes"></textarea>
        </section>
        """)
    metadata = "".join(f"<li><b>{escape(k)}</b>: {escape(v)}</li>" for k, v in page.metadata.items())
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(page.title)}</title>
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f7f7f5; color: #202124; }}
    header {{ padding: 24px 32px; background: #fff; border-bottom: 1px solid #ddd; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 24px; display: grid; gap: 16px; }}
    .item {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 16px; }}
    h1, h2 {{ margin: 0 0 12px; }}
    img {{ max-width: 100%; height: auto; display: block; border: 1px solid #ddd; }}
    textarea {{ box-sizing: border-box; margin-top: 12px; width: 100%; min-height: 96px; padding: 10px; font: inherit; }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(page.title)}</h1>
    {f"<p>{escape(page.notes)}</p>" if page.notes else ""}
    {f"<ul>{metadata}</ul>" if metadata else ""}
  </header>
  <main>{''.join(cards) if cards else '<p>No review items.</p>'}</main>
</body>
</html>
"""


def write_review_html(path: str | Path, page: ReviewPage) -> Path:
    target = Path(path)
    atomic_write_text(target, render_review_html(page))
    return target
