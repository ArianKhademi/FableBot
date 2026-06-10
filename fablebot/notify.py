"""Report delivery: console, archive file, and optional webhook.

Set ``FABLEBOT_WEBHOOK_URL`` to a Slack/Discord-style webhook to get the two
daily messages pushed to your phone.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


class Notifier:
    def __init__(self, archive_dir: Path | str = "reports"):
        self.archive_dir = Path(archive_dir)
        self.webhook_url = os.environ.get("FABLEBOT_WEBHOOK_URL")

    def send(self, title: str, body: str) -> None:
        message = f"{title}\n{'=' * len(title)}\n{body}"
        print(message, flush=True)
        self._archive(title, message)
        self._post_webhook(message)

    def _archive(self, title: str, message: str) -> None:
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        slug = title.lower().replace(" ", "-").replace("/", "-")
        (self.archive_dir / f"{stamp}_{slug}.txt").write_text(message)

    def _post_webhook(self, message: str) -> None:
        if not self.webhook_url:
            return
        try:
            import requests

            requests.post(
                self.webhook_url,
                json={"text": message, "content": message},
                timeout=10,
            )
        except Exception as exc:  # delivery failure must never kill the bot
            print(f"[notify] webhook delivery failed: {exc}", flush=True)
