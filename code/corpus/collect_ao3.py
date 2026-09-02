#!/usr/bin/env python3
"""Collect public AO3 candidates with robots.txt checks and rate limits."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.robotparser import RobotFileParser


LOGGER = logging.getLogger("ao3-collector")
ABO_MARKERS = (
    "alpha/beta/omega",
    "omegaverse",
    "a/b/o dynamics",
    "alpha beta omega",
)


def tag_slug(tag: str) -> str:
    encoded = (
        tag.replace("/", "*s*")
        .replace(".", "*d*")
        .replace("&", "*a*")
    )
    return quote(encoded, safe="*")


def work_ids(html: str) -> list[str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    result = []
    for item in soup.select("li.work[id]"):
        match = re.fullmatch(r"work_(\d+)", item.get("id", ""))
        if match:
            result.append(match.group(1))
    return result


def parse_work(html: str) -> dict:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    chapters = soup.select("#chapters .userstuff")
    if chapters:
        text = "\n\n".join(chapter.get_text("\n", strip=True) for chapter in chapters)
    else:
        body = soup.select_one("#workskin .userstuff")
        text = body.get_text("\n", strip=True) if body else ""
    tags = [node.get_text(" ", strip=True) for node in soup.select("dd.tags a.tag")]
    language = soup.select_one("dd.language")
    rating = soup.select_one("dd.rating")
    return {
        "text": text,
        "tags": tags,
        "language": language.get_text(" ", strip=True) if language else "",
        "rating": rating.get_text(" ", strip=True) if rating else "",
    }


class PoliteSession:
    def __init__(self, base_url: str, user_agent: str, delay: float, seed: int):
        import requests

        if len(user_agent.strip()) < 20 or "contact" not in user_agent.lower():
            raise ValueError(
                "--user-agent must identify the research client and include a contact address"
            )
        self.base_url = base_url.rstrip("/")
        self.delay = max(delay, 1.0)
        self.rng = random.Random(seed)
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": user_agent, "Accept-Language": "en-US,en;q=0.9"}
        )
        robots_url = f"{self.base_url}/robots.txt"
        response = self.session.get(robots_url, timeout=30)
        response.raise_for_status()
        self.robots = RobotFileParser(robots_url)
        self.robots.parse(response.text.splitlines())
        self.user_agent = user_agent
        self.last_request = 0.0

    def get(self, url: str, retries: int = 4) -> str | None:
        if not self.robots.can_fetch(self.user_agent, url):
            raise PermissionError(f"robots.txt does not permit collection of {url}")
        for attempt in range(retries):
            wait = self.delay - (time.monotonic() - self.last_request)
            if wait > 0:
                time.sleep(wait + self.rng.uniform(0.0, min(2.0, self.delay / 3)))
            response = self.session.get(url, timeout=45)
            self.last_request = time.monotonic()
            if response.status_code == 200:
                return response.text
            if response.status_code in {429, 503}:
                retry_after = response.headers.get("Retry-After", "60")
                try:
                    backoff = min(float(retry_after), 300.0)
                except ValueError:
                    backoff = 60.0
                LOGGER.warning("server requested backoff (%s); waiting %.0fs", response.status_code, backoff)
                time.sleep(backoff)
                continue
            if response.status_code in {401, 403, 404, 451}:
                LOGGER.info("skipping unavailable public page (%s): %s", response.status_code, url)
                return None
            LOGGER.warning("HTTP %s on attempt %s for %s", response.status_code, attempt + 1, url)
            time.sleep(10 * (attempt + 1))
        return None


def excluded(work: dict, register: dict) -> bool:
    tags = " ".join(work["tags"]).casefold()
    if any(tag.casefold() in tags for tag in register.get("exclude_tags", [])):
        return True
    if register["name"] != "ABO_omegaverse" and any(marker in tags for marker in ABO_MARKERS):
        return True
    language = work.get("language", "")
    return bool(language and "english" not in language.casefold())


def collect_register(client: PoliteSession, register: dict, raw_dir: Path, max_pages: int) -> dict:
    destination = raw_dir / register["name"]
    destination.mkdir(parents=True, exist_ok=True)
    existing = {path.stem for path in destination.glob("*.json")}
    target = int(register.get("candidate_target", register["target_exemplars"]))
    accepted = len(existing)
    inspected = 0
    seen_work_ids: set[str] = set()

    for page in range(1, max_pages + 1):
        if accepted >= target:
            break
        listing = (
            f"{client.base_url}/tags/{tag_slug(register['tag'])}/works"
            f"?work_search%5Bsort_column%5D=kudos_count&page={page}"
        )
        html = client.get(listing)
        if not html:
            break
        ids = work_ids(html)
        if not ids:
            break
        for work_id in ids:
            if accepted >= target:
                break
            if work_id in seen_work_ids:
                continue
            seen_work_ids.add(work_id)
            source_url = (
                f"{client.base_url}/works/{work_id}"
                "?view_full_work=true&view_adult=true"
            )
            source_id = hashlib.sha256(source_url.encode()).hexdigest()[:20]
            if source_id in existing:
                continue
            page_html = client.get(source_url)
            inspected += 1
            if not page_html:
                continue
            work = parse_work(page_html)
            if excluded(work, register) or len(work["text"]) < 1500:
                continue
            record = {
                "source_id": source_id,
                "source_url": source_url,
                "register": register["name"],
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "language": work["language"],
                "rating": work["rating"],
                "tags": work["tags"],
                "text": work["text"],
            }
            (destination / f"{source_id}.json").write_text(
                json.dumps(record, ensure_ascii=False), encoding="utf-8"
            )
            existing.add(source_id)
            accepted += 1
            LOGGER.info("%s: %d/%d candidates", register["name"], accepted, target)
    return {"accepted": accepted, "inspected_this_run": inspected, "target": target}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("ao3_registers.json")))
    parser.add_argument("--raw-dir", required=True, help="controlled local directory; never commit it")
    parser.add_argument("--user-agent", required=True, help="research client string containing 'contact' and an address")
    parser.add_argument("--delay", type=float, help="seconds between requests; defaults to config value")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()

    try:
        __import__("requests")
        __import__("bs4")
    except ImportError as error:
        raise SystemExit("install requirements-data.txt before collecting AO3 candidates") from error

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    selected = config["registers"]
    if args.only:
        wanted = set(args.only)
        selected = [item for item in selected if item["name"] in wanted]
        missing = wanted - {item["name"] for item in selected}
        if missing:
            raise ValueError(f"unknown registers: {sorted(missing)}")

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = PoliteSession(
        config["site"],
        args.user_agent,
        args.delay or float(config["default_delay_seconds"]),
        args.seed,
    )
    progress = {}
    for register in selected:
        progress[register["name"]] = collect_register(client, register, raw_dir, args.max_pages)
        (raw_dir / "progress.json").write_text(
            json.dumps(progress, indent=2), encoding="utf-8"
        )
    print(json.dumps(progress, indent=2))


if __name__ == "__main__":
    main()
