"""Download the demo images from Wikimedia Commons and write assets/ATTRIBUTION.md."""

import html
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
USER_AGENT = "GhostLens-student-project/0.1 (UCS668 coursework, Thapar Institute)"
THUMB_WIDTH = 1280

# (target path inside assets/, Commons file title, what it is used for)
MANIFEST = [
    ("level1/room217.jpg", "File:Room 217 Stanley Hotel Estes Park Colorado 2024.jpg",
     "Level 1: source of the dark CCTV frame (clue: room number 217)"),
    ("level2/padlock.jpg", "File:Padlock infront of isolated white background 01.jpg",
     "Level 2: single object for classification"),
    ("level2/pocket_watch.jpg", "File:Pocket Watch (Savonette).jpg",
     "Level 2: single object for classification"),
    ("level2/teddy_bear.jpg", "File:Teddy bear early 1900s - Smithsonian Museum of Natural History.jpg",
     "Level 2: single object for classification"),
    ("level3/antique_room.jpg", "File:Interior of living room 01.JPG",
     "Level 3: detection scene (chairs, table, tea set)"),
    ("level2/living_room.jpg", "File:Living room (Unsplash).jpg",
     "Level 2: busy scene showing that classification gives one label"),
]


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _strip_tags(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text)).strip()


def file_info(title: str) -> dict:
    query = urllib.parse.urlencode({
        "action": "query", "format": "json", "titles": title, "prop": "imageinfo",
        "iiprop": "url|extmetadata", "iiurlwidth": THUMB_WIDTH,
    })
    data = json.loads(_get(f"https://commons.wikimedia.org/w/api.php?{query}"))
    page = next(iter(data["query"]["pages"].values()))
    info = page["imageinfo"][0]
    meta = info.get("extmetadata", {})
    return {
        "url": info.get("thumburl") or info["url"],
        "page": info["descriptionurl"],
        "license": meta.get("LicenseShortName", {}).get("value", "unknown"),
        "author": _strip_tags(meta.get("Artist", {}).get("value", "unknown")),
    }


def main() -> None:
    lines = ["# Image attribution", "",
             "All images are from Wikimedia Commons and used under their stated licences.", ""]
    for target, title, purpose in MANIFEST:
        dest = ASSETS / target
        info = file_info(title)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(_get(info["url"]))
            print(f"downloaded {target}")
        else:
            print(f"exists     {target}")
        lines += [f"- `{target}`: {purpose}",
                  f"  - Source: {info['page']}",
                  f"  - Author: {info['author']}",
                  f"  - Licence: {info['license']}", ""]
    (ASSETS / "ATTRIBUTION.md").write_text("\n".join(lines))
    print("wrote assets/ATTRIBUTION.md")


if __name__ == "__main__":
    main()
