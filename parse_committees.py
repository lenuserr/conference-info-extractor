#!/usr/bin/env python3
"""Parser that extracts all person names from https://acit.tech/index.php/committees"""

import urllib.request
from html.parser import HTMLParser


class CommitteeParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.names = []
        self.current_section = None
        self.in_strong = False
        self.strong_text = ""
        self.in_heading = False
        self.heading_text = ""
        self.heading_tag = None

    def handle_starttag(self, tag, attrs):
        if tag in ("h2", "h3"):
            self.in_heading = True
            self.heading_tag = tag
            self.heading_text = ""
        elif tag == "strong":
            self.in_strong = True
            self.strong_text = ""

    def handle_endtag(self, tag):
        if tag in ("h2", "h3") and self.in_heading:
            self.in_heading = False
            self.current_section = self.heading_text.strip()
        elif tag == "strong" and self.in_strong:
            self.in_strong = False
            name = self.strong_text.strip()
            if name and self._looks_like_name(name):
                self.names.append({"name": name, "section": self.current_section})

    def handle_data(self, data):
        if self.in_heading:
            self.heading_text += data
        elif self.in_strong:
            self.strong_text += data

    @staticmethod
    def _looks_like_name(text: str) -> bool:
        """Heuristic: a name has 2–4 words, each starting with a capital letter,
        no digits, not too long."""
        if len(text) > 60 or any(ch.isdigit() for ch in text):
            return False
        words = text.split()
        if not (2 <= len(words) <= 5):
            return False
        return all(w[0].isupper() for w in words if w)


def fetch_page(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; CommitteeParser/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8", errors="replace")


def main():
    url = "https://acit.tech/index.php/committees"
    print(f"Fetching {url} …")
    html = fetch_page(url)

    parser = CommitteeParser()
    parser.feed(html)

    names = parser.names
    print(f"\nFound {len(names)} names:\n")

    current_section = None
    for entry in names:
        if entry["section"] != current_section:
            current_section = entry["section"]
            print(f"\n=== {current_section} ===")
        print(f"  {entry['name']}")

    # Also save to a plain text file
    output_path = "names.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        current_section = None
        for entry in names:
            if entry["section"] != current_section:
                current_section = entry["section"]
                f.write(f"\n=== {current_section} ===\n")
            f.write(f"  {entry['name']}\n")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
