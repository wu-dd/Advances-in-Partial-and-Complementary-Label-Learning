"""Regenerate the Markdown paper lists and the README overview from data/papers.json.

data/papers.json is the single source of truth. To add a paper, append an entry to
"papers" (topic, section, title, url, venue, year, code) and run:

    python3 scripts/build.py

Only the text between the <!-- papers:start --> / <!-- papers:end --> markers in each
list file, and between <!-- stats:start --> / <!-- stats:end --> in README.md, is rewritten.
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "papers.json")


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def paper_line(p):
    s = f"- [{p['title']}]({p['url']}) ({p['venue']} {p['year']})"
    if p.get("code"):
        s += f" [:octocat:]({p['code']})"
    return s


def replace_block(text, start, end, block, path):
    if start not in text or end not in text:
        raise SystemExit(f"{path}: missing {start} / {end} markers")
    a = text.index(start) + len(start)
    b = text.index(end)
    return text[:a] + "\n" + block.rstrip() + "\n" + text[b:]


def main():
    data = json.load(open(DATA, encoding="utf-8"))
    papers = data["papers"]
    stats = []
    for topic in data["topics"]:
        ps = [(i, p) for i, p in enumerate(papers) if p["topic"] == topic["id"]]
        known = {s["id"] for g in topic["groups"] for s in g["sections"]}
        for _, p in ps:
            if p["section"] not in known:
                raise SystemExit(f"unknown section {p['section']!r} for paper {p['title']!r}")
        by_sec = {}
        for i, p in ps:
            by_sec.setdefault(p["section"], []).append((i, p))

        toc, body = ["## Contents"], []
        for g in topic["groups"]:
            secs = [s for s in g["sections"] if by_sec.get(s["id"])]
            if not secs:
                continue
            gid = slug(g["title"])
            toc.append(f"- [{g['title']}](#{gid})")
            body.append(f'<a name="{gid}" />\n\n## <u>{g["title"]}</u>\n')
            for s in secs:
                toc.append(f"  - [{s['title']}](#{s['id']})")
                items = sorted(by_sec[s["id"]], key=lambda ip: (-ip[1]["year"], ip[0]))
                body.append(f'<a name="{s["id"]}" />\n\n### <u>{s["title"]}</u>\n')
                body.append("\n".join(paper_line(p) for _, p in items) + "\n")
        block = "\n".join(toc) + "\n\n" + "\n".join(body)

        n = len(ps)
        nc = sum(1 for _, p in ps if p.get("code"))
        path = os.path.join(ROOT, topic["file"])
        text = open(path, encoding="utf-8").read()
        text = replace_block(text, "<!-- papers:start -->", "<!-- papers:end -->", block, path)
        text = re.sub(r"Number%20of%20Items-\d+", f"Number%20of%20Items-{n}", text)
        text = re.sub(r"\(\d+ papers have official code\)", f"({nc} papers have official code)", text)
        open(path, "w", encoding="utf-8").write(text)
        stats.append((topic, n, nc))
        print(f"{topic['file']}: {n} papers, {nc} with code")

    total = sum(n for _, n, _ in stats)
    rows = ["| Topic | What is imperfect | Papers | With code |", "|---|---|---:|---:|"]
    for topic, n, nc in stats:
        rows.append(f"| [{topic['title']}](./{topic['file']}) | {topic['summary']} | {n} | {nc} |")
    rows.append(f"| **Total** | | **{total}** | **{sum(nc for _, _, nc in stats)}** |")
    path = os.path.join(ROOT, "README.md")
    text = open(path, encoding="utf-8").read()
    text = replace_block(text, "<!-- stats:start -->", "<!-- stats:end -->", "\n".join(rows), path)
    text = re.sub(r"Number%20of%20Items-\d+", f"Number%20of%20Items-{total}", text)
    open(path, "w", encoding="utf-8").write(text)
    print(f"README.md: {total} papers in total")


if __name__ == "__main__":
    main()
