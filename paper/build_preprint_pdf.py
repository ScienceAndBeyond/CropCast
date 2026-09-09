"""Build a readable HTML and PDF version of the irrigation preprint."""

from __future__ import annotations

import html
import base64
import json
import os
import re
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
MANUSCRIPT = PAPER / "irrigation_weather_coupling.md"
BIB = PAPER / "references.bib"
HTML_OUT = PAPER / "irrigation_weather_coupling.html"
PDF_OUT = PAPER / "irrigation_weather_coupling.pdf"
DOC_OUT = PAPER / "irrigation_weather_coupling.doc"
TITLE = "Irrigation Is Associated With Weaker Weather-Yield Links in Kansas and Nebraska Corn Counties, 2008-2018"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


def strip_bib_value(value: str) -> str:
    value = value.strip().rstrip(",").strip()
    if value.startswith("{") and value.endswith("}"):
        value = value[1:-1]
    return value.replace("{", "").replace("}", "").replace("--", "–")


def parse_bib(path: Path) -> dict[str, dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    entries: dict[str, dict[str, str]] = {}
    for match in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", text, flags=re.S):
        key = match.group(1).strip()
        body = match.group(2)
        fields: dict[str, str] = {}
        for line in body.splitlines():
            if "=" not in line:
                continue
            name, value = line.split("=", 1)
            fields[name.strip().lower()] = strip_bib_value(value)
        entries[key] = fields
    return entries


def split_authors(author_field: str) -> list[str]:
    if not author_field:
        return []
    return [part.strip() for part in author_field.split(" and ")]


def last_name(author: str) -> str:
    author = author.strip().strip("{}")
    if "," in author:
        return author.split(",", 1)[0].strip()
    return author.split()[-1] if author.split() else author


def citation_label(entry: dict[str, str]) -> str:
    author_field = entry.get("author", "")
    if "USDA National Agricultural Statistics Service" in author_field:
        return f"USDA NASS, {entry.get('year', 'n.d.')}"
    authors = split_authors(author_field)
    year = entry.get("year", "n.d.")
    if not authors:
        label = entry.get("title", "Unknown source")
    elif len(authors) == 1:
        label = last_name(authors[0])
    elif len(authors) == 2:
        label = f"{last_name(authors[0])} and {last_name(authors[1])}"
    else:
        label = f"{last_name(authors[0])} et al."
    return f"{label}, {year}"


def format_authors(author_field: str) -> str:
    if "USDA National Agricultural Statistics Service" in author_field:
        return "USDA National Agricultural Statistics Service"
    authors = split_authors(author_field)
    if not authors:
        return ""
    names = []
    for author in authors:
        if "," in author:
            last, rest = [part.strip() for part in author.split(",", 1)]
            names.append(f"{last}, {rest}")
        else:
            names.append(author.strip("{}"))
    if len(names) == 1:
        return names[0]
    return "; ".join(names[:-1]) + f"; and {names[-1]}"


def format_reference(entry: dict[str, str]) -> str:
    authors = format_authors(entry.get("author", ""))
    year = entry.get("year", "")
    title = entry.get("title", "")
    journal = entry.get("journal", "")
    volume = entry.get("volume", "")
    number = entry.get("number", "")
    pages = entry.get("pages", "")
    doi = entry.get("doi", "")
    url = entry.get("url", "")
    howpublished = entry.get("howpublished", "")
    note = entry.get("note", "")

    bits: list[str] = []
    if authors:
        bits.append(authors)
    if year:
        bits.append(f"({year}).")
    if title:
        bits.append(f"{title}.")
    if journal:
        vol = volume
        if number:
            vol = f"{vol}({number})" if vol else f"({number})"
        journal_part = journal
        if vol:
            journal_part += f", {vol}"
        if pages:
            journal_part += f", {pages}"
        bits.append(journal_part + ".")
    elif howpublished:
        bits.append(howpublished + ".")
    if doi:
        bits.append(f"https://doi.org/{doi}.")
    elif url:
        bits.append(url + ".")
    if note:
        bits.append(note + ".")
    return " ".join(bits)


def inline_markup(text: str, bib: dict[str, dict[str, str]], cited: list[str]) -> str:
    placeholders: dict[str, str] = {}

    def stash(value: str) -> str:
        token = f"@@PLACEHOLDER{len(placeholders)}@@"
        placeholders[token] = value
        return token

    def cite_repl(match: re.Match[str]) -> str:
        keys = [part.strip().lstrip("@") for part in match.group(1).split(";")]
        labels = []
        for key in keys:
            if key in bib:
                if key not in cited:
                    cited.append(key)
                labels.append(citation_label(bib[key]))
            else:
                labels.append(key)
        return stash(f"({'; '.join(labels)})")

    text = re.sub(r"\[(@[A-Za-z0-9_:-]+(?:;\s*@[A-Za-z0-9_:-]+)*)\]", cite_repl, text)

    def code_repl(match: re.Match[str]) -> str:
        return stash(f"<code>{html.escape(match.group(1))}</code>")

    text = re.sub(r"`([^`]+)`", code_repl, text)
    text = html.escape(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = text.replace("R²", '<span class="nowrap">R<sup>2</sup></span>')
    for token, value in placeholders.items():
        text = text.replace(token, value)
    return text


def reference_sort_key(entry: dict[str, str]) -> tuple[str, str]:
    author_field = entry.get("author", "")
    if "USDA National Agricultural Statistics Service" in author_field:
        author = "USDA NASS"
    else:
        authors = split_authors(author_field)
        author = last_name(authors[0]) if authors else entry.get("title", "")
    return author.lower(), entry.get("year", "")


def table_to_html(lines: list[str], bib: dict[str, dict[str, str]], cited: list[str]) -> str:
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(cells)
    header = rows[0]
    body = rows[2:]
    parts = ["<table>", "<thead><tr>"]
    for cell in header:
        parts.append(f"<th>{inline_markup(cell, bib, cited)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in body:
        parts.append("<tr>")
        for cell in row:
            parts.append(f"<td>{inline_markup(cell, bib, cited)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def markdown_body(md: str, bib: dict[str, dict[str, str]]) -> tuple[str, list[str]]:
    lines = md.splitlines()
    cited: list[str] = []
    out: list[str] = []
    para: list[str] = []
    i = 0

    def flush_para() -> None:
        if not para:
            return
        text = " ".join(part.strip() for part in para).strip()
        para.clear()
        if text in {
            "zₐ(c,t) = z(c,t) - mean_c(z)",
            "yieldₐ = β₀ + β₁ PRCPₐ + β₂ EDD_TMAXₐ + β₃ VPDₐ + β₄ TMAXₐ + ε",
        }:
            out.append(f"<p class=\"equation\">{inline_markup(text, bib, cited)}</p>")
        elif text.startswith("**Table ") or text.startswith("**Figure ") or text.startswith("**Appendix Table "):
            out.append(f"<p class=\"caption\">{inline_markup(text, bib, cited)}</p>")
        elif text.startswith("`") and text.endswith("`") and text.count("`") == 2:
            out.append(f"<p class=\"equation\">{inline_markup(text, bib, cited)}</p>")
        else:
            out.append(f"<p>{inline_markup(text, bib, cited)}</p>")

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == "## References":
            flush_para()
            out.append("<h2>References</h2>")
            out.append("<!-- references inserted here -->")
            i += 1
            while i < len(lines) and not lines[i].startswith("## "):
                i += 1
            continue

        if not stripped:
            flush_para()
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and set(lines[i + 1].strip()) <= {"|", "-", ":", " "}:
            flush_para()
            table_lines = [lines[i], lines[i + 1]]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            out.append(table_to_html(table_lines, bib, cited))
            continue

        image_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            flush_para()
            alt, src = image_match.groups()
            caption = ""
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].strip().startswith("**Figure "):
                caption = lines[j].strip()
                i = j
            if caption:
                out.append(
                    f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}">'
                    f'<figcaption>{inline_markup(caption, bib, cited)}</figcaption></figure>'
                )
            else:
                out.append(
                    f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}"></figure>'
                )
            i += 1
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            flush_para()
            level = len(heading_match.group(1))
            text = inline_markup(heading_match.group(2), bib, cited)
            out.append(f"<h{level}>{text}</h{level}>")
            i += 1
            continue

        para.append(line)
        i += 1

    flush_para()
    return "\n".join(out), cited


def build_html() -> None:
    bib = parse_bib(BIB)
    body, cited = markdown_body(MANUSCRIPT.read_text(encoding="utf-8"), bib)
    cited = sorted(cited, key=lambda key: reference_sort_key(bib[key]) if key in bib else (key, ""))
    refs = "\n".join(
        f"<p>{html.escape(format_reference(bib[key]))}</p>"
        for key in cited
        if key in bib
    )
    body = body.replace("<!-- references inserted here -->", f"<div class=\"refs\">{refs}</div>")

    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(TITLE)}</title>
<style>
@page {{
  size: letter;
  margin: 0.75in 0.72in 0.82in;
}}
body {{
  color: #202124;
  font-family: Arial, Helvetica, sans-serif;
  font-size: 10.5pt;
  line-height: 1.45;
  margin: 0 auto;
  max-width: 760px;
}}
h1 {{
  font-size: 21pt;
  line-height: 1.15;
  margin: 0 0 18px;
}}
h2 {{
  break-after: avoid;
  border-top: 1px solid #c9c9c9;
  font-size: 14pt;
  margin: 26px 0 8px;
  padding-top: 12px;
  page-break-after: avoid;
}}
h3 {{
  break-after: avoid;
  font-size: 12pt;
  margin: 18px 0 6px;
  page-break-after: avoid;
}}
p {{
  margin: 0 0 9px;
}}
code {{
  background: #f3f4f4;
  border-radius: 3px;
  font-family: Consolas, monospace;
  font-size: 9.5pt;
  overflow-wrap: anywhere;
  padding: 1px 3px;
}}
sup {{
  line-height: 0;
}}
.nowrap {{
  white-space: nowrap;
}}
table {{
  break-inside: avoid;
  border-collapse: collapse;
  font-size: 9pt;
  margin: 8px 0 14px;
  page-break-inside: avoid;
  width: 100%;
}}
th, td {{
  border: 1px solid #cfcfcf;
  padding: 5px 6px;
  vertical-align: top;
}}
th {{
  background: #f1f3f4;
  font-weight: 700;
}}
td:not(:first-child), th:not(:first-child) {{
  text-align: right;
}}
figure {{
  break-inside: avoid;
  margin: 14px 0 4px;
  page-break-inside: avoid;
}}
figcaption {{
  break-before: avoid;
  font-size: 9.3pt;
  margin-top: 6px;
  page-break-before: avoid;
}}
img {{
  display: block;
  height: auto;
  max-width: 100%;
}}
.caption {{
  break-after: avoid;
  font-size: 9.3pt;
  page-break-after: avoid;
}}
.equation {{
  font-family: "Cambria Math", "Times New Roman", serif;
  font-size: 12pt;
  margin: 10px 0 14px;
  text-align: center;
}}
.refs {{
  font-size: 9.5pt;
}}
.refs p {{
  margin: 0 0 7px 0.25in;
  text-indent: -0.25in;
}}
</style>
</head>
<body>
{body}
</body>
</html>
"""
    HTML_OUT.write_text(html_text, encoding="utf-8")
    DOC_OUT.write_text(html_text, encoding="utf-8")


def build_pdf() -> None:
    if not EDGE.exists():
        raise SystemExit(f"Microsoft Edge was not found at {EDGE}")

    with tempfile.TemporaryDirectory(prefix="cropcast-edge-pdf-") as profile:
        proc = subprocess.Popen(
            [
                str(EDGE),
                "--headless",
                "--disable-gpu",
                "--remote-debugging-port=0",
                f"--user-data-dir={profile}",
                HTML_OUT.resolve().as_uri(),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            port_file = Path(profile) / "DevToolsActivePort"
            deadline = time.time() + 15
            while time.time() < deadline and not port_file.exists():
                time.sleep(0.1)
            if not port_file.exists():
                raise RuntimeError("Edge did not open a DevTools port.")

            port = port_file.read_text(encoding="utf-8").splitlines()[0]
            page = wait_for_page(port)
            ws = WebSocket(page["webSocketDebuggerUrl"])
            try:
                ws.command("Page.enable")
                time.sleep(0.8)
                result = ws.command(
                    "Page.printToPDF",
                    {
                        "printBackground": True,
                        "displayHeaderFooter": True,
                        "headerTemplate": "<span></span>",
                        "footerTemplate": (
                            "<div style='width:100%;font-size:8px;color:#555;"
                            "text-align:center;'><span class='pageNumber'></span></div>"
                        ),
                        "preferCSSPageSize": True,
                    },
                    timeout=30,
                )
            finally:
                ws.close()
            PDF_OUT.write_bytes(base64.b64decode(result["data"]))
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def wait_for_page(port: str) -> dict[str, str]:
    url = f"http://127.0.0.1:{port}/json/list"
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            pages = json.loads(urllib.request.urlopen(url, timeout=1).read().decode("utf-8"))
        except Exception:
            time.sleep(0.1)
            continue
        for page in pages:
            if page.get("type") == "page" and "webSocketDebuggerUrl" in page:
                return page
        time.sleep(0.1)
    raise RuntimeError("Could not find the Edge page target.")


class WebSocket:
    def __init__(self, ws_url: str) -> None:
        parsed = urlparse(ws_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80
        path = parsed.path
        if parsed.query:
            path += "?" + parsed.query
        self.sock = socket.create_connection((host, port), timeout=10)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(request.encode("ascii"))
        response = b""
        while b"\r\n\r\n" not in response:
            response += self.sock.recv(4096)
        if b" 101 " not in response.split(b"\r\n", 1)[0]:
            raise RuntimeError("WebSocket handshake with Edge failed.")
        self.next_id = 1

    def command(self, method: str, params: dict[str, object] | None = None,
                timeout: int = 10) -> dict[str, object]:
        msg_id = self.next_id
        self.next_id += 1
        self.send_json({"id": msg_id, "method": method, "params": params or {}})
        deadline = time.time() + timeout
        while time.time() < deadline:
            message = json.loads(self.recv_text())
            if message.get("id") != msg_id:
                continue
            if "error" in message:
                raise RuntimeError(message["error"])
            return message.get("result", {})
        raise TimeoutError(f"Timed out waiting for {method}.")

    def send_json(self, payload: dict[str, object]) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        mask = os.urandom(4)
        header = bytearray([0x81])
        length = len(data)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.extend([0x80 | 126, (length >> 8) & 255, length & 255])
        else:
            header.append(0x80 | 127)
            header.extend(length.to_bytes(8, "big"))
        masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(data))
        self.sock.sendall(bytes(header) + mask + masked)

    def recv_text(self) -> str:
        chunks: list[bytes] = []
        while True:
            first = self.recv_exact(2)
            fin = first[0] & 0x80
            opcode = first[0] & 0x0F
            length = first[1] & 0x7F
            if length == 126:
                length = int.from_bytes(self.recv_exact(2), "big")
            elif length == 127:
                length = int.from_bytes(self.recv_exact(8), "big")
            masked = first[1] & 0x80
            mask = self.recv_exact(4) if masked else b""
            payload = self.recv_exact(length)
            if masked:
                payload = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
            if opcode == 8:
                raise RuntimeError("Edge closed the WebSocket.")
            if opcode in (1, 2, 0):
                chunks.append(payload)
            if fin:
                return b"".join(chunks).decode("utf-8")

    def recv_exact(self, n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                raise RuntimeError("WebSocket connection closed.")
            data += chunk
        return data

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def main() -> None:
    build_html()
    build_pdf()
    print(PDF_OUT)


if __name__ == "__main__":
    main()
