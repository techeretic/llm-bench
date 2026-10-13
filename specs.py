"""Technical-spec writing prompts, each with machine-checkable instructions."""
import re

SPECS = [
 {"id": "ratelimit",
  "prompt": """Write a technical design spec for a per-API-key rate limiter that sits in front of an OpenAI-compatible LLM gateway (one gateway process today, possibly 3 replicas later). Limits are on requests/minute AND tokens/minute, and token usage is only known after a streamed response finishes.

Requirements for the document:
- Use exactly these top-level (##) sections, in this order: Overview, Goals, Non-Goals, Design, API, Failure Modes, Open Questions.
- Under API, document every HTTP response header the limiter adds in a markdown table with columns Header | Type | Description.
- Include exactly one Mermaid sequence diagram (```mermaid block).
- Open Questions must be a numbered list.
- Stay under 1400 words.""",
  "sections": ["Overview", "Goals", "Non-Goals", "Design", "API", "Failure Modes", "Open Questions"],
  "max_words": 1400, "mermaid": 1, "table_cols": ["Header", "Type", "Description"], "numbered_in": "Open Questions"},
 {"id": "uploads",
  "prompt": """Write a feature spec for adding resumable, chunked uploads (files up to 50 GB) to an existing self-hosted file-sharing web app that currently does single-request multipart uploads to a Python backend storing files on local disk.

Requirements for the document:
- Use exactly these top-level (##) sections, in this order: Summary, Background, Requirements, Protocol, Data Model, Error Handling, Testing Plan, Rollout.
- Under Error Handling, include a markdown table with columns Code | HTTP Status | Meaning | Client Action.
- Under Data Model, give exactly one JSON Schema (a ```json block) for the upload-session record.
- Requirements must be written as a numbered list where every item uses an uppercase RFC 2119 keyword (MUST, SHOULD, MAY, ...).
- Do not include any implementation code other than the JSON Schema.
- Stay under 1600 words.""",
  "sections": ["Summary", "Background", "Requirements", "Protocol", "Data Model", "Error Handling", "Testing Plan", "Rollout"],
  "max_words": 1600, "json_blocks": 1, "table_cols": ["Code", "HTTP Status", "Meaning", "Client Action"],
  "numbered_in": "Requirements", "rfc2119_in": "Requirements", "only_code": ["json"]},
 {"id": "cli",
  "prompt": """Write a spec for a command-line tool named `cfgdiff` that compares two configuration dumps (YAML, JSON or TOML, possibly different formats) semantically rather than textually, for use in CI pipelines.

Requirements for the document:
- Use exactly these top-level (##) sections, in this order: Purpose, Synopsis, Options, Semantics, Output Formats, Exit Codes, Examples, Out of Scope.
- Synopsis must be a single ```text block.
- Options must be a markdown table with columns Flag | Argument | Default | Description.
- Exit Codes must be a markdown table with columns Code | Meaning.
- Give at least 3 examples, each as a ```console block.
- Do not include any implementation source code in any programming language.
- Stay under 1300 words.""",
  "sections": ["Purpose", "Synopsis", "Options", "Semantics", "Output Formats", "Exit Codes", "Examples", "Out of Scope"],
  "max_words": 1300, "table_cols": ["Flag", "Argument", "Default", "Description"], "table_cols2": ["Code", "Meaning"],
  "min_console": 3, "only_code": ["text", "console", "json", "yaml", "toml", ""]},
 {"id": "pgmigrate",
  "prompt": """Write a migration spec for moving a single PostgreSQL 16 instance (400 GB, ~2k writes/s peak, used by three internal services) to a primary + one streaming replica with automatic failover, with less than 30 seconds of write unavailability during cutover.

Requirements for the document:
- Use exactly these top-level (##) sections, in this order: Context, Target Architecture, Migration Plan, Cutover Runbook, Rollback Plan, Risks, Success Criteria.
- Cutover Runbook and Rollback Plan must each be numbered lists of concrete steps.
- Risks must be a markdown table with columns Risk | Likelihood | Impact | Mitigation, where Likelihood and Impact are each one of Low, Medium, High.
- Success Criteria must be a bulleted list of measurable criteria, each containing at least one number.
- Stay under 1500 words.""",
  "sections": ["Context", "Target Architecture", "Migration Plan", "Cutover Runbook", "Rollback Plan", "Risks", "Success Criteria"],
  "max_words": 1500, "table_cols": ["Risk", "Likelihood", "Impact", "Mitigation"], "numbered_in": "Cutover Runbook",
  "numbered_in2": "Rollback Plan", "lmh": True, "bullets_with_numbers_in": "Success Criteria"},
 {"id": "webhooks",
  "prompt": """Write a protocol spec for outbound webhook delivery from a SaaS product to customer endpoints: event envelope, versioning, HMAC signatures with key rotation, retries with backoff, ordering guarantees, and replay protection.

Requirements for the document:
- Use exactly these top-level (##) sections, in this order: Scope, Terminology, Event Envelope, Signing, Delivery and Retries, Versioning, Security Considerations, Glossary.
- Use uppercase RFC 2119 keywords (MUST, SHOULD, MAY) for every normative statement; use at least 10 of them in total.
- Event Envelope must contain exactly one ```json example of an event.
- Delivery and Retries must include the retry schedule as a markdown table with columns Attempt | Delay | Cumulative Time.
- Glossary must be a bulleted list of terms in alphabetical order, formatted as "- **Term**: definition".
- Stay under 1500 words.""",
  "sections": ["Scope", "Terminology", "Event Envelope", "Signing", "Delivery and Retries", "Versioning", "Security Considerations", "Glossary"],
  "max_words": 1500, "json_blocks": 1, "table_cols": ["Attempt", "Delay", "Cumulative Time"], "min_rfc2119": 10,
  "glossary_alpha": "Glossary"},
 {"id": "jobqueue",
  "prompt": """Write a design spec for a lightweight durable job queue for a homelab: jobs are submitted over HTTP by a few scripts, executed by worker processes on 2 machines, stored in SQLite on one machine, with at-least-once execution, per-job timeouts, retries, and a way to see what failed.

Requirements for the document:
- Use exactly these top-level (##) sections, in this order: Problem, Constraints, Architecture, Job Lifecycle, Storage Schema, HTTP API, Operational Concerns, Alternatives Considered.
- Job Lifecycle must include exactly one Mermaid state diagram (```mermaid block starting with stateDiagram).
- Storage Schema must give the schema as exactly one ```sql block of CREATE TABLE statements.
- HTTP API must be a markdown table with columns Method | Path | Request | Response.
- Alternatives Considered must discuss at least 3 named alternatives, each as a ### subsection.
- Stay under 1500 words.""",
  "sections": ["Problem", "Constraints", "Architecture", "Job Lifecycle", "Storage Schema", "HTTP API", "Operational Concerns", "Alternatives Considered"],
  "max_words": 1500, "mermaid": 1, "mermaid_state": True, "sql_blocks": 1, "table_cols": ["Method", "Path", "Request", "Response"],
  "min_h3_in": ("Alternatives Considered", 3)},
]

RFC = r"\b(MUST NOT|MUST|SHALL NOT|SHALL|SHOULD NOT|SHOULD|RECOMMENDED|MAY|OPTIONAL|REQUIRED)\b"

def _sections(md):
    """Return [(title, body)] for ## headings, ignoring fenced code."""
    out, cur, buf, fence = [], None, [], False
    for line in md.splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
        m = None if fence else re.match(r"^##\s+(?:\d+[.)]?\s*)?(.+?)\s*#*\s*$", line)
        if m and not line.startswith("###"):
            if cur is not None:
                out.append((cur, "\n".join(buf)))
            cur, buf = m.group(1).strip().strip("*"), []
        else:
            buf.append(line)
    if cur is not None:
        out.append((cur, "\n".join(buf)))
    return out

def _blocks(md):
    return re.findall(r"^```([\w-]*)[^\n]*\n(.*?)^```", md, re.S | re.M)

def _tables(text):
    """Header-row cell lists for every markdown table."""
    lines = text.splitlines()
    res = []
    for i in range(len(lines) - 1):
        if lines[i].strip().startswith("|") and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1]):
            res.append([c.strip().strip("*`") for c in lines[i].strip().strip("|").split("|")])
    return res

def _table_rows(text, header):
    lines = text.splitlines()
    for i in range(len(lines) - 1):
        if lines[i].strip().startswith("|") and [c.strip().strip("*`") for c in lines[i].strip().strip("|").split("|")] == header:
            rows = []
            for l in lines[i + 2:]:
                if not l.strip().startswith("|"):
                    break
                rows.append([c.strip() for c in l.strip().strip("|").split("|")])
            return rows
    return []

def words(md):
    return len(re.findall(r"\S+", md))

def check(spec, md):
    """Return {check_name: bool}."""
    r = {}
    secs = _sections(md)
    titles = [t for t, _ in secs]
    body = dict(secs)
    r["sections_exact_order"] = titles == spec["sections"]
    r["word_limit"] = words(md) <= spec["max_words"]
    blocks = _blocks(md)
    langs = [b[0].lower() for b in blocks]
    tables = _tables(md)
    if "mermaid" in spec:
        r["mermaid_count"] = langs.count("mermaid") == spec["mermaid"]
    if spec.get("mermaid_state"):
        mer = [b for l, b in blocks if l.lower() == "mermaid"]
        r["mermaid_is_state"] = bool(mer) and mer[0].strip().startswith("stateDiagram") and "mermaid" in "".join(
            re.findall(r"```mermaid", body.get("Job Lifecycle", "")))
    if "json_blocks" in spec:
        r["json_block_count"] = langs.count("json") == spec["json_blocks"]
    if "sql_blocks" in spec:
        sql = [b for l, b in blocks if l.lower() == "sql"]
        r["sql_block"] = len(sql) == spec["sql_blocks"] and "CREATE TABLE" in sql[0].upper()
    for key in ("table_cols", "table_cols2"):
        if key in spec:
            r[key] = spec[key] in tables
    if "min_console" in spec:
        r["console_examples"] = langs.count("console") >= spec["min_console"]
    if "only_code" in spec:
        r["no_impl_code"] = all(l in spec["only_code"] for l in langs)
    if spec["id"] == "cli":
        syn = body.get("Synopsis", "")
        r["synopsis_text_block"] = len(re.findall(r"^```", syn, re.M)) == 2 and "```text" in syn
    for key in ("numbered_in", "numbered_in2"):
        if key in spec:
            s = body.get(spec[key], "")
            r[f"numbered:{spec[key]}"] = len(re.findall(r"^\s*\d+[.)]\s", s, re.M)) >= 2
    if "rfc2119_in" in spec:
        items = re.findall(r"^\s*\d+[.)]\s+(.*)", body.get(spec["rfc2119_in"], ""), re.M)
        r["rfc2119_every_item"] = bool(items) and all(re.search(RFC, i) for i in items)
    if "min_rfc2119" in spec:
        prose = re.sub(r"```.*?```", "", md, flags=re.S)
        r["rfc2119_count"] = len(re.findall(RFC, prose)) >= spec["min_rfc2119"]
    if spec.get("lmh"):
        rows = _table_rows(md, spec["table_cols"])
        r["risk_levels_valid"] = bool(rows) and all(
            len(x) >= 3 and x[1].strip("* ") in ("Low", "Medium", "High") and x[2].strip("* ") in ("Low", "Medium", "High")
            for x in rows)
    if "bullets_with_numbers_in" in spec:
        items = re.findall(r"^\s*[-*]\s+(.*)", body.get(spec["bullets_with_numbers_in"], ""), re.M)
        r["measurable_criteria"] = bool(items) and all(re.search(r"\d", i) for i in items)
    if "glossary_alpha" in spec:
        items = re.findall(r"^\s*-\s+\*\*(.+?)\*\*\s*:", body.get(spec["glossary_alpha"], ""), re.M)
        r["glossary_alpha_format"] = len(items) >= 3 and [i.lower() for i in items] == sorted(i.lower() for i in items)
    if "min_h3_in" in spec:
        sec, n = spec["min_h3_in"]
        r["alternatives_h3"] = len(re.findall(r"^###\s+\S", body.get(sec, ""), re.M)) >= n
    return r
