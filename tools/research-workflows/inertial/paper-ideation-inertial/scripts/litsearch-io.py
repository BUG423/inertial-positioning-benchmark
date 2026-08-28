#!/usr/bin/env python3
"""litsearch-io.py — IO 领域专用文献检索 (自包含版, v4)

针对深度学习惯性定位 (IO) 领域:
- 不做 CCF-A venue 过滤 (IO 论文跨 cs.RO / cs.CV / cs.LG，过滤会漏 ICRA/IROS/RA-L)
- 三源并发: arXiv + Semantic Scholar + OpenAlex
- 任一源失败不影响其他源；末尾报告每个源的状态 (ok / fail / 限流)

v2 拓宽召回 (解决"搜到的论文太少"):
- **保底 --min (默认 10)**: strict 轮去重后不足 min，自动放宽再搜一轮 (broaden)，
  尽量保证至少 min 篇。
- **arXiv 自适应**: strict 轮用 AND (精确)；broaden 轮切 OR (任一词命中)，
  避免多词 AND 退化成只召回 0-1 篇。
- **OpenAlex 改相关性排序**: 此前按 publication_date 倒序，会把"最新但不相关"
  的论文顶上来 (噪声主因)；改为 relevance_score 排序。
- **Semantic Scholar 抗限流**: 429 退避重试 (1.5s/3s)，并支持可选环境变量
  S2_API_KEY (带 key 限流额度大得多)。
- **合并后按相关性排序**: 用"query 词在标题/摘要的命中数"打分排序 (标题命中加权)，
  而非按年份；最相关的进 top-N，把宽召回引入的噪声压到后面。

v3: arXiv 命中 0 时源内切 OR 自救; S2 退避遵守 Retry-After 并重试 5 次; 修正每源耗时计时。

v4 (加强筛选 + 扩充检索区域):
- **IO 领域再排序 (默认开)**: 本工具是 *惯性定位* 专用，混进来的 visual / LiDAR /
  radar / VIO / SLAM 等"别的里程计家族"是主要噪声。默认按标题里的 off-domain 模态词
  扣分、核心 IO 词加分，把噪声压到后面 (软过滤，保召回)。`--no-io-rerank` 可关。
- **--strict-io 硬过滤**: 标题含 visual/LiDAR/radar/VIO/SLAM 等非纯 IMU 模态词的直接丢弃。
  方向 A/C/D (纯惯性) 建议加；方向 E (多传感器融合) 别加，否则误杀。
- **--also 跨域并搜**: 分号分隔的附加 query，与 --query 结果合并去重再统一重排。
  方向 A (架构搬运) 用它把源领域 (CV/NLP) 的方法名一起搜进来，"尽量都搜"。
- **--exclude**: 逗号分隔的额外排除词，标题命中按 off-domain 处理。

用法:
    python3 litsearch-io.py --query "inertial odometry knowledge distillation" --min 10
    python3 litsearch-io.py --query "IMU domain generalization unseen device" --years 3 --limit 20 --json
    python3 litsearch-io.py --query "..." --broaden        # 直接宽口径, 跳过 strict 轮
    python3 litsearch-io.py --query "inertial odometry IMU pedestrian Mamba" \
        --also "Mamba state space sequence model long sequence" --strict-io --min 12

退出码: 0 = 至少一个源成功；2 = 全部源失败 (调用方应转内嵌知识库 fallback)
"""

import argparse
import datetime
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

TIMEOUT = 15
UA = "litsearch-io/2.0 (academic literature search; mailto:research@example.org)"
# 相关性打分时忽略的停用词 (避免 "of/for/the" 这种污染命中计数)
STOP = {"the", "a", "an", "of", "for", "to", "and", "in", "on", "via", "with",
        "using", "based", "from", "by", "at", "as", "is", "are", "deep", "learning"}

# IO 领域再排序词表: 本工具是 *惯性定位* 专用，默认就该向纯 IMU 倾斜。
# off-domain = "别的里程计家族" (视觉/激光/雷达惯性) 的标志词，IO 纯惯性论文标题几乎不出现这些；
# core = 纯惯性定位的核心词。只看标题 (标题是可靠信号，摘要里提一句 "compared to visual" 不算)。
OFF_DOMAIN = {"visual", "vision", "camera", "rgb", "rgbd", "lidar", "laser",
              "radar", "vio", "lio", "loam", "slam"}
CORE_IO = {"inertial", "imu", "odometry", "pedestrian", "pdr",
           "navigation", "accelerometer", "gyroscope"}
OFF_PENALTY = 3   # 标题每命中一个 off-domain 模态词扣 3
CORE_BOOST = 1    # 标题每命中一个核心 IO 词加 1


def _get(url, headers=None, retries=3, backoff=(1.5, 3.0, 5.0)):
    """带 429 退避重试的 GET。429 时优先遵守服务端 Retry-After 头，否则按 backoff
    序列退避 (单次上限 8s，避免把整体拖太久)；其他错误立即抛出。"""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return resp.read().decode("utf-8", errors="replace"), resp.status
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429 and attempt < retries - 1:
                ra = e.headers.get("Retry-After") if e.headers else None
                try:
                    delay = float(ra) if ra else backoff[min(attempt, len(backoff) - 1)]
                except (TypeError, ValueError):
                    delay = backoff[min(attempt, len(backoff) - 1)]
                time.sleep(min(delay, 8.0))
                continue
            raise
    raise last  # pragma: no cover


def _terms(query):
    """query 分词 (去停用词)，用于相关性打分与 arXiv OR 扩召回。"""
    return [t for t in re.split(r"\s+", query.strip()) if t and t.lower() not in STOP]


def _norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def _relevance(paper, terms):
    """相关性打分 = 标题+摘要里命中的 query 词数，标题命中额外加权。"""
    if not terms:
        return 0
    title = (paper.get("title") or "").lower()
    hay = (title + " " + (paper.get("abstract") or "")).lower()
    body_hits = sum(1 for t in terms if t.lower() in hay)
    title_hits = sum(1 for t in terms if t.lower() in title)
    return body_hits + title_hits


def _io_adjust(paper, extra_off=frozenset()):
    """IO 领域再排序增量: 标题里每个 off-domain (visual/lidar/radar/vio...) 命中扣分，
    每个核心 IO 词命中加分。返回 (adj, off_hits, core_hits)。只看标题。"""
    title_tokens = set(re.split(r"[^a-z0-9]+", (paper.get("title") or "").lower()))
    off = len(title_tokens & (OFF_DOMAIN | set(extra_off)))
    core = len(title_tokens & CORE_IO)
    return CORE_BOOST * core - OFF_PENALTY * off, off, core


# ---------------------------------------------------------------- arXiv
def fetch_arxiv(query, years, limit, broaden=False):
    cutoff = datetime.datetime.now().year - years
    # arXiv 的 all: 前缀只作用于紧跟的一个 token；多词必须各自加前缀。
    # strict 轮用 AND (精确，但词多时容易召回 0-1 篇)；broaden 轮用 OR (任一词命中，
    # 大幅扩召回)，再靠最终相关性排序把最相关的顶回前面。
    terms = _terms(query) or ["inertial", "odometry"]
    ns = {"a": "http://www.w3.org/2005/Atom"}

    def _query(joiner):
        search_q = joiner.join(f"all:{urllib.parse.quote(t)}" for t in terms)
        url = (
            f"http://export.arxiv.org/api/query?search_query={search_q}"
            f"&start=0&max_results={limit}&sortBy=relevance&sortOrder=descending"
        )
        body, _ = _get(url)
        root = ET.fromstring(body)
        rows = []
        for e in root.findall("a:entry", ns):
            published = (e.findtext("a:published", default="", namespaces=ns) or "")[:4]
            if published and published.isdigit() and int(published) < cutoff:
                continue
            raw_id = e.findtext("a:id", default="", namespaces=ns) or ""
            arxiv_id = raw_id.rsplit("/abs/", 1)[-1].split("v")[0] if "/abs/" in raw_id else ""
            authors = [a.findtext("a:name", default="", namespaces=ns)
                       for a in e.findall("a:author", ns)]
            rows.append({
                "title": (e.findtext("a:title", default="", namespaces=ns) or "").strip().replace("\n", " "),
                "year": published,
                "venue": "arXiv",
                "arxiv_id": arxiv_id,
                "authors": [a for a in authors if a][:6],
                "abstract": (e.findtext("a:summary", default="", namespaces=ns) or "").strip().replace("\n", " "),
                "url": raw_id,
                "source": "arxiv",
            })
        return rows

    out = _query("+OR+" if broaden else "+AND+")
    # 源内自救: strict 轮多词 AND 极易召回 0 篇 (词一多交集就空)。命中 0 且本可放宽时，
    # arXiv 源内直接切 OR 再搜一轮——不依赖全局 broaden (否则其他源凑够 min 时，
    # arXiv 的 0 召回会被悄悄掩盖)。OR 引入的噪声由下游相关性排序 + top-N 截断兜底。
    if not broaden and not out and len(terms) > 1:
        out = _query("+OR+")
    return out


# ------------------------------------------------------- Semantic Scholar
def fetch_s2(query, years, limit, broaden=False):
    cutoff = datetime.datetime.now().year - years
    fields = "title,year,venue,abstract,externalIds,authors"
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search?"
        + urllib.parse.urlencode({
            "query": query,
            "limit": limit,
            "fields": fields,
            "year": f"{cutoff}-",
        })
    )
    # 可选 API key: 带 key 限流额度大得多，显著缓解 429。
    headers = {}
    key = os.environ.get("S2_API_KEY")
    if key:
        headers["x-api-key"] = key
    # S2 匿名额度极易 429；多给几次退避机会 (Retry-After 优先，见 _get)。
    body, _ = _get(url, headers=headers, retries=5)
    data = json.loads(body)
    out = []
    for p in data.get("data", []) or []:
        ext = p.get("externalIds") or {}
        out.append({
            "title": (p.get("title") or "").strip(),
            "year": str(p.get("year") or ""),
            "venue": p.get("venue") or "",
            "arxiv_id": ext.get("ArXiv", "") or "",
            "authors": [a.get("name", "") for a in (p.get("authors") or [])][:6],
            "abstract": (p.get("abstract") or "").strip(),
            "url": f"https://www.semanticscholar.org/paper/{p.get('paperId', '')}",
            "source": "s2",
        })
    return out


# ------------------------------------------------------------- OpenAlex
def fetch_openalex(query, years, limit, broaden=False):
    cutoff = datetime.datetime.now().year - years
    url = (
        "https://api.openalex.org/works?"
        + urllib.parse.urlencode({
            "search": query,
            "per-page": min(limit, 200),
            "filter": f"from_publication_date:{cutoff}-01-01",
            "sort": "relevance_score:desc",  # v2: 相关性排序 (此前 publication_date 排 = 噪声主因)
        })
    )
    body, _ = _get(url)
    data = json.loads(body)
    out = []
    for w in data.get("results", []) or []:
        ids = w.get("ids") or {}
        arxiv_id = ""
        for loc in (w.get("locations") or []):
            src = (loc.get("source") or {})
            if src and "arxiv" in (src.get("display_name", "") or "").lower():
                landing = loc.get("landing_page_url", "") or ""
                m = re.search(r"(\d{4}\.\d{4,5})", landing)
                if m:
                    arxiv_id = m.group(1)
        authors = [a.get("author", {}).get("display_name", "")
                   for a in (w.get("authorships") or [])]
        out.append({
            "title": (w.get("title") or "").strip(),
            "year": str(w.get("publication_year") or ""),
            "venue": (((w.get("primary_location") or {}).get("source") or {}).get("display_name")) or "",
            "arxiv_id": arxiv_id,
            "authors": [a for a in authors if a][:6],
            "abstract": "",  # OpenAlex 仅给倒排索引，省略
            "url": ids.get("doi") or w.get("id", ""),
            "source": "openalex",
        })
    return out


SOURCES = {"arxiv": fetch_arxiv, "s2": fetch_s2, "openalex": fetch_openalex}


def _gather(query, years, limit, sources, broaden):
    """并发拉一轮，返回 (papers, status)。"""
    results, status = [], {}
    starts = {}
    with ThreadPoolExecutor(max_workers=len(sources)) as ex:
        futs = {}
        for s in sources:
            starts[s] = time.time()  # 在提交时记录，as_completed 后再记会量到 ~0s
            futs[ex.submit(SOURCES[s], query, years, limit, broaden)] = s
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                papers = fut.result()
                results.extend(papers)
                status[name] = {"state": "ok", "n": len(papers)}
            except urllib.error.HTTPError as e:
                status[name] = {"state": "fail", "error": f"HTTP {e.code}" + (" 限流" if e.code == 429 else "")}
            except Exception as e:  # noqa: BLE001
                status[name] = {"state": "fail", "error": f"{type(e).__name__}: {e}"}
            status[name]["elapsed"] = round(time.time() - starts[name], 1)
    return results, status


def _dedupe(results):
    """按标题去重 (保留信息更全的：有 abstract 优先)。"""
    seen, deduped = {}, []
    for p in sorted(results, key=lambda x: (0 if x.get("abstract") else 1)):
        key = _norm_title(p["title"])
        if not key or key in seen:
            continue
        seen[key] = True
        deduped.append(p)
    return deduped


def run(query, years, limit, sources, min_results, broaden=False,
        io_rerank=True, strict_io=False, exclude=None):
    terms = _terms(query)
    rounds = []

    # 第一轮
    results, status = _gather(query, years, limit, sources, broaden)
    rounds.append("broaden" if broaden else "strict")
    deduped = _dedupe(results)

    # 保底: 去重后不足 min，自动放宽再搜一轮 (arXiv 切 OR + 年限翻倍 + 更大 limit)
    if len(deduped) < min_results and not broaden:
        wider_years = years * 2
        wider_limit = max(limit * 2, min_results * 3)
        more, status2 = _gather(query, wider_years, wider_limit, sources, broaden=True)
        results += more
        # 状态合并: 扩展轮成功的源覆盖第一轮失败的源 (例如 S2 第一轮 429、扩展轮重试成功)
        for k, v in status2.items():
            if k not in status or status[k]["state"] != "ok":
                status[k] = {**v, "via": "broadened"}
        deduped = _dedupe(results)
        rounds.append(f"broadened(years={wider_years})")

    # v4: IO 领域再排序 (默认开) + 可选硬过滤。final_score = 相关性 + IO 领域增量。
    # 软模式: off-domain 论文被压到后面但保留 (保召回)；strict 模式: 标题带 off-domain
    # 模态词的直接丢弃 (纯惯性方向用)。
    extra_off = {t.lower() for t in (exclude or [])}
    scored = []
    dropped = 0
    noise = 0
    for p in deduped:
        rel = _relevance(p, terms)
        # 相关性地板: 标题+摘要里一个 query 词都没命中 = 纯噪声 (多见于 OpenAlex 的
        # 模糊 search 召回，如毫不相关的综述)。有查询词时直接丢，别让它占 top-N。
        if terms and rel == 0:
            noise += 1
            continue
        adj, off, core = _io_adjust(p, extra_off) if io_rerank else (0, 0, 0)
        if strict_io and off > 0:
            dropped += 1
            continue  # 标题含 visual/LiDAR/radar/VIO 等非纯 IMU 模态词 → 丢弃
        p["relevance"] = rel
        p["io_adjust"] = adj
        p["off_domain_hits"] = off
        p["final_score"] = rel + adj
        scored.append(p)

    # 按 final_score 排序 (而非年份)，同分按年份新者优先；最相关进 top-N
    scored.sort(key=lambda p: (p["final_score"], p.get("year", "")), reverse=True)
    tags = []
    if noise:
        tags.append(f"noise(drop={noise})")
    if strict_io and dropped:
        tags.append(f"strict-io(drop={dropped})")
    rounds += tags
    return scored[:limit], status, rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--also", default="",
                    help="附加检索词 (分号分隔)，与 --query 结果合并去重再统一重排 "
                         "(方向 A 跨域搜源领域 CV/NLP 方法名用)")
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--min", type=int, default=10,
                    help="保底篇数：strict 轮去重后不足此数则自动放宽再搜一轮")
    ap.add_argument("--broaden", action="store_true",
                    help="直接用 OR 宽口径搜 (跳过 strict 轮)")
    ap.add_argument("--strict-io", dest="strict_io", action="store_true",
                    help="硬过滤：标题含 visual/LiDAR/radar/VIO/SLAM 等非纯 IMU 模态词的直接丢弃 "
                         "(方向 A/C/D 纯惯性建议加；方向 E 融合别加，会误杀)")
    ap.add_argument("--no-io-rerank", dest="no_io_rerank", action="store_true",
                    help="关闭 IO 领域再排序 (默认开启，向纯 IMU 倾斜)")
    ap.add_argument("--exclude", default="",
                    help="额外排除词 (逗号分隔)，标题命中按 off-domain 处理")
    ap.add_argument("--source", default="all", help="all | arxiv,s2,openalex")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    sources = list(SOURCES) if args.source == "all" else [
        s.strip() for s in args.source.split(",") if s.strip() in SOURCES
    ]
    if not sources:
        print("无有效源", file=sys.stderr)
        sys.exit(2)

    queries = [args.query] + [q.strip() for q in args.also.split(";") if q.strip()]
    exclude = [t.strip() for t in args.exclude.split(",") if t.strip()]
    io_rerank = not args.no_io_rerank

    t0 = time.time()
    per_query_lists, status, rounds = [], {}, []
    for q in queries:
        papers_q, status_q, rounds_q = run(
            q, args.years, args.limit, sources, args.min,
            args.broaden, io_rerank, args.strict_io, exclude)
        per_query_lists.append(papers_q)
        rounds.append(f"[{q[:24]}] {' → '.join(rounds_q)}")
        for k, v in status_q.items():
            if k not in status or status[k]["state"] != "ok":
                status[k] = v

    if len(per_query_lists) > 1:
        # 平衡合并: 全局按标题去重 (保最高分实例)，再按 query 轮转交错取——否则宽
        # --also 查询 (源领域 CV/NLP) 会把 IO 本体结果整体挤出 top-N。轮转保证每条
        # query 的高分结果都有代表 (query0[0], query1[0], query0[1], query1[1], ...)。
        best = {}
        for lst in per_query_lists:
            for p in lst:
                k = _norm_title(p.get("title", ""))
                if k and (k not in best or p.get("final_score", 0) > best[k].get("final_score", 0)):
                    best[k] = p
        papers, seen = [], set()
        maxlen = max((len(l) for l in per_query_lists), default=0)
        for i in range(maxlen):
            for lst in per_query_lists:
                if i < len(lst):
                    k = _norm_title(lst[i].get("title", ""))
                    if k and k not in seen:
                        seen.add(k)
                        papers.append(best[k])
                if len(papers) >= args.limit:
                    break
            if len(papers) >= args.limit:
                break
    else:
        papers = (per_query_lists[0] if per_query_lists else [])[:args.limit]

    elapsed = round(time.time() - t0, 1)
    all_failed = all(v["state"] == "fail" for v in status.values())

    if args.json:
        print(json.dumps({
            "query": args.query,
            "also": queries[1:],
            "years": args.years,
            "min_results": args.min,
            "io_rerank": io_rerank,
            "strict_io": args.strict_io,
            "rounds": rounds,
            "elapsed_sec": elapsed,
            "sources": status,
            "n_papers": len(papers),
            "met_min": len(papers) >= args.min,
            "papers": papers,
        }, ensure_ascii=False, indent=2))
    else:
        qline = args.query + (f"  (+{len(queries) - 1} also)" if len(queries) > 1 else "")
        print(f"检索 query: {qline}  (近 {args.years} 年)"
              + ("  [strict-io]" if args.strict_io else "")
              + ("  [io-rerank off]" if not io_rerank else ""))
        print(f"用时 {elapsed}s  命中 {len(papers)} 篇 (去重后)  "
              f"轮次: {' ; '.join(rounds)}  保底 min={args.min}: "
              f"{'达标 ✅' if len(papers) >= args.min else '未达标 ⚠️ (建议手动补搜或转知识库 fallback)'}")
        for name, st in status.items():
            line = f"  - {name}: {st['state']} ({st.get('elapsed', '?')}s)"
            if st["state"] == "ok":
                line += f"  {st['n']} 篇"
                if st.get("via") == "broadened":
                    line += " (扩展轮救回)"
            else:
                line += f"  {st.get('error', '')}"
            print(line)
        print("-" * 60)
        for i, p in enumerate(papers, 1):
            aid = f"  arXiv:{p['arxiv_id']}" if p.get("arxiv_id") else ""
            score = p.get("final_score", p.get("relevance", 0))
            flag = " ⚠off-domain" if p.get("off_domain_hits") else ""
            print(f"[{i}] (score={score}){flag} {p['title']} ({p.get('year', '?')}, {p.get('venue', '?')}){aid}")

    sys.exit(2 if all_failed else 0)


if __name__ == "__main__":
    main()
