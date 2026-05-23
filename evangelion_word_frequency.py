#!/usr/bin/env python3
"""
Generate a word frequency document from all Evangelion episode data.
Outputs both JSON and HTML.
"""

import json
import os
import html as html_mod
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
EVA_DIR = os.path.join(BASE, "evangelion")
FREQ_PATH = os.path.join(BASE, "japanese_frequency.json")
JLPT_PATH = os.path.join(BASE, "jlpt_frequency.json")
OUTPUT_JSON = os.path.join(BASE, "evangelion", "word_frequency.json")
OUTPUT_HTML = os.path.join(BASE, "evangelion", "word_frequency.html")

FILTER_POS = {"particle", "auxiliary verb", "copula", "conjunction"}
FILTER_WORDS = {
    "の", "に", "は", "を", "が", "で", "と", "も", "か", "な", "よ", "ね",
    "だ", "です", "ます", "た", "て", "る", "い", "する", "いる", "ある",
    "この", "その", "あの", "これ", "それ", "あれ", "ここ", "そこ", "あそこ",
    "私", "僕", "俺", "あなた", "彼", "彼女",
    "ない", "なる", "くる", "来る", "行く", "言う", "思う", "見る",
    "さん", "くん", "ちゃん", "様",
    "何", "どう", "どこ", "いつ", "なぜ",
    "もう", "まだ", "とても", "すごく",
    "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
    "ん", "え", "あ", "うん", "ああ", "えっ", "はい", "いいえ",
    "こと", "もの", "ため", "よう", "ところ", "わけ",
}


def load_reference_data():
    with open(FREQ_PATH) as f:
        general_freq = json.load(f)
    with open(JLPT_PATH) as f:
        jlpt_words = set(json.load(f).keys())
    return general_freq, jlpt_words


def collect_vocabulary():
    word_data = defaultdict(lambda: {
        "readings": defaultdict(int),
        "meanings": defaultdict(int),
        "count": 0,
        "episodes": set(),
        "occurrences": [],
    })

    episode_names = []
    for ep_dir in sorted(os.listdir(EVA_DIR)):
        ep_path = os.path.join(EVA_DIR, ep_dir, "data.json")
        if not os.path.isfile(ep_path):
            continue
        episode_names.append(ep_dir)
        with open(ep_path) as f:
            segments = json.load(f)
        for seg_idx, seg in enumerate(segments):
            for v in seg.get("vocabulary", []):
                word = v["word"]
                entry = word_data[word]
                entry["count"] += 1
                entry["readings"][v.get("reading", "")] += 1
                entry["meanings"][v.get("meaning", "")] += 1
                entry["episodes"].add(ep_dir)
                entry["occurrences"].append({
                    "episode": ep_dir,
                    "index": seg_idx,
                    "text": seg.get("text", ""),
                })

    return word_data, episode_names


def assign_tier(general_rank):
    if general_rank is None:
        return "unlisted"
    if general_rank <= 1000:
        return "top-1k"
    if general_rank <= 3000:
        return "top-3k"
    if general_rank <= 5000:
        return "top-5k"
    if general_rank <= 10000:
        return "top-10k"
    return "10k+"


def build_frequency_list(word_data, general_freq, jlpt_words):
    results = []
    for word, info in word_data.items():
        if word in FILTER_WORDS or len(word) == 0:
            continue
        if all(c in "ぁ-ん" for c in word) and len(word) == 1:
            continue

        top_reading = max(info["readings"], key=info["readings"].get) if info["readings"] else ""
        top_meaning = max(info["meanings"], key=info["meanings"].get) if info["meanings"] else ""
        gen_rank = general_freq.get(word)
        tier = assign_tier(gen_rank)

        results.append({
            "word": word,
            "reading": top_reading,
            "meaning": top_meaning,
            "count": info["count"],
            "episodes": len(info["episodes"]),
            "general_rank": gen_rank,
            "tier": tier,
            "in_jlpt": word in jlpt_words,
            "occurrences": info["occurrences"],
        })

    results.sort(key=lambda x: (-x["count"], x["general_rank"] or 99999))
    return results


def generate_html(results, episode_count):
    tier_order = ["unlisted", "10k+", "top-10k", "top-5k", "top-3k", "top-1k"]
    tier_labels = {
        "top-1k": "Top 1,000 (very common)",
        "top-3k": "Top 3,000",
        "top-5k": "Top 5,000",
        "top-10k": "Top 10,000",
        "10k+": "Beyond 10,000 (rare)",
        "unlisted": "Not in frequency list",
    }
    tier_colors = {
        "top-1k": "#94a3b8",
        "top-3k": "#60a5fa",
        "top-5k": "#34d399",
        "top-10k": "#fbbf24",
        "10k+": "#f87171",
        "unlisted": "#c084fc",
    }
    tier_bg = {
        "top-1k": "#f1f5f9",
        "top-3k": "#eff6ff",
        "top-5k": "#ecfdf5",
        "top-10k": "#fffbeb",
        "10k+": "#fef2f2",
        "unlisted": "#faf5ff",
    }

    ep_display = {}
    for r in results:
        for occ in r.get("occurrences", []):
            ep = occ["episode"]
            if ep not in ep_display:
                num = ep.replace("episode_", "")
                ep_display[ep] = f"Ep {int(num)}"

    def occurrence_links(w):
        occs = w.get("occurrences", [])
        if not occs:
            return ""
        items = []
        seen = set()
        for occ in occs:
            key = (occ["episode"], occ["index"])
            if key in seen:
                continue
            seen.add(key)
            ep_label = ep_display.get(occ["episode"], occ["episode"])
            text = html_mod.escape(occ["text"][:60])
            href = f'index.html#{occ["episode"]}/{occ["index"]}'
            items.append(f'<a href="{href}" class="occ-link" target="_blank">'
                         f'<span class="occ-ep">{ep_label}</span>{text}</a>')
        return (f'<div class="occ-toggle" onclick="this.nextElementSibling.classList.toggle(\'show\')">'
                f'{len(items)} occurrences</div>'
                f'<div class="occ-list">{"".join(items)}</div>')

    by_tier = defaultdict(list)
    for r in results:
        by_tier[r["tier"]].append(r)

    total_words = len(results)
    total_occurrences = sum(r["count"] for r in results)
    jlpt_count = sum(1 for r in results if r["in_jlpt"])
    multi_ep = sum(1 for r in results if r["episodes"] >= 3)

    stats_html = f"""
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-number">{total_words}</div>
        <div class="stat-label">Unique words</div>
      </div>
      <div class="stat-card">
        <div class="stat-number">{total_occurrences:,}</div>
        <div class="stat-label">Total occurrences</div>
      </div>
      <div class="stat-card">
        <div class="stat-number">{episode_count}</div>
        <div class="stat-label">Episodes</div>
      </div>
      <div class="stat-card">
        <div class="stat-number">{jlpt_count}</div>
        <div class="stat-label">JLPT words</div>
      </div>
      <div class="stat-card">
        <div class="stat-number">{multi_ep}</div>
        <div class="stat-label">Appear in 3+ episodes</div>
      </div>
    </div>
    """

    sidebar_links = []
    tier_sections = []
    for tier in tier_order:
        words = by_tier.get(tier, [])
        if not words:
            continue
        label = tier_labels[tier]
        color = tier_colors[tier]
        bg = tier_bg[tier]
        anchor = tier.replace("+", "plus")
        sidebar_links.append(f'<a href="#{anchor}" class="sidebar-link" data-section="{anchor}">{label} ({len(words)})</a>')

        rows = []
        for w in words:
            jlpt_badge = '<span class="badge jlpt">JLPT</span>' if w["in_jlpt"] else ""
            rank_str = f'#{w["general_rank"]:,}' if w["general_rank"] else "—"
            ep_num = w["episodes"]
            ep_bar = f'<span class="ep-count" title="{ep_num} episodes">' + "█" * ep_num + f" {ep_num}</span>"
            occ_html = occurrence_links(w)
            rows.append(f"""
            <tr>
              <td class="word-cell"><span class="word">{w["word"]}</span><span class="reading">{w["reading"]}</span></td>
              <td class="meaning-cell">{w["meaning"]}{jlpt_badge}</td>
              <td class="count-cell">{w["count"]}</td>
              <td class="rank-cell">{rank_str}</td>
              <td class="ep-cell">{ep_bar}</td>
            </tr>
            <tr class="occ-row"><td colspan="5">{occ_html}</td></tr>""")

        tier_sections.append(f"""
        <section id="{anchor}" class="tier-section">
          <h2 style="border-left: 4px solid {color}; padding-left: 12px;">
            {label}
            <span class="tier-count">{len(words)} words</span>
          </h2>
          <table>
            <thead>
              <tr>
                <th>Word</th>
                <th>Meaning</th>
                <th>Count</th>
                <th>Freq Rank</th>
                <th>Episodes</th>
              </tr>
            </thead>
            <tbody>
              {"".join(rows)}
            </tbody>
          </table>
        </section>
        """)

    # Study priority: words appearing 3+ times, not in top-1k general frequency
    study_words = [r for r in results if r["count"] >= 3 and r["tier"] not in ("top-1k",)]
    study_words.sort(key=lambda x: (-x["episodes"], -x["count"]))
    study_rows = []
    for w in study_words[:50]:
        jlpt_badge = '<span class="badge jlpt">JLPT</span>' if w["in_jlpt"] else ""
        tier_badge = f'<span class="badge tier-badge" style="background:{tier_colors[w["tier"]]}">{w["tier"]}</span>'
        occ_html = occurrence_links(w)
        study_rows.append(f"""
        <tr>
          <td class="word-cell"><span class="word">{w["word"]}</span><span class="reading">{w["reading"]}</span></td>
          <td class="meaning-cell">{w["meaning"]}{jlpt_badge}{tier_badge}</td>
          <td class="count-cell">{w["count"]}</td>
          <td class="ep-cell">{w["episodes"]} eps</td>
        </tr>
        <tr class="occ-row"><td colspan="4">{occ_html}</td></tr>""")

    study_section = f"""
    <section id="study-priority" class="tier-section">
      <h2 style="border-left: 4px solid #8b5cf6; padding-left: 12px;">
        Study Priority
        <span class="tier-count">Top 50 recurring words (3+ occurrences, excluding top-1k common words)</span>
      </h2>
      <table>
        <thead>
          <tr><th>Word</th><th>Meaning</th><th>Count</th><th>Spread</th></tr>
        </thead>
        <tbody>{"".join(study_rows)}</tbody>
      </table>
    </section>
    """
    sidebar_links.insert(0, '<a href="#study-priority" class="sidebar-link" data-section="study-priority">Study Priority (Top 50)</a>')
    sidebar_links.insert(0, '<a href="#overview" class="sidebar-link" data-section="overview">Overview</a>')

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evangelion Word Frequency</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #fafafa; color: #1e293b; line-height: 1.7; }}
  .layout {{ display: flex; min-height: 100vh; }}
  .sidebar {{
    width: 260px; position: fixed; top: 0; left: 0; height: 100vh;
    background: #fff; border-right: 1px solid #e2e8f0;
    overflow-y: auto; padding: 20px 0; z-index: 10;
  }}
  .sidebar h3 {{ padding: 0 20px 12px; font-size: 14px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }}
  .sidebar-link {{
    display: block; padding: 8px 20px; color: #475569; text-decoration: none;
    font-size: 13px; border-left: 3px solid transparent; transition: all 0.15s;
  }}
  .sidebar-link:hover {{ background: #f8fafc; color: #1e293b; }}
  .sidebar-link.active {{ border-left-color: #6366f1; color: #4f46e5; background: #eef2ff; font-weight: 600; }}
  .content {{ margin-left: 260px; max-width: 900px; padding: 40px 40px 80px; }}
  h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 8px; }}
  .subtitle {{ color: #64748b; margin-bottom: 32px; font-size: 15px; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 16px; margin-bottom: 40px; }}
  .stat-card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; text-align: center; }}
  .stat-number {{ font-size: 28px; font-weight: 700; color: #4f46e5; }}
  .stat-label {{ font-size: 13px; color: #64748b; margin-top: 4px; }}
  .tier-section {{ margin-bottom: 48px; }}
  h2 {{ font-size: 20px; font-weight: 600; margin-bottom: 16px; }}
  .tier-count {{ font-size: 13px; font-weight: 400; color: #94a3b8; margin-left: 8px; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; border: 1px solid #e2e8f0; }}
  thead {{ background: #f8fafc; }}
  th {{ padding: 10px 14px; text-align: left; font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.3px; }}
  td {{ padding: 10px 14px; border-top: 1px solid #f1f5f9; font-size: 14px; }}
  tr:hover {{ background: #fafafe; }}
  .word-cell {{ white-space: nowrap; }}
  .word {{ font-size: 18px; font-weight: 600; }}
  .reading {{ color: #94a3b8; font-size: 12px; margin-left: 6px; }}
  .meaning-cell {{ color: #475569; max-width: 280px; }}
  .count-cell {{ font-weight: 600; text-align: center; }}
  .rank-cell {{ color: #94a3b8; text-align: center; font-size: 13px; }}
  .ep-cell {{ font-size: 12px; color: #94a3b8; }}
  .ep-count {{ letter-spacing: -1px; }}
  .badge {{ display: inline-block; font-size: 10px; padding: 1px 6px; border-radius: 4px; margin-left: 6px; vertical-align: middle; font-weight: 600; }}
  .jlpt {{ background: #dbeafe; color: #2563eb; }}
  .tier-badge {{ color: #fff; font-size: 9px; }}
  .occ-row td {{ padding: 0 14px 0; border-top: none; }}
  .occ-toggle {{
    font-size: 12px; color: #6366f1; cursor: pointer; padding: 2px 0 6px;
    user-select: none;
  }}
  .occ-toggle:hover {{ text-decoration: underline; }}
  .occ-list {{
    display: none; padding: 4px 0 10px; max-height: 200px; overflow-y: auto;
  }}
  .occ-list.show {{ display: block; }}
  .occ-link {{
    display: block; padding: 4px 8px; margin: 1px 0; border-radius: 4px;
    text-decoration: none; color: #334155; font-size: 13px;
    transition: background 0.1s;
  }}
  .occ-link:hover {{ background: #eef2ff; }}
  .occ-ep {{
    display: inline-block; width: 44px; font-size: 11px; font-weight: 600;
    color: #6366f1; margin-right: 6px; flex-shrink: 0;
  }}
  @media print {{
    .sidebar {{ display: none; }}
    .content {{ margin-left: 0; }}
  }}
  @media (max-width: 768px) {{
    .sidebar {{ display: none; }}
    .content {{ margin-left: 0; padding: 20px; }}
  }}
</style>
</head>
<body>
<div class="layout">
  <nav class="sidebar">
    <h3>Evangelion Vocab</h3>
    {"".join(sidebar_links)}
  </nav>
  <main class="content">
    <h1>Evangelion Word Frequency</h1>
    <p class="subtitle">Vocabulary analysis across {episode_count} episodes of Neon Genesis Evangelion</p>
    <section id="overview">
      <h2>Overview</h2>
      {stats_html}
    </section>
    {study_section}
    {"".join(tier_sections)}
  </main>
</div>
<script>
document.addEventListener("DOMContentLoaded", () => {{
  const links = document.querySelectorAll(".sidebar-link");
  const sections = document.querySelectorAll("section[id]");
  const observer = new IntersectionObserver(entries => {{
    entries.forEach(e => {{
      if (e.isIntersecting) {{
        links.forEach(l => l.classList.remove("active"));
        const active = document.querySelector(`.sidebar-link[data-section="${{e.target.id}}"]`);
        if (active) active.classList.add("active");
      }}
    }});
  }}, {{ rootMargin: "-20% 0px -60% 0px" }});
  sections.forEach(s => observer.observe(s));
}});
</script>
</body>
</html>"""
    return html


def main():
    general_freq, jlpt_words = load_reference_data()
    word_data, episode_names = collect_vocabulary()
    results = build_frequency_list(word_data, general_freq, jlpt_words)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    html = generate_html(results, len(episode_names))
    with open(OUTPUT_HTML, "w") as f:
        f.write(html)

    print(f"Generated {len(results)} words from {len(episode_names)} episodes")
    print(f"  JSON: {OUTPUT_JSON}")
    print(f"  HTML: {OUTPUT_HTML}")

    by_tier = defaultdict(int)
    for r in results:
        by_tier[r["tier"]] += 1
    for tier in ["top-1k", "top-3k", "top-5k", "top-10k", "10k+", "unlisted"]:
        print(f"  {tier}: {by_tier.get(tier, 0)}")


if __name__ == "__main__":
    main()
