"""Generate map_data.js (the data behind index.html) from the public master.

    python tools/build_map.py

Reads data/WP4_master_full.csv and data/framework_v5_1.csv and writes
map_data.js at the repository root. Only the analysed inventory is mapped
(fit 'Direct match' or 'Relevant proxy'); records excluded after review stay in
the master but are not drawn. Rerun after every change to the master.
"""
import hashlib
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / "data" / "WP4_master_full.csv"
FRAMEWORK = REPO / "data" / "framework_v5_1.csv"
OUT = REPO / "map_data.js"

CITIES = {  # label, country, marker position and zoom used when a city is opened
    "Antwerp": ("Antwerp", "Belgium", 51.2194, 4.4025, 5.0),
    "Rotterdam": ("Rotterdam", "Netherlands", 51.9244, 4.4777, 5.2),
    "Bogota": ("Bogotá", "Colombia", 4.711, -74.0721, 4.8),
    "Ljubljana": ("Ljubljana", "Slovenia", 46.0569, 14.5058, 5.6),
    "Lima": ("Lima", "Peru", -12.0464, -77.0428, 4.8),
    "Kampala": ("Kampala", "Uganda", 0.3476, 32.5825, 5.4),
}
DOMAIN_COLORS = {
    "Total PA": "#6abe6a", "Recreational PA": "#e8607a", "Transport PA": "#f29e4c",
    "Supportive active transport environment": "#1e3a50",
    "Supportive recreational and sports environment": "#9b7fc4",
    "NCDs": "#f7c948", "Air quality": "#43b8a9",
}
# Catalogued platforms carry a pipeline tag in the master; show the name used in the paper's tables.
PLATFORM_NAMES = {
    "df_a": "SiStat (Statistical Office px API)",
    "df_b": "Uganda Bureau of Statistics",
    "df_c": "Stad in Cijfers (indicator portal, Swing viewer)",
    "df_d": "Datos Abiertos Bogota",
    "df_e": "Plataforma Nacional de Datos Abiertos",
    "df_f": "Onderzoek010 / Wijkprofiel (indicator portal, Swing viewer), 2024 export",
    "df_g": "Stad in Kaarten (geoportal)",
    "df_h": "ArcGIS Hub Movilidad (SIMUR)",
    "df_i": "Onderzoek010 / Wijkprofiel (indicator portal, Swing viewer), 2026 export",
    "df_j": "Bogota Como Vamos (survey variable workbook)",
    "df_k": "Lima Como Vamos (survey variable workbook)",
}
DESC_MAX = 700  # characters shown on the map; the master holds the full text

# Record kinds, in display order
DIRECT, DIRECT_SPAN, PROXY_LINKED, CONTEXT = 0, 1, 2, 3


def source_name(raw):
    tag = re.search(r"\((df_[a-k])\)\s*$", raw)
    return PLATFORM_NAMES[tag.group(1)] if tag else raw.strip()


def short(text, n):
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def main():
    m = pd.read_csv(MASTER, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    fw = pd.read_csv(FRAMEWORK, dtype=str, keep_default_na=False)
    live = m[m["fit"] != "PROPOSED EXIT"].copy()

    domains = list(dict.fromkeys(fw["domain"]))
    clusters = list(dict.fromkeys(fw["cluster"]))
    indicators = list(fw["indicator"])
    d_idx = {d: i for i, d in enumerate(domains)}
    c_idx = {c: i for i, c in enumerate(clusters)}
    i_idx = {x: i for i, x in enumerate(indicators)}
    c_dom = dict(zip(fw["cluster"], fw["domain"]))

    sources, spatial, temporal = [], [], []

    def code(lst, value):
        if value not in lst:
            lst.append(value)
        return lst.index(value)

    records = {c: [] for c in CITIES}
    for r in live.itertuples(index=False):
        assert c_dom[r.cluster] == r.domain, (r.uid, r.cluster, r.domain)
        spans = [i_idx[s] for s in r.spans.strip(";").split(";") if s] if r.spans else []
        if r.fit == "Direct match" and r.indicator_direct:
            kind, ind = DIRECT, i_idx[r.indicator_direct]
        elif r.fit == "Direct match":
            assert spans, r.uid
            kind, ind = DIRECT_SPAN, -1
        elif r.proxy_link:
            kind, ind = PROXY_LINKED, i_idx[r.proxy_link]
        else:
            kind, ind = CONTEXT, -1
        url = r.resource_url.strip()
        records[r.city].append([
            r.uid, short(r.title, 300), short(r.description_en, DESC_MAX), code(sources, source_name(r.source)),
            url if re.match(r"^https?://", url) else "", kind, c_idx[r.cluster], ind, spans,
            code(spatial, r.spatial_label), code(temporal, r.temporal_label),
            int(r.year_min) if r.year_min else None, int(r.year_max) if r.year_max else None,
        ])

    payload = {
        "meta": {
            "generated": date.today().isoformat(),
            "master_sha256": hashlib.sha256(MASTER.read_bytes()).hexdigest(),
            "records": int(len(live)),
            "excluded": int((m["fit"] == "PROPOSED EXIT").sum()),
        },
        "domains": [{"name": d, "color": DOMAIN_COLORS[d]} for d in domains],
        "clusters": [{"name": c, "domain": d_idx[c_dom[c]]} for c in clusters],
        "indicators": [{"name": x, "cluster": c_idx[c], "op": op}
                       for x, c, op in zip(fw["indicator"], fw["cluster"], fw["operationalisation"])],
        "sources": sources, "spatial": spatial, "temporal": temporal,
        "cities": {k: {"label": v[0], "country": v[1], "lat": v[2], "lon": v[3], "zoom": v[4],
                       "n": len(records[k])} for k, v in CITIES.items()},
        "records": records,
    }
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    OUT.write_text("/* Generated by tools/build_map.py from data/WP4_master_full.csv. Do not edit by hand. */\n"
                   f"window.MAP_DATA = {body};\n", encoding="utf-8")

    n = sum(len(v) for v in records.values())
    assert n == len(live) == 8071, n
    print(f"wrote {OUT.name}: {n:,} records, {OUT.stat().st_size / 1e6:.2f} MB")
    print({k: len(v) for k, v in records.items()})


if __name__ == "__main__":
    main()
