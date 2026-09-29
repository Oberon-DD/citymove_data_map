"""Check whether the records' links still resolve.

    python tools/check_links.py

Writes data/link_check.csv with one row per distinct resource_url in the
analysed inventory: the HTTP status, a verdict and the date of the check.

  ok           the page answered (2xx or 3xx)
  dead         the page is gone (404 or 410); the map then links to the
               platform's start page instead (tools/build_map.py)
  error        another HTTP error (403, 500, ...); left as it is
  unreachable  no answer from the checking location. Some platforms only
               answer visitors from certain countries, so this says nothing
               about the link itself; left as it is.

Requests are polite: at most four at a time per site. When a site does not
answer at all, its remaining links are marked unreachable without requesting
each of them. SiStat answers 500 for a table that does not exist (tested with
invented table numbers), so there a 500 counts as dead.
"""
import csv
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from urllib.parse import quote, urlsplit

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / "data" / "WP4_master_full.csv"
OUT = REPO / "data" / "link_check.csv"
UA = "Mozilla/5.0 (compatible; citymove-link-check; +https://github.com/Oberon-DD/citymove_data_map)"
PER_SITE = 4
TIMEOUT = 20


DEAD_ON_500 = {"pxweb.stat.si"}


def fetch(url):
    req = urllib.request.Request(quote(url, safe=":/?#[]@!$&'()*+,;=%"), headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            r.read(1024)
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception as e:  # timeouts, refused connections, DNS and TLS failures
        return type(e).__name__


def verdict(status, site=""):
    if isinstance(status, int):
        if status < 400:
            return "ok"
        if status in (404, 410) or (status == 500 and site in DEAD_ON_500):
            return "dead"
        return "error"
    return "unreachable"


def check_site(urls):
    first = fetch(urls[0])
    if verdict(first) == "unreachable" and verdict(fetch(urls[0])) == "unreachable":
        return {u: first for u in urls}
    with ThreadPoolExecutor(PER_SITE) as ex:
        found = dict(zip(urls, ex.map(fetch, urls)))
    for u, s in found.items():  # one retry for anything that did not answer cleanly
        if verdict(s) != "ok":
            found[u] = fetch(u)
    return found


def main():
    m = pd.read_csv(MASTER, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    urls = sorted(set(m.loc[m["fit"] != "PROPOSED EXIT", "resource_url"]) - {""})
    sites = defaultdict(list)
    for u in urls:
        sites[urlsplit(u).netloc].append(u)
    results = {}
    with ThreadPoolExecutor(8) as ex:
        for found in ex.map(check_site, sites.values()):
            results.update(found)
    today = date.today().isoformat()
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["url", "site", "http_status", "verdict", "checked_on"])
        for u in urls:
            site = urlsplit(u).netloc
            w.writerow([u, site, results[u], verdict(results[u], site), today])
    counts = pd.Series([verdict(s, urlsplit(u).netloc) for u, s in results.items()]).value_counts().to_dict()
    print(f"{len(urls):,} links on {len(sites)} sites: {counts}")
    for site, us in sorted(sites.items(), key=lambda kv: -len(kv[1])):
        v = pd.Series([verdict(results[u], site) for u in us]).value_counts().to_dict()
        if set(v) != {"ok"}:
            print(f"  {site}: {v}")


if __name__ == "__main__":
    main()
