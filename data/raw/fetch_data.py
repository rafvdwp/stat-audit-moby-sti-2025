"""
Cara pakai:
  python fetch_data.py
  python fetch_data.py --token YOUR_GITHUB_TOKEN   (untuk rate limit lebih tinggi)
  python fetch_data.py --max-pages 5               (batasi jumlah halaman)
"""

import requests
import pandas as pd
import os
import time
import argparse
from datetime import datetime

# ── Konfigurasi ─────────────────────────────────────────────────────────────
OWNER = "moby"
REPO  = "moby"
BASE_URL = f"https://api.github.com/repos/{OWNER}/{REPO}"
OUTPUT_DIR = "data/raw"

# ── Helper Functions ─────────────────────────────────────────────────────────

def get_headers(token=None):
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_paginated(endpoint, params, headers, max_pages=10, label="data"):
    """Ambil semua halaman dari endpoint GitHub API."""
    all_items = []
    page = 1
    while page <= max_pages:
        params["page"] = page
        params["per_page"] = 100
        resp = requests.get(endpoint, params=params, headers=headers)

        # Handle rate limit
        if resp.status_code == 403:
            reset_time = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset_time - time.time(), 0) + 5
            print(f"  ⚠ Rate limited. Menunggu {int(wait)} detik...")
            time.sleep(wait)
            continue

        if resp.status_code != 200:
            print(f"  ✗ Error {resp.status_code} pada halaman {page}: {resp.json().get('message','')}")
            break

        items = resp.json()
        if not items:
            break

        all_items.extend(items)
        remaining = int(resp.headers.get("X-RateLimit-Remaining", 999))
        print(f"  Halaman {page}/{max_pages} → {len(items)} {label} "
              f"(total: {len(all_items)}, sisa quota: {remaining})")

        if len(items) < 100:
            break
        page += 1
        time.sleep(0.5)  # jaga rate limit

    return all_items


# ── Fetch Issues ─────────────────────────────────────────────────────────────

def fetch_issues(headers, max_pages):
    print(f"\n📥 Mengambil Issues dari {OWNER}/{REPO} ...")
    endpoint = f"{BASE_URL}/issues"
    params = {"state": "all", "sort": "created", "direction": "desc"}
    raw = fetch_paginated(endpoint, params, headers, max_pages, label="issues")

    # Filter: hilangkan pull requests (GitHub API mengembalikan PR di endpoint /issues)
    issues_only = [i for i in raw if "pull_request" not in i]
    print(f"  → {len(issues_only)} issues (setelah filter PR)")
    return issues_only


def parse_issues(raw_issues):
    """Konversi list dict menjadi DataFrame."""
    rows = []
    for i in raw_issues:
        rows.append({
            "id":           i.get("id"),
            "number":       i.get("number"),
            "title":        i.get("title", ""),
            "state":        i.get("state"),
            "created_at":   i.get("created_at"),
            "updated_at":   i.get("updated_at"),
            "closed_at":    i.get("closed_at"),
            "user_login":   i.get("user", {}).get("login"),
            "user_type":    i.get("user", {}).get("type"),
            "comments":     i.get("comments", 0),
            "labels":       ",".join([l["name"] for l in i.get("labels", [])]),
            "assignees":    ",".join([a["login"] for a in i.get("assignees", [])]),
            "milestone":    (i.get("milestone") or {}).get("title"),
            "body_length":  len(i.get("body") or ""),
            "url":          i.get("html_url"),
        })
    return pd.DataFrame(rows)


def clean_issues(df):
    """Bersihkan dan tambah fitur turunan."""
    df = df.copy()
    for col in ["created_at", "updated_at", "closed_at"]:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    df["age_days"] = (
        df["closed_at"].fillna(pd.Timestamp.now(tz="UTC")) - df["created_at"]
    ).dt.days.clip(lower=0)

    df["has_assignee"]  = df["assignees"].str.len() > 0
    df["label_count"]   = df["labels"].apply(lambda x: len(x.split(",")) if x else 0)
    df["is_bot"]        = df["user_login"].str.lower().str.contains("bot", na=False)
    df["month_created"] = df["created_at"].dt.to_period("M").astype(str)
    return df


# ── Fetch Pull Requests ───────────────────────────────────────────────────────

def fetch_pull_requests(headers, max_pages):
    print(f"\n📥 Mengambil Pull Requests dari {OWNER}/{REPO} ...")
    endpoint = f"{BASE_URL}/pulls"
    params = {"state": "all", "sort": "created", "direction": "desc"}
    raw = fetch_paginated(endpoint, params, headers, max_pages, label="PRs")
    return raw


def parse_pull_requests(raw_prs):
    rows = []
    for p in raw_prs:
        rows.append({
            "id":             p.get("id"),
            "number":         p.get("number"),
            "title":          p.get("title", ""),
            "state":          p.get("state"),
            "created_at":     p.get("created_at"),
            "updated_at":     p.get("updated_at"),
            "closed_at":      p.get("closed_at"),
            "merged_at":      p.get("merged_at"),
            "user_login":     p.get("user", {}).get("login"),
            "user_type":      p.get("user", {}).get("type"),
            "draft":          p.get("draft", False),
            "comments":       p.get("comments", 0),
            "review_comments": p.get("review_comments", 0),
            "commits":        p.get("commits", 0),
            "additions":      p.get("additions", 0),
            "deletions":      p.get("deletions", 0),
            "changed_files":  p.get("changed_files", 0),
            "base_branch":    p.get("base", {}).get("ref"),
            "head_branch":    p.get("head", {}).get("ref"),
            "labels":         ",".join([l["name"] for l in p.get("labels", [])]),
            "url":            p.get("html_url"),
        })
    return pd.DataFrame(rows)


def clean_pull_requests(df):
    df = df.copy()
    for col in ["created_at", "updated_at", "closed_at", "merged_at"]:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    df["is_merged"]    = df["merged_at"].notna()
    df["time_to_close"] = (
        df["closed_at"].fillna(pd.Timestamp.now(tz="UTC")) - df["created_at"]
    ).dt.days.clip(lower=0)
    df["net_lines"]    = df["additions"] - df["deletions"]
    df["is_large_pr"]  = df["changed_files"] > 10
    df["month_created"] = df["created_at"].dt.to_period("M").astype(str)
    return df


# ── Fetch Commits ─────────────────────────────────────────────────────────────

def fetch_commits(headers, max_pages):
    print(f"\n📥 Mengambil Commits dari {OWNER}/{REPO} ...")
    endpoint = f"{BASE_URL}/commits"
    params = {"per_page": 100}
    raw = fetch_paginated(endpoint, params, headers, max_pages, label="commits")
    return raw


def parse_commits(raw_commits):
    rows = []
    for c in raw_commits:
        commit_data = c.get("commit", {})
        author      = commit_data.get("author", {}) or {}
        committer   = commit_data.get("committer", {}) or {}
        rows.append({
            "sha":              c.get("sha", "")[:10],
            "message":          commit_data.get("message", "").split("\n")[0][:120],
            "author_name":      author.get("name"),
            "author_email":     author.get("email"),
            "author_date":      author.get("date"),
            "committer_name":   committer.get("name"),
            "committer_date":   committer.get("date"),
            "github_login":     (c.get("author") or {}).get("login"),
            "comment_count":    commit_data.get("comment_count", 0),
            "url":              c.get("html_url"),
        })
    return pd.DataFrame(rows)


def clean_commits(df):
    df = df.copy()
    for col in ["author_date", "committer_date"]:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    df["hour_of_day"]   = df["author_date"].dt.hour
    df["day_of_week"]   = df["author_date"].dt.day_name()
    df["month_created"] = df["author_date"].dt.to_period("M").astype(str)
    df["is_merge"]      = df["message"].str.lower().str.startswith("merge")
    df["msg_length"]    = df["message"].str.len()
    return df


# ── Save CSV ──────────────────────────────────────────────────────────────────

def save(df_raw, df_clean, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw_path   = os.path.join(OUTPUT_DIR, f"{name}_raw.csv")
    clean_path = os.path.join(OUTPUT_DIR, f"{name}_clean.csv")
    df_raw.to_csv(raw_path,   index=False)
    df_clean.to_csv(clean_path, index=False)
    print(f"  ✔ Raw   → {raw_path}  ({len(df_raw)} baris, {df_raw.shape[1]} kolom)")
    print(f"  ✔ Clean → {clean_path} ({len(df_clean)} baris, {df_clean.shape[1]} kolom)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch data dari moby/moby GitHub repo")
    parser.add_argument("--token",     type=str, default=None,
                        help="GitHub Personal Access Token (opsional, tapi disarankan)")
    parser.add_argument("--max-pages", type=int, default=5,
                        help="Jumlah maksimum halaman per endpoint (default: 5 = ~500 item)")
    args = parser.parse_args()

    headers = get_headers(args.token)

    # Cek koneksi & rate limit
    resp = requests.get("https://api.github.com/rate_limit", headers=headers)
    if resp.status_code == 200:
        rl = resp.json()["rate"]
        print(f"🔑 GitHub API → limit: {rl['limit']}, tersisa: {rl['remaining']}, "
              f"reset: {datetime.fromtimestamp(rl['reset']).strftime('%H:%M:%S')}")
        if rl["remaining"] < 50:
            print("⚠ Quota API hampir habis! Gunakan --token untuk quota lebih tinggi.")
    else:
        print("⚠ Tidak bisa cek rate limit. Lanjut...")

    start = time.time()

    # ── Issues ──────────────────────────────────────────────────────────────
    raw_issues   = fetch_issues(headers, args.max_pages)
    df_iss_raw   = parse_issues(raw_issues)
    df_iss_clean = clean_issues(df_iss_raw)
    save(df_iss_raw, df_iss_clean, "issues")

    # ── Pull Requests ────────────────────────────────────────────────────────
    raw_prs      = fetch_pull_requests(headers, args.max_pages)
    df_pr_raw    = parse_pull_requests(raw_prs)
    df_pr_clean  = clean_pull_requests(df_pr_raw)
    save(df_pr_raw, df_pr_clean, "pull_requests")

    # ── Commits ──────────────────────────────────────────────────────────────
    raw_commits   = fetch_commits(headers, args.max_pages)
    df_cm_raw     = parse_commits(raw_commits)
    df_cm_clean   = clean_commits(df_cm_raw)
    save(df_cm_raw, df_cm_clean, "commits")

    elapsed = time.time() - start
    print(f"\n✅ Selesai dalam {elapsed:.1f} detik.")
    print(f"📁 Semua file tersimpan di: {os.path.abspath(OUTPUT_DIR)}/")
    print("\nRingkasan:")
    print(f"  Issues       : {len(df_iss_clean)} baris")
    print(f"  Pull Requests: {len(df_pr_clean)} baris")
    print(f"  Commits      : {len(df_cm_clean)} baris")


if __name__ == "__main__":
    main()