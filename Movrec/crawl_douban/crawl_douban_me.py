# -*- coding: utf-8 -*-
"""
crawl_douban_me.py
抓取自己的豆瓣主页信息：概要、近期广播、最近标记的电影/图书（分页可控）
用法示例：
  python crawl_douban_me.py --profile https://www.douban.com/people/your_uid/ \
      --cookies 'bid=xxx; dbcl2="xxx"; ck=xxx' \
      --pages 3 --out_prefix douban_me
"""

import re
import time
import random
import argparse
import json
from urllib.parse import urljoin, urlencode

import requests
from bs4 import BeautifulSoup
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type
import pandas as pd


MOBILE_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

def parse_cookie_str(cookie_str: str) -> dict:
    """把 'a=1; b=2; dbcl2="xxx"' 转成 dict"""
    jar = {}
    for part in cookie_str.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            jar[k.strip()] = v.strip()
    return jar

class DoubanClient:
    def __init__(self, cookies: str | None = None, base: str = "https://www.douban.com"):
        self.base = base.rstrip("/") + "/"
        self.sess = requests.Session()
        self.sess.headers.update({
            "User-Agent": MOBILE_UA,
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer": "https://www.douban.com/",
            "Connection": "keep-alive",
        })
        if cookies:
            self.sess.cookies.update(parse_cookie_str(cookies))

    @retry(wait=wait_fixed(2), stop=stop_after_attempt(3), reraise=True)
    def get(self, url: str, params: dict | None = None) -> requests.Response:
        resp = self.sess.get(url, params=params, timeout=15)
        if resp.status_code in (403, 429):
            # 403/429 重试
            raise requests.HTTPError(f"HTTP {resp.status_code}")
        resp.raise_for_status()
        return resp

    def soup(self, path_or_url: str, params: dict | None = None) -> BeautifulSoup:
        url = path_or_url if path_or_url.startswith("http") else urljoin(self.base, path_or_url)
        r = self.get(url, params=params)
        return BeautifulSoup(r.text, "lxml")

def text_or_none(el):
    return el.get_text(strip=True) if el else None

def safe_int(s):
    try:
        return int(re.sub(r"[^\d]", "", s))
    except Exception:
        return None

# =============== 解析器们（可能因页面改版需要微调）================

def parse_profile_summary(soup: BeautifulSoup) -> dict:
    """
    个人主页概要（/people/<uid>/）：
    - 昵称、签名、所在地、关注/被关注/日记/相册/评论/广播等数量（能抓多少抓多少）
    """
    out = {}

    # 昵称（多处可能变化，做些兜底）
    h1 = soup.select_one("#db-usr-profile h1, h1")
    out["nickname"] = text_or_none(h1)

    # 签名 / 个人介绍
    signature = soup.select_one(".signature, .user-intro, #user_intro")
    out["signature_or_intro"] = text_or_none(signature)

    # 地点（有时在 .user-info 内）
    pl_items = soup.select(".user-info .pl")
    for pl in pl_items:
        t = pl.get_text(" ", strip=True)
        if "常居" in t or "所在地" in t:
            out["location"] = t.replace("常居:", "").replace("所在地:", "").strip()

    # 关注/被关注
    # 一般在 .user-opt 或 .user-info 里有链接：关注(x) | 被关注(x)
    follow_links = soup.select('a[href*="/contacts"] , a[href*="/rev_contacts"]')
    for a in follow_links:
        href = a.get("href", "")
        if "/contacts" in href and "/rev_contacts" not in href:
            out["following"] = safe_int(a.get_text())
        elif "/rev_contacts" in href:
            out["followers"] = safe_int(a.get_text())

    # 各种计数（读书/电影/音乐/日记/相册/同城/广播等），页面常见在 .mod .info 或侧边栏
    counters = {}
    for a in soup.select("a[href]"):
        txt = a.get_text(strip=True)
        href = a.get("href", "")
        # 简单启发式匹配，尽量别误伤
        if "movie" in href and any(k in txt for k in ["看过", "在看", "想看", "电影"]):
            counters.setdefault("movie", set()).add(txt)
    out["counters_raw"] = {k: sorted(list(v)) for k, v in counters.items()}

    return out

def parse_collect_grid(soup: BeautifulSoup) -> list[dict]:
    """
    标记页面网格：/people/<uid>/collect?...
    抓条目名、年份/信息、标记时间、条目链接与海报
    """
    rows = []
    # 常见选择器：.grid-view .item
    for it in soup.select(".grid-view .item, .collect-list .item, .interest-list .item"):
        title_el = it.select_one(".title a, .info h2 a, .title a.nbg")
        title = text_or_none(title_el)
        link = title_el.get("href") if title_el else None
        info = text_or_none(it.select_one(".intro, .pub, .info"))
        date = text_or_none(it.select_one(".date, .time, .collect-date"))
        poster_el = it.select_one("img")
        poster = poster_el.get("src") if poster_el else None
        rows.append({"title": title, "info": info, "mark_time": date, "link": link, "poster": poster})
    return rows

# =============== 抓取流程 ===============

def crawl_profile(client: DoubanClient, profile_url: str) -> dict:
    soup = client.soup(profile_url)
    out = parse_profile_summary(soup)
    # 尝试抽取 uid（从 URL 或页面里）
    m = re.search(r"/people/([^/]+)/", profile_url)
    out["uid"] = m.group(1) if m else None
    return out

def crawl_collect(client: DoubanClient, uid: str, pages: int = 2, delay=(1.2, 2.5)) -> list[dict]:
    """
    只爬取用户想看的电影
    """
    all_rows = []
    domain = "https://movie.douban.com"
    base_url = f"{domain}/people/{uid}/wish"

    for p in range(pages):
        params = {"start": p * 15, "sort": "time"}
        try:
            soup = client.soup(base_url, params=params)
            rows = parse_collect_grid(soup)
            all_rows.extend(rows)
        except Exception as e:
            print(f"[WARN] movie/wish 第 {p+1} 页抓取失败：{e}")
        time.sleep(random.uniform(*delay))
    return all_rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, help="你的主页URL，如 https://www.douban.com/people/<uid>/")
    ap.add_argument("--cookies", default=None, help="浏览器复制的cookie整串（建议）")
    ap.add_argument("--pages", type=int, default=2, help="分页数量（广播/标记各抓多少页）")
    ap.add_argument("--out_prefix", default="douban_me", help="输出文件前缀")
    args = ap.parse_args()

    client = DoubanClient(cookies=args.cookies)

    # 主页概要
    prof = crawl_profile(client, args.profile)
    uid = prof.get("uid")
    if not uid:
        # 兜底从URL再取一次
        m = re.search(r"/people/([^/]+)/", args.profile)
        uid = m.group(1) if m else None

    if not uid:
        raise SystemExit("没解析到 uid，请检查 --profile 参数是否类似 https://www.douban.com/people/<uid>/")

    # 广播
    statuses = crawl_statuses(client, uid=uid, pages=args.pages)

    # 电影/图书：想看、在看、看过（各抓若干页）
    movies_wish = crawl_collect(client, uid, cat="movie", interest="wish", pages=args.pages)
    movies_do   = crawl_collect(client, uid, cat="movie", interest="do", pages=args.pages)
    movies_done = crawl_collect(client, uid, cat="movie", interest="collect", pages=args.pages)

    books_wish  = crawl_collect(client, uid, cat="book",  interest="wish", pages=args.pages)
    books_do    = crawl_collect(client, uid, cat="book",  interest="do",   pages=args.pages)
    books_done  = crawl_collect(client, uid, cat="book",  interest="collect", pages=args.pages)

    # 汇总 JSON
    bundle = {
        "profile": prof,
        "statuses": statuses,
        "movies": {"wish": movies_wish, "doing": movies_do, "done": movies_done},
        "books":  {"wish": books_wish,  "doing": books_do,  "done": books_done},
        "meta": {"pages": args.pages}
    }

    # 写 JSON
    json_path = f"{args.out_prefix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    print(f"[OK] 写入 {json_path}")

    # 也导出 CSV（方便筛选）
    def to_csv(rows, name):
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(f"{args.out_prefix}_{name}.csv", index=False, encoding="utf-8-sig")
            print(f"[OK] 写入 {args.out_prefix}_{name}.csv")

    to_csv(statuses, "statuses")
    to_csv(movies_wish, "movies_wish")
    to_csv(movies_do,   "movies_doing")
    to_csv(movies_done, "movies_done")
    to_csv(books_wish,  "books_wish")
    to_csv(books_do,    "books_doing")
    to_csv(books_done,  "books_done")

if __name__ == "__main__":
    main()
