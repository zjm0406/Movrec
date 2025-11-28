
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_douban_covers.py

根据一组豆瓣电影ID（最多 10 个）抓取每部电影的封面图片 URL 列表。
也可选择下载封面到本地目录。

示例用法:
  python get_douban_covers.py --ids 1292052 1295644
  python get_douban_covers.py --ids-file ids.txt --download --out-dir covers

输出:
  - 默认打印 JSON 列表到 stdout（包含每个 id 对应的 cover_url）
  - 若 --download 则把图片保存到指定目录，并在输出中包含本地文件路径
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from typing import List, Optional, Dict

import requests
from bs4 import BeautifulSoup
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type

# 常用 UA，避免被简单的反爬机制拒绝
MOBILE_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

def parse_cookie_str(cookie_str: str) -> dict:
    """把 'a=1; b=2; dbcl2="xxx"' 转成 dict"""
    jar = {}
    if not cookie_str:
        return jar
    for part in cookie_str.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            jar[k.strip()] = v.strip().strip('"')
    return jar

class DoubanClient:
    def __init__(self, cookies: Optional[str] = None, base: str = "https://www.douban.com"):
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

    @retry(wait=wait_fixed(1), stop=stop_after_attempt(3), reraise=True)
    def get(self, url: str, params: Optional[dict] = None, timeout: int = 15) -> requests.Response:
        resp = self.sess.get(url, params=params, timeout=timeout)
        if resp.status_code in (403, 429):
            # 触发重试策略
            raise requests.HTTPError(f"HTTP {resp.status_code}")
        resp.raise_for_status()
        return resp

    def soup(self, url: str, params: Optional[dict] = None) -> BeautifulSoup:
        r = self.get(url, params=params)
        return BeautifulSoup(r.text, "lxml")

def normalize_img_url(url: str) -> str:
    """
    有时豆瓣返回的图片是缩略或带有参数的地址，这里尝试做最小的标准化：
    - 如果是相对 URL，直接返回（不过豆瓣通常给绝对 URL）
    - 返回原始 URL（不做强制替换尺寸，避免出现不可用 URL）
    """
    return url

def extract_cover_from_soup(soup: BeautifulSoup) -> Optional[str]:
    """
    尝试多个选择器以适配不同页面布局：
    - div#mainpic img
    - img[rel="v:image"]
    - img[itemprop="image"]
    - .poster img
    - a.nbgnbg img
    返回第一个匹配到的 src 或 data-src。
    """
    selectors = [
        "div#mainpic img",
        "img[rel='v:image']",
        "img[itemprop='image']",
        ".poster img",
        "a.nbgnbg img",
        "img.poster-img",
    ]
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            for attr in ("src", "data-src", "data-original"):
                v = el.get(attr)
                if v:
                    v = v.strip()
                    # 有些图片 URL 带有缩略参数或前缀，直接返回
                    return normalize_img_url(v)
    # 兜底：尝试从 meta 标签查找大图
    meta = soup.select_one("meta[property='og:image'], meta[name='image']")
    if meta and meta.get("content"):
        return normalize_img_url(meta.get("content").strip())
    return None

def fetch_movie_cover_by_id(client: DoubanClient, movie_id: str) -> Dict[str, Optional[str]]:
    """
    抓取单个电影的封面 URL。
    返回 dict: {"id": movie_id, "cover_url": "...", "error": "..."}
    """
    out = {"id": movie_id, "cover_url": None, "error": None}
    url = f"https://movie.douban.com/subject/{movie_id}/"
    try:
        soup = client.soup(url)
        cover = extract_cover_from_soup(soup)
        if cover:
            out["cover_url"] = cover
        else:
            out["error"] = "cover_not_found"
    except Exception as e:
        out["error"] = f"fetch_error: {e}"
    # 随机短暂停顿，礼貌爬虫
    time.sleep(random.uniform(0.6, 1.5))
    return out

def download_image(url: str, path: str, client: Optional[DoubanClient] = None) -> None:
    """
    下载图片到本地文件（覆盖同名文件）。
    使用 client 的会话可以带 cookie/headers。
    """
    sess = client.sess if client else requests
    # 使用 stream 模式
    try:
        resp = sess.get(url, stream=True, timeout=20)
        resp.raise_for_status()
        with open(path, "wb") as f:
            for chunk in resp.iter_content(1024 * 8):
                if chunk:
                    f.write(chunk)
    except Exception as e:
        raise RuntimeError(f"download_failed: {e}")

def main():
    ap = argparse.ArgumentParser(description="根据豆瓣电影 ID 列表抓取封面图片 URL（最多 10 个 ID）")
    ap.add_argument("--ids", nargs="+", help="一组豆瓣电影 ID，例如: 1292052 1295644", default=[])
    ap.add_argument("--ids-file", help="包含电影 ID 的文本文件（每行一个 ID）")
    ap.add_argument("--cookies", help="可选：浏览器 Cookie 字符串（提升成功率）", default=None)
    ap.add_argument("--download", action="store_true", help="是否把图片下载到本地目录")
    ap.add_argument("--out-dir", default="covers", help="下载图片保存目录（如果启用 --download）")
    ap.add_argument("--max", type=int, default=10, help="最多处理多少个 ID（默认 10）")
    args = ap.parse_args()

    ids: List[str] = []
    if args.ids:
        ids.extend(args.ids)
    if args.ids_file:
        if not os.path.exists(args.ids_file):
            raise SystemExit(f"ids-file 不存在: {args.ids_file}")
        with open(args.ids_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.append(line)

    # 去重与清洗
    ids = [re.sub(r"[^\d]", "", i) for i in ids if i and re.search(r"\d", i)]
    if not ids:
        raise SystemExit("没有提供有效的电影 ID（通过 --ids 或 --ids-file）")
    if len(ids) > args.max:
        print(f"[WARN] 输入 ID 数量 {len(ids)} 超过最大值 {args.max}，将只处理前 {args.max} 个")
        ids = ids[: args.max]

    client = DoubanClient(cookies=args.cookies)

    results = []
    for mid in ids:
        print(f"[INFO] 抓取封面: {mid}")
        res = fetch_movie_cover_by_id(client, mid)
        results.append(res)

    # 如果需要下载
    if args.download:
        os.makedirs(args.out_dir, exist_ok=True)
        for item in results:
            if item.get("cover_url") and not item.get("error"):
                # 构造本地文件名：{id}.{ext}
                m = re.search(r"\.(jpg|jpeg|png|webp|gif)(?:\?|$)", item["cover_url"], re.IGNORECASE)
                ext = m.group(1) if m else "jpg"
                fname = f"{item['id']}.{ext}"
                fpath = os.path.join(args.out_dir, fname)
                try:
                    print(f"  下载 {item['cover_url']} -> {fpath}")
                    download_image(item["cover_url"], fpath, client=client)
                    item["local_path"] = fpath
                except Exception as e:
                    item["error"] = f"download_error: {e}"
            else:
                print(f"  跳过下载（无封面或已有错误）：{item['id']} error={item.get('error')}")

    # 输出 JSON（stdout）
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()