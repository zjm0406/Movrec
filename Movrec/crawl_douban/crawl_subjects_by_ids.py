# -*- coding: utf-8 -*-
"""
按 Douban subject id 抓取电影详情，输出 CSV
用法示例：
  python crawl_subjects_by_ids.py --ids 1291858,1292262,1291999 --cookies 'll=...; bid=...; dbcl2="..."; ck=...'
或从文件读：
  python crawl_subjects_by_ids.py --ids-file ids.txt --cookies '...'
"""

import argparse
import csv
import random
import re
import time
from typing import List, Dict

import requests
from bs4 import BeautifulSoup

import sys
import io

# 修改标准输出流编码为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

MOBILE_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

def parse_cookie_str(cookie_str: str) -> dict:
    jar = {}
    if not cookie_str:
        return jar
    for part in cookie_str.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            jar[k.strip()] = v.strip()
    return jar

def get_soup(sess: requests.Session, url: str) -> BeautifulSoup:
    r = sess.get(url, timeout=20)
    r.raise_for_status()
    if not r.encoding or r.encoding.lower() == "iso-8859-1":
        r.encoding = r.apparent_encoding
    return BeautifulSoup(r.text, "lxml")

def text(el):
    return el.get_text(strip=True) if el else None

def join_texts(els):
    return " / ".join([e.get_text(strip=True) for e in els]) if els else None

def parse_subject_page(soup: BeautifulSoup) -> Dict[str, str]:
    # 标题 + 年份
    title_el = soup.select_one('span[property="v:itemreviewed"]') or soup.select_one("h1 span:nth-of-type(1)")
    year_el = soup.select_one("span.year")
    title = text(title_el)
    year = text(year_el).strip("()") if year_el else None

    # 评分 + 票数
    rating = text(soup.select_one("strong.ll.rating_num"))
    votes = text(soup.select_one("span[property='v:votes']"))

    # 导演 / 类型
    directors = join_texts(soup.select("a[rel='v:directedBy']"))
    genres = join_texts(soup.select("span[property='v:genre']"))

    # 片长
    durations = join_texts(soup.select("span[property='v:runtime']"))

    # 国家/地区、又名等在“info”块里（用正则兜底）
    info_text = text(soup.select_one("#info")) or ""
    def field(regex):
        m = re.search(regex, info_text, flags=re.S)
        if not m: return None
        value = m.group(1)
        # 去掉多余换行与空白
        return re.sub(r"\s+", " ", value.strip())

    countries = field(r"制片国家/地区:\s*([^\n]+)")
    aka = field(r"又名:\s*([^\n]+)")

    return {
        "title": title,
        "year": year,
        "rating": rating,
        "votes": votes,
        "directors": directors,
        "genres": genres,
        "durations": durations,
        "countries": countries,
        "aka": aka,
    }

def parse_movie_reviews(soup: BeautifulSoup, movie_id: str, sess: requests.Session, max_reviews=5) -> List[str]:
    """
    爬取指定电影的短评
    现在默认最多保留 5 条（修改点①：默认值从 30 改为 5）
    """
    reviews_list = []
    try:
        # 只爬第一页即可（修改点②：range(2) -> range(1)）
        for page in range(1):
            if len(reviews_list) >= max_reviews:
                break

            review_url = f"https://movie.douban.com/subject/{movie_id}/comments"
            # 这里留着参数以便将来扩展翻页
            params = {
                'start': page * 20,
                'limit': 20,
                'status': 'P',
                'sort': 'new_score'
            }

            try:
                # 简化用法：复用 get_soup 获取第一页
                review_soup = get_soup(sess, review_url)

                # 提取短评内容
                comment_items = review_soup.find_all('div', class_='comment-item')

                for item in comment_items:
                    if len(reviews_list) >= max_reviews:
                        break

                    comment = item.find('span', class_='short')
                    if comment:
                        review_text = comment.get_text().strip()
                        # 过滤太短或无效内容
                        if (len(review_text) > 5 and
                            not review_text.startswith('未通过验证') and
                            '账号异常' not in review_text):
                            reviews_list.append(review_text)

                # 随机延时，尽量温和
                time.sleep(random.uniform(1, 2))

            except Exception as e:
                print(f"  短评第{page+1}页爬取失败: {e}")
                continue

    except Exception as e:
        print(f"爬取电影 {movie_id} 短评时出错: {e}")

    # 最终只保留最多 max_reviews 条
    return reviews_list[:max_reviews]

def load_ids(args) -> List[str]:
    ids: List[str] = []
    if args.ids:
        ids.extend([x.strip() for x in args.ids.split(",") if x.strip()])
    if args.ids_file:
        with open(args.ids_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip().strip(",")
                if s:
                    ids.append(s)
    # 去重保序
    seen = set()
    uniq = []
    for s in ids:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", help="逗号分隔的 Douban subject id 列表，如 1291858,1292262,...")
    ap.add_argument("--ids-file", help="包含多个 subject id 的文本文件（每行一个）")
    ap.add_argument("--cookies", default=None, help="从已登录浏览器复制的 cookie 字符串")
    ap.add_argument("--out", default="subjects_result.csv", help="输出 CSV 文件名")
    ap.add_argument("--delay-min", type=float, default=2.0)
    ap.add_argument("--delay-max", type=float, default=4.0)
    args = ap.parse_args()

    ids = load_ids(args)
    if not ids:
        print("请通过 --ids 或 --ids-file 提供至少一个 subject id")
        return

    sess = requests.Session()
    sess.headers.update({
        "User-Agent": MOBILE_UA,
        "Referer": "https://movie.douban.com/",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    })
    if args.cookies:
        sess.cookies.update(parse_cookie_str(args.cookies))

    out_fields = ["subject_id","title","year","rating","votes","directors","genres","durations","countries","aka","url","reviews","reviews_count"]
    rows = []

    for sid in ids[:20]:
        url = f"https://movie.douban.com/subject/{sid}/"
        try:
            soup = get_soup(sess, url)
            info = parse_subject_page(soup)
            info["subject_id"] = sid
            info["url"] = url

            # === 短评：最多 5 条 ===
            print(f"  正在爬取短评...", end="")
            reviews = parse_movie_reviews(soup, sid, sess, max_reviews=5)  # 修改点③：明确传入 5
            info["reviews"] = " | ".join(reviews)
            info["reviews_count"] = len(reviews)
            print(f"获得 {len(reviews)} 条短评")

            rows.append(info)
            print(f"[OK] {sid} {info.get('title')} ({info.get('year')}) 评分:{info.get('rating')} 票数:{info.get('votes')}")
        except Exception as e:
            print(f"[WARN] 抓取失败 {sid}: {e}")
        time.sleep(random.uniform(args.delay_min, args.delay_max))

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[DONE] 共写入 {len(rows)} 条 -> {args.out}")

if __name__ == "__main__":
    main()
