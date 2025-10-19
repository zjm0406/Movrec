#!/usr/bin/env python3
"""
增强 movies.csv 数据集 - 添加短评数据
"""

import pandas as pd
import requests
import time
import random
from bs4 import BeautifulSoup
import os

def parse_movie_reviews(movie_id: str, sess: requests.Session, max_reviews=20) -> list:
    """爬取短评的函数"""
    reviews_list = []
    
    try:
        print(f"    正在爬取短评...", end="")
        
        for page in range(2):  # 爬取前2页
            if len(reviews_list) >= max_reviews:
                break
                
            review_url = f"https://movie.douban.com/subject/{movie_id}/comments"
            params = {'start': page * 20, 'limit': 20, 'status': 'P', 'sort': 'new_score'}
            
            try:
                response = sess.get(review_url, params=params, timeout=15)
                soup = BeautifulSoup(response.text, 'html.parser')
                comment_items = soup.find_all('div', class_='comment-item')
                
                for item in comment_items:
                    if len(reviews_list) >= max_reviews:
                        break
                    comment = item.find('span', class_='short')
                    if comment:
                        review_text = comment.get_text().strip()
                        if len(review_text) > 5:
                            reviews_list.append(review_text)
                
                time.sleep(random.uniform(1, 2))
                
            except Exception as e:
                print(f"第{page+1}页失败: {e}", end=" ")
                continue
                
        print(f"获得 {len(reviews_list)} 条短评")
                
    except Exception as e:
        print(f"爬取失败: {e}")
    
    return reviews_list

def enhance_movies_csv():
    """增强 movies.csv 数据集"""
    print("=== 开始增强 movies.csv 数据集 ===")
    
    # 查找数据集文件
    input_file = "../movies.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ 找不到文件: {input_file}")
        print("正在搜索文件...")
        # 在项目根目录搜索
        for root, dirs, files in os.walk("../.."):
            if "movies.csv" in files:
                input_file = os.path.join(root, "movies.csv")
                break
    
    if not os.path.exists(input_file):
        print("❌ 没有找到 movies.csv 文件")
        return
    
    print(f"找到数据集: {input_file}")
    
    # 读取数据
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(input_file, encoding='gbk')
        except Exception as e:
            print(f"❌ 读取文件失败: {e}")
            return
    
    print(f"数据集信息:")
    print(f"  - 总电影数: {len(df)}")
    print(f"  - 列名: {list(df.columns)}")
    print(f"  - 前3部电影:")
    for i in range(min(3, len(df))):
        movie_name = df.iloc[i].get('NAME', '未知')
        print(f"    {i+1}. {movie_name}")
    
    # 创建增强版数据集
    output_file = "movies_with_reviews.csv"
    
    # 添加短评列（如果不存在）
    if 'reviews' not in df.columns:
        df['reviews'] = ''
    if 'reviews_count' not in df.columns:
        df['reviews_count'] = 0
    
    # 设置会话
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://movie.douban.com/",
    })
    
    # 检查是否有电影ID列
    has_movie_id = any(col in df.columns for col in ['MOVIE_ID', 'movie_id', 'douban_id', 'subject_id'])
    
    if not has_movie_id:
        print("\n⚠️  数据集没有电影ID列，需要先获取豆瓣ID")
        print("由于时间关系，我们先测试已知的电影ID")
        # 使用已知的电影ID测试前3部
        test_ids = ["1291546", "1300267", "1291578"]  # 霸王别姬, 乱世佳人, 独立时代
    else:
        # 获取ID列名
        id_col = [col for col in ['MOVIE_ID', 'movie_id', 'douban_id', 'subject_id'] if col in df.columns][0]
        test_ids = df[id_col].head(3).tolist()
    
    print(f"\n开始增强前 3 部电影的数据...")
    
    enhanced_count = 0
    for i in range(min(3, len(df))):
        movie_name = df.iloc[i].get('NAME', f'电影{i+1}')
        
        # 如果已经有短评数据，跳过
        if pd.notna(df.iloc[i].get('reviews')) and df.iloc[i].get('reviews_count', 0) > 0:
            print(f"{i+1}. {movie_name} - 已有短评数据，跳过")
            continue
        
        if i < len(test_ids) and test_ids[i]:
            movie_id = str(test_ids[i])
            print(f"{i+1}. {movie_name} (ID: {movie_id})")
            
            try:
                reviews = parse_movie_reviews(movie_id, sess, max_reviews=15)
                df.at[i, 'reviews'] = " | ".join(reviews)
                df.at[i, 'reviews_count'] = len(reviews)
                enhanced_count += 1
                
            except Exception as e:
                print(f"  ❌ 失败: {e}")
        else:
            print(f"{i+1}. {movie_name} - 没有可用的电影ID，跳过")
        
        # 延时避免被封
        if i < 2:  # 前2部之后延时
            wait_time = random.uniform(3, 5)
            print(f"    等待 {wait_time:.1f} 秒...")
            time.sleep(wait_time)
    
    #保存结果
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n🎉 增强完成！")
    print(f"✅ 成功增强 {enhanced_count} 部电影的数据")
    print(f"📁 增强版数据集: {output_file}")
    print(f"📊 总电影数: {len(df)}")
    
    # 显示增强结果摘要
    print(f"\n增强结果摘要:")
    enhanced_movies = df[df['reviews_count'] > 0].head(5)
    for _, row in enhanced_movies.iterrows():
        print(f"  - {row['NAME']}: {row['reviews_count']} 条短评")

if __name__ == "__main__":
    enhance_movies_csv()
