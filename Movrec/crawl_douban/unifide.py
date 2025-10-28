import argparse
import pandas as pd
import numpy as np
from crawl_douban.crawl_douban_me import DoubanClient, crawl_collect
from tabulate import tabulate

# Load the movies dataset and similarity matrix from Doumini_V2
def load_movie_data():
    try:
        movies = pd.read_csv('movies.csv')
        movies_new = movies[movies['DOUBAN_SCORE'] >= 6.5]
        movies_new = movies_new[(movies_new['DOUBAN_VOTES'] >= 3000)].sort_values(
            by=['DOUBAN_SCORE', 'DOUBAN_VOTES'], 
            ascending=[False, False]
        )
        
        # Load similarity matrix (this should be pre-computed)
        similarity = np.load('similarity_matrix.npy')
        
        return movies_new, similarity
    except Exception as e:
        print(f"Error loading movie data: {e}")
        return None, None

def recommend_from_wishlist(movies_new, similarity, wish_list, topk=10):
    """
    Generate recommendations based on user's wish list
    """
    recommendations = []
    
    # Find indices of wish list movies in our dataset
    valid_movies = []
    for movie in wish_list:
        if movie in movies_new['NAME'].values:
            valid_movies.append(movie)
    
    if not valid_movies:
        print("No valid movies found in wish list")
        return None
    
    # Get movie indices
    input_indices = []
    for movie in valid_movies:
        movie_idx = movies_new[movies_new['NAME'] == movie].index[0]
        pos = list(movies_new.index).index(movie_idx)
        input_indices.append(pos)
    
    # Calculate average similarity
    combined_similarity = np.mean([similarity[i] for i in input_indices], axis=0)
    
    # Get top recommendations
    sim_scores = list(enumerate(combined_similarity))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [(i, score) for i, score in sim_scores if i not in input_indices]
    
    # Get top K recommendations
    top_indices = [i for i, score in sim_scores[:topk]]
    top_scores = [score for i, score in sim_scores[:topk]]
    
    # Format recommendations
    for j, sim_score in zip(top_indices, top_scores):
        movie_data = movies_new.iloc[j]
        recommendations.append({
            "排名": len(recommendations) + 1,
            "电影名称": movie_data['NAME'],
            "豆瓣评分": float(movie_data['DOUBAN_SCORE']),
            "相似度": float(f"{sim_score:.3f}"),
            "导演": str(movie_data.get('DIRECTORS', '未知'))[:15],
            "年份": str(movie_data.get('YEAR', '未知')),
        })
    
    return pd.DataFrame(recommendations)

def print_recommendations(recommendations, title="电影推荐结果"):
    """Pretty print recommendations"""
    if recommendations is None or recommendations.empty:
        print("没有推荐结果")
        return
        
    print("\n" + "="*80)
    print(f"🎬 {title}")
    print("="*80)
    
    print(tabulate(
        recommendations,
        headers='keys',
        tablefmt='fancy_grid',
        showindex=False,
        numalign="center",
        stralign="left"
    ))
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="豆瓣电影推荐系统")
    parser.add_argument("--profile", required=True, help="豆瓣用户主页URL")
    parser.add_argument("--cookies", help="豆瓣网站Cookie")
    parser.add_argument("--topk", type=int, default=10, help="推荐数量")
    args = parser.parse_args()
    
    # 1. Initialize Douban client
    client = DoubanClient(cookies=args.cookies)
    
    # 2. Extract user ID from profile URL
    import re
    match = re.search(r"/people/([^/]+)/", args.profile)
    if not match:
        raise ValueError("无效的豆瓣主页URL")
    uid = match.group(1)
    
    # 3. Crawl user's wish list
    print("正在获取用户想看列表...")
    wish_movies = crawl_collect(client, uid, cat="movie", interest="wish", pages=7)  # 7 pages ≈ 100 movies
    
    # 4. Extract movie titles from wish list
    wish_list = []
    for movie in wish_movies[:100]:  # Take first 100 movies
        if movie['title']:
            wish_list.append(movie['title'])
    
    print(f"成功获取 {len(wish_list)} 部想看电影")
    
    # 5. Load movie data and similarity matrix
    print("加载电影数据库...")
    movies_new, similarity = load_movie_data()
    if movies_new is None or similarity is None:
        raise RuntimeError("无法加载电影数据库")
    
    # 6. Generate recommendations
    print("正在生成推荐...")
    recommendations = recommend_from_wishlist(
        movies_new,
        similarity,
        wish_list,
        topk=args.topk
    )
    
    # 7. Print recommendations
    print_recommendations(recommendations)

if __name__ == "__main__":
    main()