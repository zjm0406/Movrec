def get_wish_list(profile_url, cookies=None, max_size=20):
    """
    获取用户的想看电影列表
    Args:
        profile_url: 豆瓣用户主页URL 
        cookies: 豆瓣cookies字符串
        max_size: 最大返回数量，默认20
    Returns:
        list: 想看电影名称列表，最多max_size个
    """
    # 初始化爬虫客户端
    client = DoubanClient(cookies=cookies)
    
    # 从URL提取用户ID
    m = re.search(r"/people/([^/]+)/", profile_url)
    if not m:
        raise ValueError("无效的豆瓣用户主页URL")
    uid = m.group(1)
    
    # 爬取想看列表
    wish_movies = crawl_collect(client, uid, pages=2)  # 爬取2页足够获取20条
    
    # 提取电影名称列表
    wish_list = []
    for movie in wish_movies:
        if movie.get('title'):
            wish_list.append(movie['title'])
            if len(wish_list) >= max_size:
                break
                
    return wish_list