import tkinter as tk
from tkinter import messagebox
import pandas as pd

# 读取本地电影数据库（确保路径正确）
try:
    movie_db = pd.read_excel("C:/Users/wangyucheng/.spyder-py3/movie_database.xlsx")
except FileNotFoundError:
    messagebox.showerror("错误", "找不到电影数据库文件，请检查路径是否正确！")
    movie_db = None

# 定义推荐函数（输入用户关键词，返回推荐结果）
def recommend_movies(user_genre, user_actor, user_liked_movies):
    if movie_db is None:
        return "数据库加载失败，无法推荐"
    
    # 复制一份数据进行操作，避免修改原始数据
    temp_db = movie_db.copy()
    # 给每部电影初始化得分
    temp_db["得分"] = 0
    
    # 规则1：类型匹配得分
    temp_db["得分"] += temp_db["类型"].apply(lambda x: 30 if user_genre in str(x) else 0)
    
    # 规则2：演员匹配得分（用户输入演员才计算）
    if user_actor:
        temp_db["得分"] += temp_db["主演演员"].apply(lambda x: 25 if user_actor in str(x) else 0)
    
    # 规则3：相似电影匹配得分（用户输入喜欢的电影才计算）
    if user_liked_movies and user_liked_movies[0]:  # 检查是否有输入
        temp_db["得分"] += temp_db["相似电影"].apply(
            lambda x: 35 if any(liked in str(x) for liked in user_liked_movies) else 0
        )
    
    # 规则4：评分加成
    temp_db.loc[temp_db["豆瓣评分"] >= 9.0, "得分"] += 10
    temp_db.loc[(temp_db["豆瓣评分"] >= 8.5) & (temp_db["豆瓣评分"] < 9.0), "得分"] += 5
    temp_db.loc[(temp_db["豆瓣评分"] >= 8.0) & (temp_db["豆瓣评分"] < 8.5), "得分"] += 2
    
    # 筛选得分>0的电影，按得分降序排序，取前5部
    recommended = temp_db[temp_db["得分"] > 0].sort_values("得分", ascending=False).head(5)
    
    # 生成推荐结果文本
    if not recommended.empty:
        result = "=== 为你推荐的5部电影 ===\n"
        for idx, row in recommended.iterrows():
            result += f"{idx+1}. 《{row['电影名称']}》\n"
            result += f"   类型：{row['类型']}\n"
            result += f"   演员：{row['主演演员']}\n"
            result += f"   评分：{row['豆瓣评分']}\n"
            result += f"   推荐理由：{row['推荐备注']}\n\n"
        return result
    else:
        return "暂无匹配的电影，可尝试更换类型或演员~"

# 处理用户输入并显示推荐结果
def handle_recommendation():
    # 获取输入框内容
    genre = genre_entry.get().strip()
    actor = actor_entry.get().strip()
    liked = [m.strip() for m in liked_entry.get().split("，") if m.strip()]
    
    # 简单验证输入
    if not genre:
        messagebox.showwarning("提示", "请输入感兴趣的电影类型！")
        return
    
    # 获取推荐结果
    result = recommend_movies(genre, actor, liked)
    
    # 显示推荐结果
    messagebox.showinfo("推荐结果", result)

# 创建窗口
root = tk.Tk()
root.title("电影推荐系统")
root.geometry("450x350")

# 添加输入组件
tk.Label(root, text="感兴趣的类型（剧情/科幻/喜剧等）：").pack(pady=5)
genre_entry = tk.Entry(root, width=50)
genre_entry.pack(pady=5)

tk.Label(root, text="喜欢的演员（可选）：").pack(pady=5)
actor_entry = tk.Entry(root, width=50)
actor_entry.pack(pady=5)

tk.Label(root, text="喜欢的电影（用逗号分隔，可选）：").pack(pady=5)
liked_entry = tk.Entry(root, width=50)
liked_entry.pack(pady=5)

# 推荐按钮（点击后触发推荐逻辑）
tk.Button(root, text="获取推荐", command=handle_recommendation, width=20, height=2).pack(pady=20)

# 启动GUI事件循环
root.mainloop()
