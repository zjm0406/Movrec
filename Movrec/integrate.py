# 整合crawl_douban_me.py和Doumini_V2.ipynb的功能

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 导入crawl_douban_me.py中的关键类和函数
from crawl_douban_me import (
    DoubanClient, 
    crawl_collect,
    parse_cookie_str
)

# 导入推荐系统相关函数!!在文件夹列没包含，要从doumini截取出来
from recommend_core import (
    recommend_from_collection,
    movies_new,
    similarity
)

# 导入之前定义的get_wish_list函数
from get_wish_list import get_wish_list

# GUI应用类
class IntegratedDoubanMini(MovieRecommendationApp):
    """整合了豆瓣爬虫和推荐功能的GUI应用"""
    
    def __init__(self, root):
        super().__init__(root)
        self.root.title("豆瓣Mini - 基于想看的智能推荐")
        
    def show_welcome_message(self):
        welcome_text = """✨ 欢迎使用豆瓣Mini推荐系统 ✨

两种使用方式：
1. 输入豆瓣主页URL，导入想看列表获取推荐
2. 手动输入电影名称获取推荐

首次使用建议：
• 先尝试输入豆瓣主页URL (形如 https://www.douban.com/people/your_id/)
• 点击「导入想看」，系统会自动抓取您想看的电影并据此推荐
• 如遇到访问限制，可以填入Cookie获取更好的体验

示例电影：
• 肖申克的救赎
• 霸王别姬
• 泰坦尼克号

提示：确保输入正确的URL或电影名称
"""
        self.result_text.insert(tk.END, welcome_text)
        self.result_text.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    app = IntegratedDoubanMini(root)
    root.mainloop()

if __name__ == "__main__":
    main()