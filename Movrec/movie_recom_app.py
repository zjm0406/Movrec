class MovieRecommendationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎬 电影推荐系统")
        self.root.geometry("800x600")
        
        # 添加豆瓣用户输入区域
        self.setup_douban_input()
        self.setup_ui()
        
    def setup_douban_input(self):
        # 豆瓣用户输入框架
        douban_frame = tk.Frame(self.root)
        douban_frame.pack(pady=10)
        
        tk.Label(douban_frame, text="豆瓣主页URL:", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        self.profile_entry = tk.Entry(douban_frame, width=30, font=("Arial", 12))
        self.profile_entry.pack(side=tk.LEFT, padx=5)
        
        tk.Label(douban_frame, text="Cookie(可选):", font=("Arial", 12)).pack(side=tk.LEFT, padx=(20,5))
        self.cookie_entry = tk.Entry(douban_frame, width=20, font=("Arial", 12))
        self.cookie_entry.pack(side=tk.LEFT, padx=5)
        
        import_btn = tk.Button(
            douban_frame,
            text="导入想看",
            command=self.import_wishlist,
            font=("Arial", 12, "bold"),
            bg='#2ecc71',
            fg='white',
            padx=10
        )
        import_btn.pack(side=tk.LEFT, padx=10)

    def import_wishlist(self):
        """导入用户的想看列表并进行推荐"""
        profile_url = self.profile_entry.get().strip()
        cookies = self.cookie_entry.get().strip() or None
        
        if not profile_url:
            messagebox.showwarning("输入错误", "请输入豆瓣主页URL")
            return
            
        try:
            self.status_var.set("正在获取想看列表...")
            self.root.update()
            
            # 获取想看列表
            wish_list = get_wish_list(profile_url, cookies)
            
            if not wish_list:
                messagebox.showwarning("提示", "未找到想看的电影")
                return
                
            # 更新状态并进行推荐
            self.status_var.set(f"找到 {len(wish_list)} 部想看的电影，正在推荐...")
            self.recommend_from_wishlist(wish_list)
            
        except Exception as e:
            messagebox.showerror("错误", f"获取想看列表失败: {str(e)}")
            self.status_var.set("获取想看列表失败")
            
    def recommend_from_wishlist(self, wish_list):
        """基于想看列表进行推荐"""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        
        try:
            topk = int(self.topk_var.get())
            result_df = recommend_from_collection(wish_list, topk=topk)
            
            if result_df is None:
                self.result_text.insert(tk.END, "❌ 未能基于想看列表生成推荐\n\n")
            else:
                self.display_recommendations("想看列表", result_df, topk)
                self.status_var.set(f"推荐完成 - 基于 {len(wish_list)} 部想看的电影")
                
        except Exception as e:
            self.result_text.insert(tk.END, f"❌ 推荐出错: {str(e)}")
            self.status_var.set("推荐失败")
            
        self.result_text.config(state=tk.DISABLED)