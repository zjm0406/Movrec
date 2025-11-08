# recommend_engine/engine.py
import pandas as pd
import numpy as np
import jieba
import re
import os
import pickle # 用于缓存预处理数据和模型

# --- TensorFlow/Keras 导入 ---
# 确保环境中有正确的 TF 版本
import tensorflow as tf
from tensorflow import keras

# --- Sklearn 导入 ---
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 全局变量存储模型状态 ---
# 这些将在 initialize_engine 中被填充
movies_new = None
cv = None
encoder = None
feature = None
similarity = None

import os
import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import jieba
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics.pairwise import cosine_similarity


# --- 辅助函数：创建 CountVectorizer ---
def _get_stopwords():
    """返回中文停用词列表"""
    return [
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
        "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
        "你", "会", "着", "没有", "看", "好", "自己", "这", "那",
        "为", "之", "对", "与", "而", "并", "等", "被", "及", "或",
        "但", "所以", "如果", "因为", "然后", "而且", "那么", "他们",
        "我们", "你们", "它们", "什么", "哪个", "哪些", "哪里", "时候",
        "他", "她", "它", "咱们", "大家", "谁", "怎样", "怎么", "多少", "为什么",
        "这里", "那里", "这样", "那样", "这个", "那个", "这些", "那些",
        "地", "得", "所", "过", "吗", "呢", "吧", "啊", "呀", "嘛", "哇", "啦",
        "从", "自", "以", "向", "关于", "对于", "根据", "按照", "通过", "由于",
        "并且", "或者", "虽然", "即使", "尽管", "不管", "只要", "只有", "除非",
        "最", "太", "更", "非常", "十分", "特别", "极其", "比较", "稍微", "有点",
        "刚", "才", "正在", "已经", "曾经", "马上", "立刻", "永远", "一直", "总是",
        "常常", "经常", "往往", "不断", "偶尔", "又", "再", "还", "仅", "光",
        "能", "能够", "可以", "可能", "应该", "应当", "想", "愿意", "肯", "敢",
        "来", "去", "进", "出", "回", "起", "开",
        "些", "一些", "所有", "每个", "某个", "各种", "多个", "几个", "第一", "第二",
        "就是", "只是", "可是", "真是", "也是", "不是", "正是",
        "一样", "一般", "一点", "一起", "一直", "一下", "一种", "一次"
    ]


def _create_count_vectorizer():
    """创建并返回配置好的 CountVectorizer 实例"""
    stopwords = _get_stopwords()
    cv = CountVectorizer(
        max_features=10000,
        tokenizer=lambda text: jieba.lcut(str(text)),
        stop_words=stopwords,
        token_pattern=None
    )
    return cv


# --- 主初始化函数 ---
# 假设 movies_new, encoder, feature, similarity, _build_encoder_structure 是在模块级别定义的全局变量或函数
# from somewhere import movies_new, encoder, feature, similarity, _build_encoder_structure

def initialize_engine(data_folder_path, model_cache_path="model_cache.pkl"):
    """
    初始化推荐引擎：加载数据、预处理、训练DVAE模型（如果缓存不存在）。
    """
    # 声明需要修改的全局变量
    global movies_new, cv, encoder, feature, similarity
    # 注意：'encoder' 只需声明一次，如果之前已声明过，请删除重复的 global encoder

    cache_exists = os.path.exists(model_cache_path)
    if cache_exists:
        print("🔍 尝试从缓存加载预处理模型和特征...")
        try:
            with open(model_cache_path, 'rb') as f:
                cache = pickle.load(f)
                movies_new = cache['movies_new']
                # cv 不再从缓存加载，因为它包含不可 pickle 的 lambda
                # cv = cache['cv'] # <-- 删除此行

                feature = cache['feature']
                similarity = cache['similarity']
                
                # encoder 不序列化，需要重建结构再加载权重
                # 这里假设 _build_encoder_structure 已正确定义
                _build_encoder_structure(cache['inp_dim'], cache['code_dim']) 
                encoder.load_weights(os.path.join(os.path.dirname(model_cache_path), 'encoder_weights.h5'))
                
                # 无论是否从缓存加载，都需要重新创建 cv
                # 因为它不被缓存且包含 lambda
                cv = _create_count_vectorizer() # <-- 重新创建 CountVectorizer
                
                print("✅ 成功从缓存加载!")
                return # <-- 成功加载后直接返回，无需执行下面的初始化流程
        except Exception as e:
            print(f"⚠️ 缓存加载失败: {e}，将重新计算...")

    print("🔄 开始预处理数据和训练模型...")

    # 1. 读入原始数据
    movies_path = os.path.join(data_folder_path, "movies.csv")
    movies_db_path = os.path.join(data_folder_path, "movies_db.csv")
    director_label_path = os.path.join(data_folder_path, "director_label.csv")

    movies = pd.read_csv(movies_path)
    movies_db = pd.read_csv(movies_db_path)

    # 2. 清洗 movies_db，构造 INFO
    movies_db = movies_db.drop(columns=["durations", "votes"])
    movies_db["INFO"] = (
        movies_db["genres"].fillna("").astype(str) + " " +
        movies_db["countries"].fillna("").astype(str) + " " +
        movies_db["reviews"].fillna("").astype(str)
    )
    movies_db = movies_db.drop(columns=["genres", "countries", "reviews"])
    movies_db["title"] = movies_db["title"].apply(
        lambda x: "".join(re.findall(r"[\u4e00-\u9fff]+", str(x)))
    )

    # 3. 清洗 movies，本体只保留高分电影
    movies = movies.drop(
        columns=[
            "COVER", "IMDB_ID", "MINS", "OFFICIAL_SITE", "RELEASE_DATE",
            "SLUG", "ACTOR_IDS", "DIRECTOR_IDS", "LANGUAGES", "GENRES",
            "ALIAS", "ACTORS"
        ]
    )
    movies = movies[movies["DOUBAN_SCORE"] >= 6.5]

    # 4. 构造 movies_new（评分/人数过滤）
    movies_new_filtered = movies[movies["DOUBAN_VOTES"] >= 3000] \
        .sort_values(by=["DOUBAN_SCORE", "DOUBAN_VOTES"], ascending=[False, False])[
        ["DIRECTORS", "MOVIE_ID", "NAME", "DOUBAN_SCORE",
         "STORYLINE", "TAGS", "REGIONS", "YEAR"]
    ]

    # 5. 拼接剧情 + 标签 + 地区 作为 INFO
    movies_new_filtered["INFO"] = (
        movies_new_filtered["STORYLINE"].fillna("").astype(str) + " " +
        movies_new_filtered["TAGS"].fillna("").astype(str) + " " +
        movies_new_filtered["REGIONS"].fillna("").astype(str)
    )
    movies_new_filtered = movies_new_filtered.drop(columns=["STORYLINE", "TAGS", "REGIONS"])

    # 6. 拼接 movies_db（爬虫来的数据）
    movies_db_renamed = movies_db.rename(columns={
        "subject_id": "MOVIE_ID",
        "title": "NAME",
        "year": "YEAR",
        "rating": "DOUBAN_SCORE",
        "directors": "DIRECTORS",
    })
    movies_db_renamed = movies_db_renamed[
        ["DIRECTORS", "MOVIE_ID", "NAME", "DOUBAN_SCORE", "YEAR", "INFO"]
    ]

    movies_new_combined = pd.concat([movies_new_filtered, movies_db_renamed], ignore_index=True)

    # 7. 加导演标签
    director_label = pd.read_csv(director_label_path)
    director_to_label = dict(zip(director_label["DIRECTOR"], director_label["LABEL"]))
    movies_new_combined["LABEL"] = movies_new_combined["DIRECTORS"].apply(
        lambda x: ",".join(
            {
                director_to_label.get(d.strip())
                for d in str(x).split("/")
                if director_to_label.get(d.strip())
            }
        ) if pd.notna(x) else None
    )

    # 更新全局变量 movies_new
    movies_new = movies_new_combined
    print("✅ 数据清洗完成")

    # --- BOW + DVAE ---
    # 使用辅助函数创建 CountVectorizer
    cv = _create_count_vectorizer()

    vector = cv.fit_transform(movies_new["INFO"].astype(str)).toarray().astype("float32")
    print("✅ BOW 向量构建完成")

    # DVAE 参数
    inp_dim = vector.shape[1]
    code_dim = 64
    epochs = 5  # 调试阶段设小，生产可调大
    batch_size = 256
    beta_kl = 1.0

    # 编码器
    inputs = keras.Input(shape=(inp_dim,), name="bow_counts")
    x = keras.layers.GaussianNoise(0.15)(inputs)
    x = keras.layers.Dense(1000, activation="selu")(x)
    x = keras.layers.Dense(256, activation="selu")(x)
    z_mean = keras.layers.Dense(code_dim, name="z_mean")(x)
    z_logvar = keras.layers.Dense(code_dim, name="z_logvar")(x)

    def reparameterize(args):
        mu, logvar = args
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * logvar) * eps

    z = keras.layers.Lambda(reparameterize, name="z")([z_mean, z_logvar])
    encoder = keras.Model(inputs, [z_mean, z_logvar, z], name="dvae_encoder")

    # 解码器 (用于训练)
    latent_inputs = keras.Input(shape=(code_dim,), name="z_in")
    d = keras.layers.Dense(256, activation="selu")(latent_inputs)
    d = keras.layers.Dense(1000, activation="selu")(d)
    recons = keras.layers.Dense(inp_dim, activation=None, name="recon")(d)
    decoder = keras.Model(latent_inputs, recons, name="dvae_decoder")

    # KL 正则层
    class KLDivergenceLayer(keras.layers.Layer):
        def __init__(self, beta=1.0, scale=1.0, **kwargs):
            super().__init__(**kwargs)
            self.beta = beta
            self.scale = scale

        def call(self, inputs):
            mu, logvar = inputs
            kl_per_sample = -0.5 * tf.reduce_sum(
                1.0 + logvar - tf.exp(logvar) - tf.square(mu), axis=1
            )
            kl = tf.reduce_mean(kl_per_sample) / float(self.scale)
            self.add_loss(self.beta * kl)
            return tf.zeros_like(mu[:, :1])

    z_mean_out, z_logvar_out, z_out = encoder(inputs)
    _ = KLDivergenceLayer(beta=beta_kl, scale=inp_dim, name="kl_reg")(
        [z_mean_out, z_logvar_out]
    )
    recons_out = decoder(z_out)

    vae = keras.Model(inputs, recons_out, name="dvae")
    vae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

    # 训练 VAE
    history = vae.fit(
        vector, vector,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
    )
    print("✅ DVAE 模型训练完成")

    # 提取电影语义向量 feature（z_mean）
    z_mean_val = encoder.predict(vector, verbose=0)[0]
    feature = z_mean_val.astype("float32")
    print("✅ 电影语义特征提取完成")

    # 计算余弦相似度矩阵
    similarity = cosine_similarity(feature)
    print("✅ 相似度矩阵计算完成")

    # --- 缓存模型和特征 ---
    print("💾 正在缓存模型和特征...")
    # 注意：不再缓存 'cv' 对象，因为它包含了不可 pickle 的 lambda
    cache_to_save = {
        'movies_new': movies_new,     # DataFrame
        # 'cv': cv,                   # <-- 移除此行
        'feature': feature,           # NumPy array
        'similarity': similarity,     # NumPy array
        'inp_dim': inp_dim,           # int (用于重建 encoder 结构)
        'code_dim': code_dim          # int (用于重建 encoder 结构)
        # 如果需要缓存 director_to_label，也可以加上
        # 'director_to_label': director_to_label 
    }
    
    try:
        with open(model_cache_path, 'wb') as f:
            pickle.dump(cache_to_save, f)
        encoder.save_weights(os.path.join(os.path.dirname(model_cache_path), 'encoder.weights.h5'))
        print("✅ 缓存保存成功!")
    except Exception as e:
        print(f"⚠️ 缓存保存失败: {e}")
        # 根据你的需求决定是否要在这里抛出异常
        # raise e # 如果缓存失败是致命错误，取消注释此行

    # 注意：函数结束，cv 已在此函数作用域内创建并赋值给全局变量

def _build_encoder_structure(inp_dim, code_dim):
    """重建编码器结构以便加载权重"""
    global encoder
    inputs = keras.Input(shape=(inp_dim,), name="bow_counts")
    x = keras.layers.GaussianNoise(0.15)(inputs)
    x = keras.layers.Dense(1000, activation="selu")(x)
    x = keras.layers.Dense(256, activation="selu")(x)
    z_mean = keras.layers.Dense(code_dim, name="z_mean")(x)
    z_logvar = keras.layers.Dense(code_dim, name="z_logvar")(x)

    def reparameterize(args):
        mu, logvar = args
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * logvar) * eps

    z = keras.layers.Lambda(reparameterize, name="z")([z_mean, z_logvar])
    encoder = keras.Model(inputs, [z_mean, z_logvar, z], name="dvae_encoder")


def get_movie_features():
    """获取电影特征向量"""
    return feature


def get_movies_dataframe():
    """获取处理后的电影DataFrame"""
    return movies_new


def get_similarity_matrix():
    """获取电影相似度矩阵"""
    return similarity


def recommand(movie_name, sample_top=15, pick_n=5):
    """基础推荐函数（只用内容相似）"""
    label_idx = movies_new.index[movies_new["NAME"] == movie_name]
    if len(label_idx) == 0:
        # 尝试模糊匹配
        similar_movies = movies_new[movies_new["NAME"].str.contains(movie_name, na=False, case=False)]
        if len(similar_movies) > 0:
            print(f"未精确找到《{movie_name}》，尝试模糊匹配:")
            for idx, row in similar_movies.head(3).iterrows():
                 print(f"  - {row['NAME']}")
            # 默认使用第一个匹配项
            pos = similar_movies.index[0]
        else:
            print(f"未找到影片：《{movie_name}》")
            return None
    else:
        pos = movies_new.index.get_loc(label_idx[0])

    sims = similarity[pos]
    cand = np.argsort(-sims)  # 降序
    cand = cand[cand != pos]  # 去掉自身
    top_candidates = cand[:sample_top]

    n_pick = min(pick_n, len(top_candidates))
    if n_pick == 0:
        return pd.DataFrame()
    selected = np.random.choice(top_candidates, n_pick, replace=False)

    recs = []
    for j in selected:
        recs.append({
            "电影名": movies_new.iloc[j]["NAME"],
            "豆瓣评分": movies_new.iloc[j]["DOUBAN_SCORE"],
            "流派": movies_new.iloc[j]["LABEL"],
            "相似度": sims[j],
            "导演": movies_new.iloc[j]["DIRECTORS"],
        })
    df = pd.DataFrame(recs).sort_values(by="相似度", ascending=False).reset_index(drop=True)
    return df