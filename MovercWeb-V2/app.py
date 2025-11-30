# app.py
try:
    # 尝试从旧版本 werkzeug 导入 (向后兼容)
    from werkzeug.urls import url_parse
except ImportError:
    # 如果失败，则从 urllib.parse 导入 (适用于 Werkzeug >= 3.0)
    from urllib.parse import urlparse as url_parse # 保持别名 url_parse
from forms import LoginForm, RegistrationForm
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
try:
    # 尝试从旧版本 werkzeug 导入 (向后兼容)
    from werkzeug.urls import url_parse
except ImportError:
    # 如果失败，则从 urllib.parse 导入 (适用于 Werkzeug >= 3.0)
    from urllib.parse import urlparse as url_parse
from config import Config
from models import db, User, UserMoviePreference, UserMovieDislike, UserHybridWeights, RecommendationFeedback
from recommend_engine.engine import (
    initialize_engine,
    recommand,
    get_movies_dataframe,
    get_popular_movies,
    build_user_pref_vectors_from_user,
    enhanced_recommend_for_user,
    itemcf_recommend_for_movie,
    hybrid_recommend_for_user,
    get_hybrid_weights,
    set_hybrid_weights,
    get_engine_initialization_status,
)
import os
import pandas as pd
import threading

app = Flask(__name__)
app.config.from_object(Config)

# 初始化数据库
db.init_app(app)



# 初始化 Flask-WTF CSRF 保护（在测试模式下可禁用以便自动化测试）
from flask_wtf.csrf import CSRFProtect, generate_csrf as _generate_csrf
if not app.config.get('TESTING', False) and app.config.get('WTF_CSRF_ENABLED', True):
    csrf = CSRFProtect(app)
    def _gen_csrf():
        return _generate_csrf()
else:
    csrf = None
    def _gen_csrf():
        # 在测试模式或显式禁用 CSRF 时返回空字符串，模板调用仍然安全
        return ''

# 初始化登录管理器
login = LoginManager(app)
login.login_view = 'login'

@login.user_loader
def load_user(id):
    if id is None:
        return None
    try:
        # 推荐使用 SQLAlchemy 2.0 风格的 Session.get
        return db.session.get(User, int(id))
    except Exception as e:
        # 回退到旧 API 以保持兼容性（仅在 session.get 不可用或失败时）
        print(f"[WARN] db.session.get failed in load_user: {e}")
        try:
            return User.query.get(int(id))
        except Exception:
            return None

import os # 确保文件顶部已导入 os

# --- 新增：全局标志位，用于确保引擎和数据库表只初始化一次 ---
_engine_initialized = False
_engine_initializing = False

def start_engine_initialization():
    """在后台线程中启动推荐引擎初始化（非阻塞）。

    该函数会检查全局标志，防止重复启动。它使用 recommend_engine.engine.initialize_engine
    并在完成后设置 `_engine_initialized` 标志。
    """
    global _engine_initializing, _engine_initialized
    if _engine_initialized or _engine_initializing:
        return

    def _init():
        global _engine_initializing, _engine_initialized
        try:
            _engine_initializing = True
            print("🔧 后台线程：开始初始化推荐引擎...")
            from recommend_engine.engine import initialize_engine
            data_folder = app.config.get('DATA_FOLDER', os.path.join(app.root_path, 'data'))
            model_cache_path = os.path.join(app.root_path, 'model_cache.pkl')
            try:
                initialize_engine(data_folder, model_cache_path)
                print("🔧 后台线程：推荐引擎初始化完成")
            except Exception as e:
                print(f"[ERROR] 推荐引擎初始化失败: {e}")
        finally:
            _engine_initializing = False
            _engine_initialized = True

    t = threading.Thread(target=_init, daemon=True)
    t.start()

# 立即触发后台初始化（导入 app 后会自动开始）
start_engine_initialization()
@app.route('/engine_status')
def engine_status():
    """返回推荐引擎与模型缓存的诊断信息，便于调试和监控。"""
    import os
    info = {
        'engine_initialized': bool(_engine_initialized),
        'engine_initializing': bool(_engine_initializing),
    }

    # model cache
    cache_path = os.path.join(app.root_path, 'model_cache.pkl')
    info['model_cache_exists'] = os.path.exists(cache_path)
    if info['model_cache_exists']:
        try:
            st = os.stat(cache_path)
            info['model_cache_size'] = st.st_size
            info['model_cache_mtime'] = st.st_mtime
        except Exception:
            info['model_cache_size'] = None
            info['model_cache_mtime'] = None

        # 尝试安全打开 model_cache.pkl 并返回包含的 keys 与部分描述信息，便于诊断
        try:
            import pickle
            with open(cache_path, 'rb') as cf:
                try:
                    cache_obj = pickle.load(cf)
                    if isinstance(cache_obj, dict):
                        info['model_cache_keys'] = list(cache_obj.keys())
                        def _try_shape(x):
                            try:
                                return getattr(x, 'shape', None)
                            except Exception:
                                return None
                        info['model_cache_feature_shape'] = _try_shape(cache_obj.get('feature', None))
                        info['model_cache_movies_new_shape'] = _try_shape(cache_obj.get('movies_new', None))
                        info['model_cache_similarity_shape'] = _try_shape(cache_obj.get('similarity', None))
                    else:
                        info['model_cache_keys'] = None
                except Exception as e:
                    info['model_cache_load_error'] = str(e)
        except Exception as e:
            info['model_cache_read_error'] = str(e)

    # 尝试读取推荐引擎内部全局变量（如果已导入）
    try:
        from recommend_engine import engine as eng
        info['movies_new_shape'] = getattr(eng, 'movies_new', None) and getattr(eng.movies_new, 'shape', None)
        info['feature_shape'] = getattr(eng, 'feature', None) and getattr(eng.feature, 'shape', None)
        info['similarity_shape'] = getattr(eng, 'similarity', None) and getattr(eng.similarity, 'shape', None)
        info['G_shape'] = getattr(eng, 'G', None) and getattr(eng.G, 'shape', None)
        info['D_shape'] = getattr(eng, 'D', None) and getattr(eng.D, 'shape', None)
        info['engine_init_progress_percent'] = getattr(eng, 'init_progress_percent', None)
        msgs = getattr(eng, 'init_progress_messages', None)
        if msgs is not None:
            try:
                info['engine_init_messages'] = msgs[-30:]
            except Exception:
                info['engine_init_messages'] = None
        else:
            info['engine_init_messages'] = None
    except Exception as e:
        info['engine_error'] = str(e)

    return jsonify(info)

# 注入 csrf_token 到模板上下文
@app.context_processor
def inject_csrf_token():
    # 返回可调用对象供模板生成 CSRF token；在测试时返回空字符串以便测试客户端使用
    return dict(csrf_token=_gen_csrf)


@app.context_processor
def utility_processor():
    """提供模板辅助函数：poster_url(movie)

    优先查找本地静态目录 `static/posters/{MOVIE_ID}.*`，存在则返回静态 URL；
    否则回退到 movie 中的 `POSTER` / `COVER` 字段；最后使用占位图。
    """
    from flask import url_for
    def poster_url(movie):
        try:
            mid = str(movie.get('MOVIE_ID') or movie.get('MOVIE_ID') or movie.get('MOVIE_ID') or '')
        except Exception:
            mid = ''
        if mid:
            posters_dir = os.path.join(app.static_folder, 'posters')
            if os.path.isdir(posters_dir):
                for fname in os.listdir(posters_dir):
                    if fname.startswith(mid + '.') or fname.startswith(mid + '_') or fname == mid:
                        return url_for('static', filename=f'posters/{fname}')
        # 回退到已存在的远程链接字段
        for key in ('POSTER', 'COVER', 'IMAGE', 'IMAGE_URL'):
            v = movie.get(key) if isinstance(movie, dict) else getattr(movie, key, None)
            if v:
                return v
        # 最后使用占位图（SVG）
        return url_for('static', filename='placeholder_poster.svg')

    return dict(poster_url=poster_url)




# --- 移除或注释掉旧的初始化代码 ---
# with app.app_context():
#     print("🔧 初始化推荐引擎...")
#     initialize_engine(app.config['DATA_FOLDER'])
#     print("✅ 推荐引擎初始化完成!")

# --- 移除或注释掉旧的装饰器 ---
# @app.before_first_request
# def create_tables():
#     db.create_all()
@app.route('/')
@app.route('/index')
def index():
    # 获取热门电影：豆瓣评分 >= 6.5，评分人数 >= 3000，按评分和投票数降序取前 100 部
    print("\n--- DEBUG INDEX ROUTE ---")
    data_folder = app.config.get('DATA_FOLDER', os.path.join(app.root_path, 'data'))
    popular_movies_df = get_popular_movies(
        data_folder_path=data_folder,
        count=100,
        min_score=8.0,
        min_votes=50000
    )
    
    print(f"get_popular_movies() 返回类型: {type(popular_movies_df)}")
    if popular_movies_df is not None and not popular_movies_df.empty:
        print(f"获取热门电影数: {len(popular_movies_df)}")
        print(f"列名: {list(popular_movies_df.columns)}")
        print(f"首行示例:\n{popular_movies_df.iloc[0] if not popular_movies_df.empty else 'N/A'}")
        movies_list = popular_movies_df.to_dict('records')
    else:
        print("未能获取热门电影列表")
        movies_list = []
        flash("暂时无法加载热门电影列表。")
    
    print("--- DEBUG INDEX ROUTE END ---\n")
    
    # 传递给模板
    return render_template('index.html', movies=movies_list)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    form = RegistrationForm()
    # --- 新增调试代码：打印验证状态和错误 ---
    print("--- DEBUG REGISTER FORM ---")
    print(f"Form is submitted: {form.is_submitted()}")
    print(f"Form is valid: {form.validate()}") # 这会触发验证
    if form.errors:
        print("Form errors:", form.errors)
    print("--- DEBUG REGISTER FORM END ---")
    # --- 新增调试代码结束 ---
    
    if form.validate_on_submit(): # 这里面包含了 is_submitted() 和 validate()
        username = form.username.data
        email = form.email.data
        password = form.password.data

        print(f"--- DEBUG REGISTER START ---")
        print(f"Attempting to register user: {username}, email: {email}")

        user = User(username=username, email=email)
        user.set_password(password)
        print(f"Password hash generated: {user.password_hash}")

        db.session.add(user)
        try:
            db.session.commit()
            print(f"User {username} committed to database successfully.")
            inserted_user = User.query.filter_by(username=username).first()
            print(f"Re-queried user from DB: {inserted_user}, Hash: {inserted_user.password_hash if inserted_user else 'N/A'}")
            print(f"--- DEBUG REGISTER END ---")
            
            flash('恭喜你，注册成功！')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            print(f"--- DEBUG REGISTER ERROR ---")
            print(f"Error committing user to database: {e}")
            print(f"--- DEBUG REGISTER ERROR ---")
            flash('注册失败，请重试。')
    
    # 如果验证失败或 GET 请求，渲染表单
    return render_template('register.html', title='Register', form=form)


@app.route('/sync_douban', methods=['GET', 'POST'])
@login_required
def sync_douban():
    """页面：让用户输入自己的豆瓣 ID 与 Cookie（暂存在 session）。"""
    if request.method == 'POST':
        douban_id = (request.form.get('douban_id') or '').strip()
        douban_cookie = (request.form.get('douban_cookie') or '').strip()
        if not douban_id or not douban_cookie:
            flash('请同时填写豆瓣 ID 与 Cookie。')
            return render_template('sync_douban.html', douban_id=douban_id, cookie=douban_cookie)

        # 临时保存在 session 中
        session['douban_sync'] = {'douban_id': douban_id, 'cookie': douban_cookie}


        if request.method == 'POST':
            douban_id = (request.form.get('douban_id') or '').strip()
            douban_cookie = (request.form.get('douban_cookie') or '').strip()
            if not douban_id or not douban_cookie:
                flash('请同时填写豆瓣 ID 与 Cookie。')
                return render_template('sync_douban.html', douban_id=douban_id, cookie=douban_cookie)

            # 临时保存在 session 中
            session['douban_sync'] = {'douban_id': douban_id, 'cookie': douban_cookie}

            # --- 新版集成：先验证 Cookie ---
            try:
                from douban_sync import crawl_douban_movies, validate_cookie
                if not validate_cookie(douban_cookie):
                    flash('豆瓣 Cookie 无效或已过期，请重新获取。')
                    return render_template('sync_douban.html', douban_id=douban_id, cookie=douban_cookie)
                data = crawl_douban_movies(douban_id, douban_cookie)
            except Exception as e:
                flash(f'豆瓣爬取失败: {e}')
                return redirect(url_for('profile'))

            watched = data.get('watched', [])
            wish = data.get('wish', [])

            # 读取本地电影ID集合（只归档本地已存在的电影）
            import pandas as _pd
            data_folder = app.config.get('DATA_FOLDER', os.path.join(app.root_path, 'data'))
            movies_csv = os.path.join(data_folder, 'movies.csv')
            movies_db_csv = os.path.join(data_folder, 'movies_db.csv')
            local_ids = set()
            try:
                if os.path.exists(movies_csv):
                    df = _pd.read_csv(movies_csv, dtype=str)
                    for col in ['MOVIE_ID', 'douban_id', 'subject_id']:
                        if col in df.columns:
                            local_ids.update(df[col].dropna().astype(str).tolist())
            except Exception:
                pass
            try:
                if os.path.exists(movies_db_csv):
                    df = _pd.read_csv(movies_db_csv, dtype=str)
                    for col in ['MOVIE_ID', 'douban_id', 'subject_id']:
                        if col in df.columns:
                            local_ids.update(df[col].dropna().astype(str).tolist())
            except Exception:
                pass
            print(f'[豆瓣同步] 本地电影ID集合共{len(local_ids)}个，示例: {list(local_ids)[:5]}')

            # 调试输出：未能匹配到本地的豆瓣ID
            unmatched_watched = [mid for mid in [str(m.get('douban_id')) for m in watched if m.get('douban_id')] if mid not in local_ids]
            unmatched_wish = [mid for mid in [str(m.get('douban_id')) for m in wish if m.get('douban_id')] if mid not in local_ids]
            print(f'[豆瓣同步] “看过”未匹配到本地的ID共{len(unmatched_watched)}个，示例: {unmatched_watched[:5]}')
            print(f'[豆瓣同步] “想看”未匹配到本地的ID共{len(unmatched_wish)}个，示例: {unmatched_wish[:5]}')

            # 归档“看过”到 UserMoviePreference，“想看”到 UserMovieDislike
            from models import UserMoviePreference, UserMovieDislike, db
            user_id = current_user.id
            count_watched, count_wish = 0, 0
            for m in watched:
                mid = str(m.get('douban_id'))
                if mid and mid in local_ids:
                    # 避免重复
                    exists = UserMoviePreference.query.filter_by(user_id=user_id, movie_douban_id=mid).first()
                    if not exists:
                        db.session.add(UserMoviePreference(user_id=user_id, movie_douban_id=mid))
                        count_watched += 1
            for m in wish:
                mid = str(m.get('douban_id'))
                if mid and mid in local_ids:
                    exists = UserMovieDislike.query.filter_by(user_id=user_id, movie_douban_id=mid).first()
                    if not exists:
                        db.session.add(UserMovieDislike(user_id=user_id, movie_douban_id=mid))
                        count_wish += 1
            try:
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                flash(f'同步归档失败: {e}')
                return redirect(url_for('profile'))

            flash(f'豆瓣同步完成！已归档“看过”{count_watched}部、“想看”{count_wish}部电影。')
            return redirect(url_for('profile'))
    # 优先从数据库读取豆瓣ID和Cookie
    from models import User
    user = User.query.get(current_user.id)
    douban_id = user.douban_id if user and user.douban_id else ''
    douban_cookie = user.douban_cookie if user and user.douban_cookie else ''
    # session 仅用于页面回显（如刚提交）
    data = session.get('douban_sync', {})
    douban_id = data.get('douban_id', douban_id)
    douban_cookie = data.get('cookie', douban_cookie)
    return render_template('sync_douban.html', douban_id=douban_id, cookie=douban_cookie)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        remember_me = form.remember_me.data

        # --- 调试信息 1: 登录尝试 ---
        print(f"--- DEBUG LOGIN ATTEMPT ---")
        print(f"Login attempt for username: '{username}'")

        user = User.query.filter_by(username=username).first()
        
        # --- 调试信息 2: 查询结果 ---
        print(f"User found in DB: {user}")
        if user:
            print(f"Stored password hash: {user.password_hash}")
            password_check_result = user.check_password(password)
            print(f"Password check result: {password_check_result}")
        else:
            print("No user found with that username.")
        print(f"--- DEBUG LOGIN ATTEMPT END ---")

        if user is None or not user.check_password(password):
            flash('无效的用户名或密码')
            return redirect(url_for('login'))
        login_user(user, remember=remember_me)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
    
    return render_template('login.html', title='Sign In', form=form)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    # 获取当前用户的喜好和厌恶列表（保持原始 id 类型）
    liked_ids = [pref.movie_douban_id for pref in current_user.liked_movies.all()]
    disliked_ids = [dis.movie_douban_id for dis in current_user.disliked_movies.all()]

    # resolution helper: 优先使用内存 movies_new，再回退到 CSV（movies.csv / movies_db.csv）查找信息
    def _resolve_movie(mid, movies_df_cache=None, movies_db_cache=None):
        # 返回 dict: NAME, DOUBAN_SCORE, MOVIE_ID, DIRECTORS, YEAR
        if mid is None:
            return {'MOVIE_ID': None, 'NAME': '未知影片', 'DOUBAN_SCORE': None}
        mid_s = str(mid)
        # 1) 尝试全局 movies_new
        mdf = get_movies_dataframe()
        if mdf is not None and not mdf.empty and 'MOVIE_ID' in mdf.columns:
            try:
                row = mdf[mdf['MOVIE_ID'].astype(str) == mid_s]
                if not row.empty:
                    r = row.iloc[0].to_dict()
                    return {
                        'MOVIE_ID': r.get('MOVIE_ID'),
                        'NAME': r.get('NAME'),
                        'DOUBAN_SCORE': r.get('DOUBAN_SCORE'),
                        'DIRECTORS': r.get('DIRECTORS'),
                        'YEAR': r.get('YEAR'),
                    }
            except Exception:
                pass

        # 2) 回退到 CSV（只在需要时加载以减少开销）
        import pandas as _pd
        data_folder = app.config.get('DATA_FOLDER', os.path.join(app.root_path, 'data'))
        movies_csv = os.path.join(data_folder, 'movies.csv')
        movies_db_csv = os.path.join(data_folder, 'movies_db.csv')

        # 尝试 movies.csv
        try:
            if movies_df_cache is None:
                if os.path.exists(movies_csv):
                    movies_df_cache = _pd.read_csv(movies_csv, dtype=str)
            if movies_df_cache is not None and not movies_df_cache.empty:
                for cid in ['MOVIE_ID', 'douban_id', 'subject_id']:
                    if cid in movies_df_cache.columns:
                        match = movies_df_cache[movies_df_cache[cid].astype(str) == mid_s]
                        if not match.empty:
                            r = match.iloc[0].to_dict()
                            name_col = 'NAME' if 'NAME' in r else ( 'title' if 'title' in r else None)
                            return {
                                'MOVIE_ID': r.get(cid),
                                'NAME': r.get(name_col, f'影片 {mid_s}') if name_col else f'影片 {mid_s}',
                                'DOUBAN_SCORE': r.get('DOUBAN_SCORE'),
                                'DIRECTORS': r.get('DIRECTORS'),
                                'YEAR': r.get('YEAR'),
                            }
        except Exception:
            pass

        # 尝试 movies_db.csv
        try:
            if movies_db_cache is None:
                if os.path.exists(movies_db_csv):
                    movies_db_cache = _pd.read_csv(movies_db_csv, dtype=str)
            if movies_db_cache is not None and not movies_db_cache.empty:
                for cid in ['subject_id', 'MOVIE_ID', 'douban_id']:
                    if cid in movies_db_cache.columns:
                        match = movies_db_cache[movies_db_cache[cid].astype(str) == mid_s]
                        if not match.empty:
                            r = match.iloc[0].to_dict()
                            name_col = 'NAME' if 'NAME' in r else ( 'title' if 'title' in r else None)
                            return {
                                'MOVIE_ID': r.get(cid),
                                'NAME': r.get(name_col, f'影片 {mid_s}') if name_col else f'影片 {mid_s}',
                                'DOUBAN_SCORE': r.get('DOUBAN_SCORE'),
                                'DIRECTORS': r.get('DIRECTORS'),
                                'YEAR': r.get('YEAR'),
                            }
        except Exception:
            pass

        # 兜底
        return {'MOVIE_ID': mid_s, 'NAME': f'未知影片 ({mid_s})', 'DOUBAN_SCORE': None}

    # 预加载 csv cache 以避免对每个 id 重复读取文件（只读一次，重用）
    import pandas as _pd
    data_folder = app.config.get('DATA_FOLDER', os.path.join(app.root_path, 'data'))
    movies_csv = os.path.join(data_folder, 'movies.csv')
    movies_db_csv = os.path.join(data_folder, 'movies_db.csv')

    movies_df_cache = None
    movies_db_cache = None
    try:
        if os.path.exists(movies_csv):
            movies_df_cache = _pd.read_csv(movies_csv, dtype=str)
    except Exception:
        movies_df_cache = None
    try:
        if os.path.exists(movies_db_csv):
            movies_db_cache = _pd.read_csv(movies_db_csv, dtype=str)
    except Exception:
        movies_db_cache = None

    liked_movies_info = [_resolve_movie(mid, movies_df_cache, movies_db_cache) for mid in liked_ids]
    disliked_movies_info = [_resolve_movie(mid, movies_df_cache, movies_db_cache) for mid in disliked_ids]

    # 轻量调试输出，便于排查为什么列表为空
    print(f"[DEBUG] profile: user_id={current_user.id} liked_count_db={len(liked_ids)} disliked_count_db={len(disliked_ids)} returned_liked={len(liked_movies_info)} returned_disliked={len(disliked_movies_info)}")

    return render_template('profile.html', title='Profile',
                           liked_movies=liked_movies_info,
                           disliked_movies=disliked_movies_info)


# 注意：由于你的电影数据主要来自 CSV，这个路由需要能访问到该数据。
# 假设 get_movies_dataframe() 返回包含所有电影信息的 DataFrame

@app.route('/liked_movies')
@login_required
def liked_movies():
    """分页显示当前用户喜欢的电影，每页 25 部"""
    page = request.args.get('page', 1, type=int)
    per_page = 25
    
    # 从 DB 获取用户喜欢的电影 ID
    liked_ids = [pref.movie_douban_id for pref in current_user.liked_movies.all()]
    
    # 计算分页
    total_count = len(liked_ids)
    total_pages = (total_count + per_page - 1) // per_page if total_count > 0 else 1
    if page < 1 or page > total_pages:
        page = 1
    
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_ids = liked_ids[start_idx:end_idx]
    
    # 解析电影信息
    import pandas as _pd
    data_folder = app.config.get('DATA_FOLDER', os.path.join(app.root_path, 'data'))
    movies_csv = os.path.join(data_folder, 'movies.csv')
    movies_db_csv = os.path.join(data_folder, 'movies_db.csv')
    
    movies_df_cache = None
    movies_db_cache = None
    try:
        if os.path.exists(movies_csv):
            movies_df_cache = _pd.read_csv(movies_csv, dtype=str)
    except Exception:
        pass
    try:
        if os.path.exists(movies_db_csv):
            movies_db_cache = _pd.read_csv(movies_db_csv, dtype=str)
    except Exception:
        pass
    
    def resolve_movie(mid, mc_df=None, mc_db=None):
        import pandas as pd_nan
        import math
        
        def clean_value(v):
            """将 NaN、None 和其他不可序列化的值转换为 None"""
            if v is None:
                return None
            if isinstance(v, float) and math.isnan(v):
                return None
            if pd_nan.isna(v):
                return None
            return v
        
        if mid is None:
            return {'MOVIE_ID': None, 'NAME': '未知影片', 'DOUBAN_SCORE': None, 'DIRECTORS': None, 'YEAR': None, 'COVER': None}
        mid_s = str(mid)
        if mc_df is not None and not mc_df.empty:
            for cid in ['MOVIE_ID', 'douban_id']:
                if cid in mc_df.columns:
                    m = mc_df[mc_df[cid].astype(str) == mid_s]
                    if not m.empty:
                        r = m.iloc[0].to_dict()
                        return {
                            'MOVIE_ID': clean_value(r.get('MOVIE_ID')),
                            'NAME': clean_value(r.get('NAME')) or f'影片 {mid_s}',
                            'DOUBAN_SCORE': clean_value(r.get('DOUBAN_SCORE')),
                            'DIRECTORS': clean_value(r.get('DIRECTORS')),
                            'YEAR': clean_value(r.get('YEAR')),
                            'COVER': clean_value(r.get('COVER'))
                        }
        if mc_db is not None and not mc_db.empty:
            for cid in ['subject_id', 'MOVIE_ID']:
                if cid in mc_db.columns:
                    m = mc_db[mc_db[cid].astype(str) == mid_s]
                    if not m.empty:
                        r = m.iloc[0].to_dict()
                        return {
                            'MOVIE_ID': clean_value(r.get(cid)),
                            'NAME': clean_value(r.get('title', r.get('NAME'))) or f'影片 {mid_s}',
                            'DOUBAN_SCORE': clean_value(r.get('rating', r.get('DOUBAN_SCORE'))),
                            'DIRECTORS': clean_value(r.get('directors')),
                            'YEAR': clean_value(r.get('year')),
                            'COVER': None
                        }
        return {'MOVIE_ID': mid_s, 'NAME': f'未知影片({mid_s})', 'DOUBAN_SCORE': None, 'DIRECTORS': None, 'YEAR': None, 'COVER': None}
    
    movies_info = [resolve_movie(mid, movies_df_cache, movies_db_cache) for mid in page_ids]
    return render_template('liked_movies.html', movies=movies_info, page=page, total_pages=total_pages, total_count=total_count)

@app.route('/disliked_movies')
@login_required
def disliked_movies():
    """分页显示当前用户不喜欢的电影，每页 25 部"""
    page = request.args.get('page', 1, type=int)
    per_page = 25
    
    # 从 DB 获取用户不喜欢的电影 ID
    disliked_ids = [dis.movie_douban_id for dis in current_user.disliked_movies.all()]
    
    # 计算分页
    total_count = len(disliked_ids)
    total_pages = (total_count + per_page - 1) // per_page if total_count > 0 else 1
    if page < 1 or page > total_pages:
        page = 1
    
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_ids = disliked_ids[start_idx:end_idx]
    
    # 解析电影信息
    import pandas as _pd
    data_folder = app.config.get('DATA_FOLDER', os.path.join(app.root_path, 'data'))
    movies_csv = os.path.join(data_folder, 'movies.csv')
    movies_db_csv = os.path.join(data_folder, 'movies_db.csv')
    
    movies_df_cache = None
    movies_db_cache = None
    try:
        if os.path.exists(movies_csv):
            movies_df_cache = _pd.read_csv(movies_csv, dtype=str)
    except Exception:
        pass
    try:
        if os.path.exists(movies_db_csv):
            movies_db_cache = _pd.read_csv(movies_db_csv, dtype=str)
    except Exception:
        pass
    
    def resolve_movie(mid, mc_df=None, mc_db=None):
        import pandas as pd_nan
        import math
        
        def clean_value(v):
            """将 NaN、None 和其他不可序列化的值转换为 None"""
            if v is None:
                return None
            if isinstance(v, float) and math.isnan(v):
                return None
            if pd_nan.isna(v):
                return None
            return v
        
        if mid is None:
            return {'MOVIE_ID': None, 'NAME': '未知影片', 'DOUBAN_SCORE': None, 'DIRECTORS': None, 'YEAR': None, 'COVER': None}
        mid_s = str(mid)
        if mc_df is not None and not mc_df.empty:
            for cid in ['MOVIE_ID', 'douban_id']:
                if cid in mc_df.columns:
                    m = mc_df[mc_df[cid].astype(str) == mid_s]
                    if not m.empty:
                        r = m.iloc[0].to_dict()
                        return {
                            'MOVIE_ID': clean_value(r.get('MOVIE_ID')),
                            'NAME': clean_value(r.get('NAME')) or f'影片 {mid_s}',
                            'DOUBAN_SCORE': clean_value(r.get('DOUBAN_SCORE')),
                            'DIRECTORS': clean_value(r.get('DIRECTORS')),
                            'YEAR': clean_value(r.get('YEAR')),
                            'COVER': clean_value(r.get('COVER'))
                        }
        if mc_db is not None and not mc_db.empty:
            for cid in ['subject_id', 'MOVIE_ID']:
                if cid in mc_db.columns:
                    m = mc_db[mc_db[cid].astype(str) == mid_s]
                    if not m.empty:
                        r = m.iloc[0].to_dict()
                        return {
                            'MOVIE_ID': clean_value(r.get(cid)),
                            'NAME': clean_value(r.get('title', r.get('NAME'))) or f'影片 {mid_s}',
                            'DOUBAN_SCORE': clean_value(r.get('rating', r.get('DOUBAN_SCORE'))),
                            'DIRECTORS': clean_value(r.get('directors')),
                            'YEAR': clean_value(r.get('year')),
                            'COVER': None
                        }
        return {'MOVIE_ID': mid_s, 'NAME': f'未知影片({mid_s})', 'DOUBAN_SCORE': None, 'DIRECTORS': None, 'YEAR': None, 'COVER': None}
    
    movies_info = [resolve_movie(mid, movies_df_cache, movies_db_cache) for mid in page_ids]
    return render_template('disliked_movies.html', movies=movies_info, page=page, total_pages=total_pages, total_count=total_count)

@app.route('/movie/<string:movie_douban_id>') # 使用 douban_id 作为 URL 参数
def movie_detail(movie_douban_id):
    # 从全局 DataFrame 获取电影信息
    movies_df = get_movies_dataframe()
    if movies_df is None or movies_df.empty:
         flash('电影数据未加载。')
         return redirect(url_for('index'))

    # 筛选特定电影
    movie_row = movies_df[movies_df['MOVIE_ID'] == movie_douban_id]
    if movie_row.empty:
        flash('未找到指定的电影。')
        return redirect(url_for('index'))

    # 将 Series 转换为字典以便在模板中使用
    movie_info = movie_row.iloc[0].to_dict()

    # 检查当前用户偏好状态 (需要在 app context 内)
    user_has_liked = False
    user_has_disliked = False
    if current_user.is_authenticated:
        # 查询关联表
        liked_entry = UserMoviePreference.query.filter_by(user_id=current_user.id, movie_douban_id=movie_douban_id).first()
        disliked_entry = UserMovieDislike.query.filter_by(user_id=current_user.id, movie_douban_id=movie_douban_id).first()
        user_has_liked = liked_entry is not None
        user_has_disliked = disliked_entry is not None

    return render_template('movie_detail_douban.html', movie=movie_info,
                           user_has_liked=user_has_liked,
                           user_has_disliked=user_has_disliked)

# --- 新增/修改：优化后的 toggle_preference API 路由 ---
# 使用 session 批量操作以提高效率并保证原子性
@app.route('/api/toggle_preference_optimized', methods=['POST'])
@login_required
def toggle_preference_optimized():
    """
    优化版本的偏好切换API，使用数据库事务确保一致性，
    并返回更新后的按钮状态给前端。
    """
    data = request.get_json()
    movie_douban_id = data.get('movie_douban_id')
    action = data.get('action') # 'like' or 'dislike'

    if not movie_douban_id or action not in ['like', 'dislike']:
        return jsonify({'error': 'Invalid data'}), 400

    try:
        # 使用显式的 commit/rollback，避免在已有事务中再次 begin 导致错误
        if action == 'like':
            UserMovieDislike.query.filter_by(
                user_id=current_user.id, movie_douban_id=movie_douban_id
            ).delete(synchronize_session=False)
            existing_like = UserMoviePreference.query.filter_by(
                user_id=current_user.id, movie_douban_id=movie_douban_id
            ).first()
            if not existing_like:
                new_pref = UserMoviePreference(user_id=current_user.id, movie_douban_id=movie_douban_id)
                db.session.add(new_pref)
                new_status = 'liked'
            else:
                db.session.delete(existing_like)
                new_status = 'none'
        else:  # dislike
            UserMoviePreference.query.filter_by(
                user_id=current_user.id, movie_douban_id=movie_douban_id
            ).delete(synchronize_session=False)
            existing_dislike = UserMovieDislike.query.filter_by(
                user_id=current_user.id, movie_douban_id=movie_douban_id
            ).first()
            if not existing_dislike:
                new_dislike = UserMovieDislike(user_id=current_user.id, movie_douban_id=movie_douban_id)
                db.session.add(new_dislike)
                new_status = 'disliked'
            else:
                db.session.delete(existing_dislike)
                new_status = 'none'

        db.session.commit()
        return jsonify({'success': True, 'new_status': new_status})

    except Exception as e:
        db.session.rollback()
        print(f"[错误] 切换偏好失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': '操作失败: ' + str(e)}), 500

@app.route('/api/toggle_preference', methods=['POST'])
@login_required
def toggle_preference():
    data = request.get_json()
    movie_douban_id = data.get('movie_douban_id')
    action = data.get('action') # 'like' or 'dislike'

    if not movie_douban_id or action not in ['like', 'dislike']:
        return jsonify({'error': 'Invalid data'}), 400

    # 查找或创建 Movie 实体（如果数据库中没有）
    # 注意：这里为了简化，我们直接操作关联表，不强制要求 Movie 表存在
    # 如果未来 Movie 表完善，这里需要先查询/创建 Movie

    # 先删除相反的操作
    if action == 'like':
        UserMovieDislike.query.filter_by(user_id=current_user.id, movie_douban_id=movie_douban_id).delete()
        # 检查是否已存在
        existing = UserMoviePreference.query.filter_by(user_id=current_user.id, movie_douban_id=movie_douban_id).first()
        if not existing:
            new_pref = UserMoviePreference(user_id=current_user.id, movie_douban_id=movie_douban_id)
            db.session.add(new_pref)
    else: # dislike
        UserMoviePreference.query.filter_by(user_id=current_user.id, movie_douban_id=movie_douban_id).delete()
        existing = UserMovieDislike.query.filter_by(user_id=current_user.id, movie_douban_id=movie_douban_id).first()
        if not existing:
            new_dislike = UserMovieDislike(user_id=current_user.id, movie_douban_id=movie_douban_id)
            db.session.add(new_dislike)

    db.session.commit()
    return jsonify({'success': True})


# --- 混合推荐权重管理 API ---
@app.route('/api/hybrid_weights', methods=['GET'])
@login_required
def get_user_weights():
    """获取当前用户的混合推荐权重配置"""
    user_weights = UserHybridWeights.query.filter_by(user_id=current_user.id).first()
    
    if user_weights:
        return jsonify({
            'success': True,
            'dvae_weight': user_weights.dvae_weight,
            'itemcf_weight': user_weights.itemcf_weight
        })
    else:
        # 返回全局默认值
        global_weights = get_hybrid_weights()
        return jsonify({
            'success': True,
            'dvae_weight': global_weights.get('dvae', 0.6),
            'itemcf_weight': global_weights.get('itemcf', 0.4)
        })


@app.route('/api/hybrid_weights', methods=['POST'])
@login_required
def set_user_weights():
    """设置并保存用户的混合推荐权重配置
    
    请求 JSON:
    {
        "dvae_weight": 0.5,
        "itemcf_weight": 0.5
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    dvae_w = data.get('dvae_weight')
    itemcf_w = data.get('itemcf_weight')

    # 支持仅传入 dvae_weight，itemcf_weight 将由 1 - dvae 计算
    if dvae_w is None and itemcf_w is None:
        return jsonify({'error': 'dvae_weight or itemcf_weight is required'}), 400

    try:
        if dvae_w is not None:
            dvae_w = float(dvae_w)
            itemcf_w = 1.0 - dvae_w
        else:
            itemcf_w = float(itemcf_w)
            dvae_w = 1.0 - itemcf_w
    except (ValueError, TypeError):
        return jsonify({'error': 'Weights must be numeric'}), 400
    
    # 检查范围
    if not (0 <= dvae_w <= 1) or not (0 <= itemcf_w <= 1):
        return jsonify({'error': 'Weights must be between 0 and 1'}), 400
    
    # 归一化
    total = dvae_w + itemcf_w
    if total == 0:
        dvae_w, itemcf_w = 0.6, 0.4
    else:
        dvae_w = dvae_w / total
        itemcf_w = itemcf_w / total
    
    try:
        # 查找或创建用户权重记录
        user_weights = UserHybridWeights.query.filter_by(user_id=current_user.id).first()
        if user_weights:
            user_weights.dvae_weight = dvae_w
            user_weights.itemcf_weight = itemcf_w
        else:
            user_weights = UserHybridWeights(
                user_id=current_user.id,
                dvae_weight=dvae_w,
                itemcf_weight=itemcf_w
            )
            db.session.add(user_weights)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'dvae_weight': dvae_w,
            'itemcf_weight': itemcf_w,
            'message': '权重配置已保存'
        })
    except Exception as e:
        db.session.rollback()
        print(f"[错误] 保存权重失败: {e}")
        import traceback
        traceback.print_exc()
        # 返回简洁的错误信息，不带重复的 '保存失败:' 前缀，前端会统一展示带前缀的提示
        return jsonify({'error': str(e)}), 500

# --- 推荐反馈 API（用于改进 itemCF） ---
@app.route('/api/recommend_feedback', methods=['POST'])
@login_required
def submit_recommendation_feedback():
    """提交对推荐结果的反馈
    
    请求 JSON:
    {
        "query_movie_id": "123456",
        "recommended_movie_id": "654321",
        "feedback": "helpful" | "not_helpful" | "dislike",
        "recommendation_method": "hybrid"
    }
    
    反馈类型：
    - helpful: 推荐很有帮助
    - not_helpful: 推荐没帮助但还可以
    - dislike: 推荐不相关或质量差
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    query_mid = data.get('query_movie_id')
    rec_mid = data.get('recommended_movie_id')
    feedback = data.get('feedback', 'not_helpful')
    method = data.get('recommendation_method', 'hybrid')
    
    if not query_mid or not rec_mid:
        return jsonify({'error': 'query_movie_id and recommended_movie_id are required'}), 400
    
    if feedback not in ['helpful', 'not_helpful', 'dislike']:
        return jsonify({'error': 'Invalid feedback type'}), 400
    
    try:
        # 保存反馈记录
        feedback_record = RecommendationFeedback(
            user_id=current_user.id,
            query_movie_id=str(query_mid),
            recommended_movie_id=str(rec_mid),
            feedback=feedback,
            recommendation_method=method
        )
        db.session.add(feedback_record)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': '感谢你的反馈！这将帮助我们改进推荐算法',
            'feedback_id': feedback_record.id
        })
    except Exception as e:
        db.session.rollback()
        print(f"[错误] 提交反馈失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': '提交失败: ' + str(e)}), 500


@app.route('/api/recommend_feedback_undo', methods=['POST'])
@login_required
def undo_recommendation_feedback():
    """撤销最近一次对某条推荐的反馈（按 user + recommended_movie_id [+ query_movie_id] 匹配）。

    请求 JSON:
    {
        "recommended_movie_id": "<id>",
        "query_movie_id": "<optional>"
    }
    """
    data = request.get_json() or {}
    rec_mid = data.get('recommended_movie_id')
    query_mid = data.get('query_movie_id')

    if not rec_mid:
        return jsonify({'error': 'recommended_movie_id is required'}), 400

    try:
        # 查找最后一条匹配的反馈记录并删除
        q = RecommendationFeedback.query.filter_by(user_id=current_user.id, recommended_movie_id=str(rec_mid))
        if query_mid:
            q = q.filter_by(query_movie_id=str(query_mid))
        record = q.order_by(RecommendationFeedback.id.desc()).first()
        if not record:
            return jsonify({'success': False, 'message': '未找到可撤销的反馈'}), 404

        db.session.delete(record)
        db.session.commit()
        return jsonify({'success': True, 'message': '反馈已撤销'})
    except Exception as e:
        db.session.rollback()
        print(f"[错误] 撤销反馈失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': '撤销失败: ' + str(e)}), 500


@app.route('/api/itemcf_feedback_stats')
def get_itemcf_feedback_stats():
    """获取 itemCF 反馈统计（用于模型改进监控）
    
    返回：
    {
        "total_feedback": 100,
        "helpful": 60,
        "not_helpful": 30,
        "dislike": 10,
        "helpful_rate": 0.60
    }
    """
    from sqlalchemy import func
    
    # 统计所有反馈
    total = db.session.query(func.count(RecommendationFeedback.id)).scalar() or 0
    helpful = db.session.query(func.count(RecommendationFeedback.id)).filter(
        RecommendationFeedback.feedback == 'helpful'
    ).scalar() or 0
    not_helpful = db.session.query(func.count(RecommendationFeedback.id)).filter(
        RecommendationFeedback.feedback == 'not_helpful'
    ).scalar() or 0
    dislike = db.session.query(func.count(RecommendationFeedback.id)).filter(
        RecommendationFeedback.feedback == 'dislike'
    ).scalar() or 0
    
    helpful_rate = helpful / total if total > 0 else 0
    
    return jsonify({
        'success': True,
        'total_feedback': total,
        'helpful': helpful,
        'not_helpful': not_helpful,
        'dislike': dislike,
        'helpful_rate': round(helpful_rate, 3),
        'note': '这些数据可用于评估和改进 itemCF 算法的质量'
    })


@app.route('/recommend', methods=['GET', 'POST'])
@login_required
def recommend():
    import time
    from recommend_engine import engine as engine_module
    import difflib
    recommendations = None
    query = ""
    engine_status = "ready"
    search_type = request.form.get('search_type', 'movie_name') if request.method == 'POST' else 'movie_name'
    guess_query = None

    if request.method == 'POST':
        query = request.form.get('movie_query', '').strip()
        if query:
            wait_count = 0
            while not _engine_initialized and wait_count < 10:
                time.sleep(0.5)
                wait_count += 1
            if not _engine_initialized:
                engine_status = "initializing"
                recommendations = pd.DataFrame()
            else:
                from models import UserHybridWeights
                user_weights_record = UserHybridWeights.query.filter_by(user_id=current_user.id).first()
                if user_weights_record:
                    user_weights = {
                        'dvae': user_weights_record.dvae_weight,
                        'itemcf': user_weights_record.itemcf_weight
                    }
                else:
                    user_weights = get_hybrid_weights()

                movies_df = get_movies_dataframe()
                if movies_df is None or movies_df.empty:
                    recommendations = pd.DataFrame()
                else:
                    # 搜索类型分流
                    if search_type == 'movie_name':
                        # 电影名称精确/模糊匹配
                        title_cols = [c for c in ['NAME', 'title', 'name'] if c in movies_df.columns]
                        # 精确匹配
                        mask = None
                        for col in title_cols:
                            m = movies_df[col].astype(str) == query
                            mask = m if mask is None else (mask | m)
                        matched = movies_df[mask] if mask is not None else pd.DataFrame()
                        if matched.empty:
                            # 未找到，做相似匹配
                            all_titles = []
                            for col in title_cols:
                                all_titles += list(movies_df[col].dropna().astype(str).unique())
                            # 用 difflib 获取最相似的名称
                            best_match = difflib.get_close_matches(query, all_titles, n=1, cutoff=0.6)
                            if best_match:
                                guess_query = best_match[0]
                                # 用最相似名称做推荐
                                try:
                                    uid = current_user.id if current_user.is_authenticated else None
                                    recommendations = hybrid_recommend_for_user(guess_query, user_id=uid, weights=user_weights, sample_top=20, pick_n=15)
                                except Exception as e:
                                    recommendations = None
                            else:
                                recommendations = pd.DataFrame()
                        else:
                            # 精确命中，直接推荐
                            try:
                                uid = current_user.id if current_user.is_authenticated else None
                                recommendations = hybrid_recommend_for_user(query, user_id=uid, weights=user_weights, sample_top=20, pick_n=15)
                            except Exception as e:
                                recommendations = None
                    elif search_type == 'director':
                        # 导演相似匹配（余弦相似度）
                        from sklearn.feature_extraction.text import TfidfVectorizer
                        director_cols = [c for c in ['DIRECTORS', 'director', 'directors'] if c in movies_df.columns]
                        from recommend_engine.engine import normalize_text
                        all_directors = []
                        director_map = {}
                        for col in director_cols:
                            vals = movies_df[col].dropna().astype(str).unique()
                            for v in vals:
                                all_directors.append(v)
                                director_map[v] = col
                        # 统一格式化所有导演名
                        norm_all_directors = [normalize_text(d) for d in all_directors]
                        norm_query = normalize_text(query)
                        # 直接模糊匹配
                        mask = None
                        for col in director_cols:
                            # 统一格式化每个导演名再做contains
                            norm_col = movies_df[col].fillna('').apply(normalize_text)
                            m = norm_col.str.contains(norm_query, na=False)
                            mask = m if mask is None else (mask | m)
                        matched = movies_df[mask] if mask is not None else pd.DataFrame()
                        if matched.empty and norm_all_directors:
                            # 余弦相似度找最相近导演（统一格式化后）
                            from sklearn.feature_extraction.text import TfidfVectorizer
                            tfidf = TfidfVectorizer().fit_transform([norm_query] + norm_all_directors)
                            from sklearn.metrics.pairwise import cosine_similarity
                            sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
                            idx = sims.argmax()
                            best_match = norm_all_directors[idx] if sims[idx] > 0.3 else None
                            if best_match:
                                # 找到原始导演名
                                orig_match = all_directors[idx]
                                guess_query = orig_match
                                mask2 = None
                                for col in director_cols:
                                    norm_col2 = movies_df[col].fillna('').apply(normalize_text)
                                    m2 = norm_col2 == best_match
                                    mask2 = m2 if mask2 is None else (mask2 | m2)
                                matched2 = movies_df[mask2] if mask2 is not None else pd.DataFrame()
                                recommendations = matched2.head(20)
                            else:
                                recommendations = pd.DataFrame()
                        else:
                            recommendations = matched.head(20)
                    elif search_type == 'douban_id':
                        # 通过豆瓣ID匹配电影，并将该电影作为输入用于相似电影推荐
                        id_cols = [c for c in ['MOVIE_ID', 'subject_id', 'douban_id'] if c in movies_df.columns]
                        mask = None
                        for col in id_cols:
                            m = movies_df[col].astype(str) == query
                            mask = m if mask is None else (mask | m)
                        matched = movies_df[mask] if mask is not None else pd.DataFrame()
                        if not matched.empty:
                            # 取第一个命中记录作为查询电影
                            row = matched.iloc[0]
                            movie_name = None
                            for ncol in ['NAME', 'title', 'name']:
                                if ncol in row.index and row.get(ncol):
                                    movie_name = row.get(ncol)
                                    break
                            if movie_name:
                                try:
                                    uid = current_user.id if current_user.is_authenticated else None
                                    recommendations = hybrid_recommend_for_user(movie_name, user_id=uid, weights=user_weights, sample_top=20, pick_n=15)
                                except Exception as e:
                                    print(f"[ERROR] hybrid recommend by douban_id failed: {e}")
                                    recommendations = pd.DataFrame()
                            else:
                                # 如果没有电影名，回退返回该电影条目
                                recommendations = matched.head(1)
                        else:
                            recommendations = pd.DataFrame()

    try:
        if isinstance(recommendations, pd.DataFrame):
            recommendations_serial = _df_to_records_safe(recommendations)
        else:
            recommendations_serial = recommendations
    except Exception:
        recommendations_serial = []

    return render_template('recommendations.html', title='Recommend', query=query, recommendations=recommendations_serial, engine_status=engine_status, search_type=search_type, guess_query=guess_query)


# --- 新增：三个模型的用户接口页面与 API ---
@app.route('/models')
def models_interface():
    # 页面包含三个单独的输入单元（cell），分别对应三个“模型”接口
    return render_template('models_interface.html')


def _df_to_records_safe(df):
    # 将 engine 返回的 DataFrame 或类似结构标准化为可序列化的 dict 列表
    import pandas as _pd
    records = []
    try:
        if df is None:
            return []
        # DataFrame-like
        if hasattr(df, 'to_dict'):
            try:
                r = df.to_dict(orient='records')
            except Exception:
                r = []
                try:
                    for _, row in getattr(df, 'iterrows', lambda: [])():
                        try:
                            r.append(dict(row))
                        except Exception:
                            pass
                except Exception:
                    r = []
            for item in r:
                if not isinstance(item, dict):
                    continue
                mid = item.get('MOVIE_ID') or item.get('douban_id') or item.get('subject_id') or item.get('MOVIE')
                name = item.get('NAME') or item.get('电影名') or item.get('title') or item.get('name')
                score = item.get('DOUBAN_SCORE') or item.get('rating') or item.get('豆瓣评分')
                sim = item.get('相似度') or item.get('similarity')
                # 导演字段可能的多语言列名
                directors = item.get('DIRECTORS') or item.get('导演') or item.get('director') or item.get('directors') or item.get('DIRECTOR')
                # 流派/标签字段可能的多语言列名
                label = item.get('LABEL') or item.get('流派') or item.get('GENRES') or item.get('TAGS') or item.get('genre')
                # 清理可能的 NaN/浮点异常值，确保为字符串或 None
                try:
                    if directors is not None:
                        if _pd.isna(directors):
                            directors = None
                        else:
                            directors = str(directors)
                except Exception:
                    try:
                        directors = str(directors)
                    except Exception:
                        directors = None
                try:
                    if label is not None:
                        if _pd.isna(label):
                            label = None
                        else:
                            label = str(label)
                except Exception:
                    try:
                        label = str(label)
                    except Exception:
                        label = None
                records.append({
                    'MOVIE_ID': str(mid) if mid is not None else None,
                    'NAME': name,
                    'DOUBAN_SCORE': score,
                    'SIMILARITY': sim,
                    'DIRECTORS': directors,
                    'LABEL': label
                })
            return records
        # list of dicts
        if isinstance(df, list):
            for item in df:
                if isinstance(item, dict):
                    mid = item.get('MOVIE_ID') or item.get('douban_id') or item.get('subject_id')
                    name = item.get('NAME') or item.get('电影名') or item.get('title')
                    score = item.get('DOUBAN_SCORE') or item.get('rating')
                    directors = item.get('DIRECTORS') or item.get('导演') or item.get('director')
                    label = item.get('LABEL') or item.get('流派') or item.get('GENRES')
                    # 清理 NaN 或非字符串值
                    try:
                        if directors is not None:
                            if _pd.isna(directors):
                                directors = None
                            else:
                                directors = str(directors)
                    except Exception:
                        try:
                            directors = str(directors)
                        except Exception:
                            directors = None
                    try:
                        if label is not None:
                            if _pd.isna(label):
                                label = None
                            else:
                                label = str(label)
                    except Exception:
                        try:
                            label = str(label)
                        except Exception:
                            label = None
                    records.append({
                        'MOVIE_ID': str(mid) if mid is not None else None,
                        'NAME': name,
                        'DOUBAN_SCORE': score,
                        'SIMILARITY': item.get('相似度'),
                        'DIRECTORS': directors,
                        'LABEL': label
                    })
            return records
    except Exception:
        return []
    return []


@app.route('/models/api/model1', methods=['POST'])
def models_api_model1():
    """Model1: 基于内容的相似度推荐（使用 recommend_engine.recommand）"""
    data = request.get_json() or request.form
    query = None
    if isinstance(data, dict):
        query = data.get('query')
    else:
        query = request.form.get('query')
    if not query:
        return jsonify({'error': 'query 不能为空'}), 400
    try:
        df = recommand(query, sample_top=30, pick_n=15)
        recs = _df_to_records_safe(df)
        return jsonify({'success': True, 'results': recs})
    except Exception as e:
        print(f"[ERROR] model1 failed: {e}")
        return jsonify({'error': '模型1调用失败'}), 500


@app.route('/models/api/model2', methods=['POST'])
def models_api_model2():
    """Model2: 增强的个性化推荐（如可用则使用用户偏好向量）"""
    data = request.get_json() or request.form
    query = None
    if isinstance(data, dict):
        query = data.get('query')
    else:
        query = request.form.get('query')
    if not query:
        return jsonify({'error': 'query 不能为空'}), 400
    try:
        user_pref = None
        if current_user.is_authenticated:
            try:
                user_pref = build_user_pref_vectors_from_user(current_user.id)
            except Exception:
                user_pref = None
        df = enhanced_recommend_for_user(query, user_pref_vectors=user_pref, sample_top=30, pick_n=15)
        recs = _df_to_records_safe(df)
        return jsonify({'success': True, 'results': recs})
    except Exception as e:
        print(f"[ERROR] model2 failed: {e}")
        return jsonify({'error': '模型2调用失败'}), 500


@app.route('/models/api/model3', methods=['POST'])
def models_api_model3():
    """Model3: 自由输入的模糊匹配（基于名称模糊搜索）"""
    data = request.get_json() or request.form
    query = None
    if isinstance(data, dict):
        query = data.get('query')
    else:
        query = request.form.get('query')
    if not query:
        return jsonify({'error': 'query 不能为空'}), 400
    try:
        df = get_movies_dataframe()
        if df is None or df.empty:
            return jsonify({'error': '电影数据未加载'}), 500
        # 尝试多种可能的标题列名
        title_cols = [c for c in ['NAME', '电影名', 'title', 'name'] if c in df.columns]
        if not title_cols:
            # 尝试所有列进行 contains
            title_cols = df.columns.tolist()
        # 构建布尔索引，按包含关系模糊匹配（不区分大小写）
        import pandas as _pd
        mask = _pd.Series([False] * len(df))
        for col in title_cols:
            try:
                mask = mask | df[col].astype(str).str.contains(str(query), case=False, na=False)
            except Exception:
                continue
        matched = df[mask]
        # 返回前 20 条
        matched = matched.head(20)
        recs = _df_to_records_safe(matched)
        return jsonify({'success': True, 'results': recs})
    except Exception as e:
        print(f"[ERROR] model3 failed: {e}")
        return jsonify({'error': '模型3调用失败'}), 500
    
@app.route('/health')
def health_check():
    return {
        "status": "success",
        "message": "Service is running",
        "engine_ready": True
    }, 200

if __name__ == '__main__':
    # 确保 instance 文件夹存在
    os.makedirs(os.path.join(app.root_path, 'instance'), exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True) # 生产环境请设置 debug=False