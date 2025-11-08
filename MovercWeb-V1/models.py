# models.py
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True, nullable=False)
    email = db.Column(db.String(120), index=True, unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    # 关联关系：一个用户有多部喜欢和不喜欢的电影
    liked_movies = db.relationship('UserMoviePreference', backref='user', lazy='dynamic',
                                   foreign_keys='UserMoviePreference.user_id',
                                   primaryjoin='User.id==UserMoviePreference.user_id')
    
    disliked_movies = db.relationship('UserMovieDislike', backref='user', lazy='dynamic',
                                      foreign_keys='UserMovieDislike.user_id',
                                      primaryjoin='User.id==UserMovieDislike.user_id')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return '<User {}>'.format(self.username)


# 注意：这里的 Movie 表主要用于关联用户偏好，实际电影信息仍从 CSV 加载
# 如果需要完全数据库化，可以将 movies_new 的内容也存入数据库
class Movie(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    douban_id = db.Column(db.String(20), unique=True, nullable=False) # 对应 movies_new['MOVIE_ID']
    title = db.Column(db.String(255), nullable=False) # 对应 movies_new['NAME']

    def __repr__(self):
        return '<Movie {}>'.format(self.title)

# 用户喜欢的电影关联表 (多对多简化为一对多，指向 Movie ID)
class UserMoviePreference(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_douban_id = db.Column(db.String(20), nullable=False) # 直接存储豆瓣ID，避免频繁查Movie表
    # 可选：添加时间戳等字段

# 用户不喜欢的电影关联表
class UserMovieDislike(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_douban_id = db.Column(db.String(20), nullable=False)