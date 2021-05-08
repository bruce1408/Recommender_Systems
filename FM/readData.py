# coding=gbk
# MovieLens 1M数据集含有来自6000名用户对4000部电影的100万条评分数据。
# 分为三个表：评分，用户信息，电影信息。这些数据都是dat文件格式
# ，可以通过pandas.read_table将各个表分别读到一个pandas DataFrame对象中
import pandas as pd
import time

start = time.clock()
filename1 = '/Users/bruce_trie/Downloads/ml-1m/users.dat'
filename2 = '/Users/bruce_trie/Downloads/ml-1m/ratings.dat'
filename3 = '/Users/bruce_trie/Downloads/ml-1m/movies.dat'
pd.options.display.max_rows = 100
uname = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table(filename1, sep='::', header=None, names=uname, engine='python')

# 年龄和职业都是使用编码的形式给出来的
print(users.head())
#    user_id gender  age  occupation    zip
# 0        1      F    1          10  48067
# 1        2      M   56          16  70072
# 2        3      M   25          15  55117
# 3        4      M   45           7  02460
# 4        5      M   25          20  55455
print(users.shape)  # (6040, 5)

rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table(filename2, header=None, sep='::', names=rnames, engine='python')
print(ratings.head())
#    user_id  movie_id  rating  timestamp
# 0        1      1193       5  978300760
# 1        1       661       3  978302109
# 2        1       914       3  978301968
# 3        1      3408       4  978300275
# 4        1      2355       5  978824291

# print(ratings.shape)  #(1000209, 4)
mnames = ['movie_id', 'title', 'genres']  # genres 表示影片的体裁是什么
movies = pd.read_table(filename3, header=None, sep='::', names=mnames, engine='python')
# print(movies.head())
#    movie_id                               title                        genres
# 0         1                    Toy Story (1995)   Animation|Children's|Comedy
# 1         2                      Jumanji (1995)  Adventure|Children's|Fantasy
# 2         3             Grumpier Old Men (1995)                Comedy|Romance
# 3         4            Waiting to Exhale (1995)                  Comedy|Drama
# 4         5  Father of the Bride Part II (1995)                        Comedy
# print(movies.shape) #(3883, 3)
