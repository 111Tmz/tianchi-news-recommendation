零基础入门推荐系统 - 新闻推荐 Top2  

比赛地址: https://tianchi.aliyun.com/competition/entrance/531842/introduction

# 解决方案
采用3种召回方式：itemcf 召回，binetwork 召回和基于 word2vec 的 i2i 召回。合并去重并删除没有召回到真实商品的用户数据后，利用特征工程+ LGB 二分类模型进行排序。

# 复现步骤
操作系统：ubuntu 16.04  
```
pip install requirements.txt
cd code
bash test.sh
```

# update
将test.sh的代码依次运行即可，注意，我在运行data.py时必须把python版本改成3.7，如果跑不了记得conda整个3.7的环境

电脑性能还可以的可以跑gpu版本，binework的会快不少，4060ti可以20分钟内跑完，整体比较费时间的就是rank和这个

最后的排序由于之前老是一步报错就得重新来，直接改成了notebook，自己修改的时候也可以跟着来。

然后本来还想试试冷启动和热度召回的，但是效果很不好，就没继续做了。

---
更改：
rank_lgb换cpu
recall_w2v，修复

换py3.11，
```
uv pip install 
numpy pandas tqdm scikit-learn multitasking annoy gensim lightgbm 
--index-url https://pypi.tuna.tsinghua.edu.cn/simple/
```

---

我已经阅读了你上传的两个文件：

* 你的 **Markdown 模板**：
* 你的 **完整 PDF 方案**：

你的要求本质是：

> **按照 README.md 的结构风格，把 PDF 内容重新整理成一份结构清晰的 Markdown 文档。**

我已经帮你 **重新结构化整合** 成一份完整的 Markdown（适合放 GitHub README / 项目说明 / 面试材料）。

---

# 天池新闻推荐系统方案整理（Markdown版）

本项目采用经典的 **Recall → Ranking 两阶段推荐架构**，通过 **多路召回 + 特征工程 + LightGBM排序** 实现新闻推荐系统。

整体流程：

```
用户行为日志
      ↓
数据处理
      ↓
多路召回
(ItemCF / BiNetwork / Word2Vec)
      ↓
召回融合
(MMS归一化 + 加权融合)
      ↓
特征工程
      ↓
LightGBM排序
      ↓
最终推荐结果
```

---

# 1 数据处理

数据来源：

```
train_click_log.csv
test_click_log.csv
article.csv
```

为了构建稳定的 **离线验证集 (offline evaluation)**：

1. 从训练用户中 **随机抽取 5w 用户**
2. 将这些用户的 **最后一次点击作为验证集 label**
3. 从训练点击日志中 **删除该点击**
4. 剩余点击 + 测试点击 → 构建历史行为

最终得到两个数据集：

```
df_click
df_query
```

* **df_click**

历史点击日志

```
user_id
click_article_id
click_timestamp
```

* **df_query**

需要预测的用户

```
user_id
click_article_id
```

其中：

```
click_article_id = -1
表示测试用户
```

---

# 2 多路召回 (Recall)

为了提高召回覆盖率，系统采用 **三路召回策略**：

```
1 ItemCF召回
2 BiNetwork召回
3 Word2Vec召回
```

最终进行融合。

---

# 2.1 ItemCF召回（改进版）

## 2.1.1 用户点击序列

先构建用户历史点击：

```python
user_item_ = df.groupby('user_id')['click_article_id'].agg(list)
user_item_dict = dict(zip(user_item_['user_id'], user_item_['click_article_id']))
```

结果：

```
user_id → [article1, article2, article3]
```

---

## 2.1.2 Item相似度计算

基于 **用户行为共现** 计算相似度。

对于同一用户点击序列：

```
items = [i1, i2, i3 ...]
```

任意两个文章：

```
i
j
```

贡献值：

```
weight =
α × 0.9^(|loc2-loc1|-1)
-------------------------
log(1 + N)
```

其中：

| 变量  | 含义    |
| --- | ----- |
| α   | 顺序权重  |
| N   | 用户点击数 |
| loc | 位置差   |

顺序权重：

```
loc2 > loc1 → α = 1
loc2 < loc1 → α = 0.7
```

位置衰减：

```
0.9^(|loc2-loc1|-1)
```

用户活跃度惩罚：

```
1 / log(1 + N)
```

累计得到：

```
c_ij
```

最终相似度：

```
sim(i,j) =
c_ij / sqrt(count(i) * count(j))
```

解决：

* 热门物品偏置
* 活跃用户偏置

---

## 2.1.3 ItemCF召回

召回逻辑：

1. 获取用户最近 **2次点击**

```
[user_last, user_second_last]
```

2. 每个点击物品取

```
Top200相似物品
```

3. 累加得分

```
rank[item] += sim_score * (0.7 ** loc)
```

loc：

```
0 → 最近点击
1 → 次近点击
```

最终：

```
Top100候选
```

---

# 2.2 BiNetwork召回

基于 **用户-物品二部图**。

---

## 2.2.1 构建倒排表

```
item → users
```

代码：

```python
item_user_ = df.groupby('click_article_id')['user_id'].agg(list)
item_user_dict = dict(zip(item_user_['click_article_id'], item_user_['user_id']))
```

---

## 2.2.2 相似度计算

基于 **共同用户数**

```
sim(i,j) +=
1 / (log(|Ui|+1) * log(|Uj|+1))
```

其中：

```
Ui = 点击 i 的用户集合
Uj = 点击 j 的用户集合
```

作用：

降低：

```
热门物品
活跃用户
```

影响。

---

## 2.2.3 召回

只使用：

```
用户最后一次点击
```

步骤：

```
1 找到最后点击文章
2 找到相似文章Top100
3 累加得分
4 取Top50
```

---

# 2.3 Word2Vec召回

利用 **用户行为序列训练Embedding**。

---

## 2.3.1 训练Word2Vec

用户点击序列：

```
[user1] → A B C D
[user2] → B E F
```

训练：

```python
model = Word2Vec(
    sentences=sentences,
    vector_size=256,
    window=3,
    sg=1,
    negative=5,
    epochs=1
)
```

参数说明：

| 参数          | 含义        |
| ----------- | --------- |
| vector_size | 向量维度      |
| window      | 上下文窗口     |
| sg          | Skip-gram |
| negative    | 负采样       |

---

## 2.3.2 构建文章向量

```
article_vec_map
```

```
article_id → embedding
```

---

## 2.3.3 Annoy召回

步骤：

```
1 获取用户最近点击文章
2 获取对应embedding
3 Annoy查询最近100个
```

距离转换为相似度：

```
sim = 2 - distance
```

最终：

```
Top50候选
```

---

# 2.4 多路召回融合

三路召回：

```
ItemCF
BiNetwork
Word2Vec
```

---

## 2.4.1 MMS归一化

不同用户得分分布不同。

采用：

```
Min-Max Scaling
```

公式：

```
score' =
(score - min)
-------------
(max - min)
```

代码逻辑：

```python
(sim_score - min_score) / (max_score - min_score)
```

归一化到：

```
[0,1]
```

---

## 2.4.2 加权融合

权重：

```
itemcf = 1
binetwork = 1
w2v = 0.1
```

原因：

```
w2v效果较差
```

融合：

```
score = w1*s1 + w2*s2 + w3*s3
```

---

## 2.4.3 召回指标

评价指标：

```
HitRate@K
MRR@K
```

融合后：

```
HitRate@50 = 0.6766
MRR@50 = 0.2466
```

---

# 3 特征工程

召回结果：

```
user_id
article_id
sim_score
label
```

构造三类特征。

---

# 3.1 用户-文章交互特征

例如：

```
user_last_click_article_itemcf_sim
```

表示：

```
候选文章
与
最后点击文章
的ItemCF相似度
```

还有：

```
user_clicked_article_itemcf_sim_sum
user_last_click_article_binetwork_sim
user_last_click_article_w2v_sim
```

---

# 3.2 用户行为特征

例如：

```
user_id_cnt
```

用户点击次数。

```
user_id_click_diff_mean
```

用户点击间隔。

```
user_click_datetime_hour
```

用户活跃时间。

---

# 3.3 文章特征

来自：

```
article.csv
```

包括：

```
category_id
words_count
created_at_ts
```

---

# 4 排序模型

使用：

```
LightGBM
```

---

# 4.1 模型参数

```python
LGBMClassifier(
    num_leaves=64,
    max_depth=10,
    learning_rate=0.05,
    n_estimators=10000,
    subsample=0.8,
    feature_fraction=0.8,
    reg_alpha=0.5,
    reg_lambda=0.5
)
```

关键参数：

| 参数            | 含义    |
| ------------- | ----- |
| num_leaves    | 叶子节点数 |
| max_depth     | 树深    |
| learning_rate | 学习率   |
| n_estimators  | 树数量   |

---

# 4.2 交叉验证

使用：

```
GroupKFold
```

原因：

```
防止同一用户出现在训练和验证中
```

代码：

```python
GroupKFold(n_splits=5)
```

训练流程：

```
1 划分5折
2 每次4折训练
3 1折验证
4 预测测试集
```

---

# 5 最终推荐

对预测结果：

```
按用户排序
```

取：

```
Top5
```

作为最终推荐。

---

# 6 最终效果

排序后指标：

```
HitRate@5  = 0.4400
HitRate@50 = 0.7861
MRR@50     = 0.2938
Accuracy   = 0.8156
```
