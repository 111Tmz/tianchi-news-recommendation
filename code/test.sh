# 处理数据,这块用py3.7以上版本好像有点问题，建议用py3.7
python data.py --mode valid --logfile test.log

# itemcf 召回
python recall_itemcf.py --mode valid --logfile 2.log

# binetwork 召回
python recall_binetworkGpu.py --mode valid --logfile 3.log

# w2v 召回
python recall_w2v.py --mode valid --logfile 4.log

# 召回合并
python recall.py --mode valid --logfile 5.log

# 排序特征
python rank_feature.py --mode valid --logfile 6.log

# lgb 模型训练
python rank_lgb.py --mode valid --logfile 7.log
