# 日本語文法誤り検出器
事前に学習済み[単語分散表現](http://www.asahi.com/shimbun/medialab/word_embedding/)をダウンロードし `hyperparams/hyperparams_japanese_ged.py` の `word_embedding` に記載した箇所にに配置する。  

以下のコマンドを実行し前処理をする：
```
mkdir model_data
python prepro.py japanese_ged
```
学習のために以下のコマンドを実行する：
```
mkdir model
python train.py japanese_ged
```
学習されたのモデルを試す：

```
python predict.py '私へ元気です。' [word|char]
```
wordとcharによって予測単位を単語と文字で切り替える。  
出力：出力文、ラベル、予測確率
