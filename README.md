# 日本語文法誤り検出器
[文法誤り検出モデル](https://drive.google.com/open?id=1Vcgyi2YwjWHzchmjyUWPkD5o5MHlalw5)をダウンロードする。
charが文字単位、wordが単語単位

```
python predict.py '私へ元気です。' [word|char]
```

wordとcharによって予測単位を単語と文字で切り替える。
