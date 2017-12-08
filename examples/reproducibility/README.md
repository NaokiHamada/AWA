# 実験の再現性を担保する
AWAは確率的な最適化アルゴリズムを用いているため，通常は実験を行うたびに異なる結果が得られます．
設定ファイルに`seed`項目を記述することで，乱数種を固定し，決定的で再現性のある動作をさせることができます．
```json
"optimizer": {
  "seed": 42
}
```
## 再現性とは
AWAにおける再現性とは， __等価な__ 設定ファイルを与えたときに， __等価な__ 結果ファイルが出力されることを意味します．
設定ファイル`a.json`と`b.json`が __等価__ であるとは，以下の2条件がともに成り立つことを意味します．
- `a.json`と`b.json`をそれぞれ読み込んだPython辞書オブジェクト`dict_a`と`dict_b`からそれぞれ`["optimizer"]["cache"]`キーを取り除いたとき，`dict_a == dict_b`が成り立つこと．
- `["optimizer"]["cache"]`の値が存在しないパスもしくは空のファイルを指すこと．

結果ファイル`a.csv`と`b.csv`が __等価__ であるとは，以下のことを意味します．
- `a.csv`と`b.csv`の内容からそれぞれ`Evaluation Start`列と`Evaluation End`列を取り除いたとき，両者の内容がファイルとして一致すること，すなわち`diff $(cut -d, -f 1,4- a.csv) $(cut -d, -f 1,4- b.csv)`の終了コードが`0`であること．

## ユーザプログラム実装上の注意点
再現性を担保するためには，上記の乱数種の固定に加えて，ユーザプログラムが純粋関数的に振る舞う必要があります．
すなわち，ユーザプログラムは同じ入力に対して常に同じ結果を出力する必要があります．
この条件が満たされない場合には，AWAの乱数種を固定しても再現性は担保されません．
ユーザプログラムが並列処理や入出力を用いているときには，純粋関数性を担保するために注意が必要です．

## 例
乱数種を固定することにより，再現性のある実験が行われることを確認してみましょう．
以下のコマンドを実行します．
```
$ ./run.sh
```

このスクリプトは2つの実験を行い，その結果を比較します．

実験は以下のように実行されます．
```
# Run the first experiment with a random seed
$ python -m awa -c config_1.json  # Produce results_1.csv

# Run the second experiment with the same seed
$ python -m awa -c config_2.json  # Produce results_2.csv
```

結果ファイル（の時刻を除く項目）を比較すると，両者が一致していることがわかります．
```
# Cut the timestamp columns off
cut -d, -f2,3 --complement results_1.csv > solutions_1.csv
cut -d, -f2,3 --complement results_2.csv > solutions_2.csv

# They should have the same solutions
diff solutions_1.csv solutions_2.csv  # No output means the same contents
```
