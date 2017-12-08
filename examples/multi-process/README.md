# 1ノードの複数プロセスで並列化する
ユーザプログラムの実行を1ノードの複数プロセスで並列化する設定ファイルの例です．
以下のように，並列化したい数だけ`wokers`の要素を並べるだけでOKです．
```json
{
  "workers": [
    { "command": "./sphere_2D.sh $parameters" },
    { "command": "./sphere_2D.sh $parameters" },
    { "command": "./sphere_2D.sh $parameters" },
    { "command": "./sphere_2D.sh $parameters" }
  ]
}
```
