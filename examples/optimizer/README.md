# 最適化手法の設定
設定ファイルを通して，最適化手法のパラメータを調節する例です．

最適化手法のパラメータは`optimizer`という項目に記述します．
```json
{
  "optimizer": {
    "seed": 42,
    "max_iters": 3,
    "max_evals": 64,
    "scalarization": "weighted_sum",
    "x0": [
      [0.7, 0.7],
      [0.3, 0.3]
    ],
    "w0": [
      [0.8, 0.2],
      [0.3, 0.7]
    ]
  }
}
```
- `seed`: 乱数の種です．
- `max_iters`: AWAの反復数です．
- `max_evals`: AWAの中で走るCMA-ES 1回あたりの最大評価回数です．
- `scalarization`: スカラー化手法です．
- `x0`: 初期解です．
- `w0`: 初期重みです．

より詳しくは[設定ファイルの仕様書](../../FORMATS.md#optimizer)をご覧ください．
