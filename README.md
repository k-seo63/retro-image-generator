# retro-image-generator

[これ](https://dailyportalz.jp/kiji/retro_PC_game-mitaina-shashin)をPythonで実装してみたい

## 環境構築

[ここ](https://zenn.dev/zenizeni/books/a64578f98450c2/viewer/c6af80)を参考にする

- pyenvはWindows版があるみたい
  - [ここ](https://qiita.com/probabilityhill/items/9a22f395a1e93206c846)を参考にして入れた
  - PowerShellはネットから落としたファイルを実行するのに手間がいるみたいだが、git bashから呼べば手間がいらないみたい
  - 3.11.1をinstallしてglobalにした
- poetry
  - PowerShellから以下を実行
    - `(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -`
  - To get started you need Poetry's bin directory in your `PATH` environment variable. の指示に従ってこのパスを環境変数に追加
- 仮想環境(venv)
  - `python -m venv .venv` で作成
  - `poetry init` でプロジェクトを初期化
    - 既にpyproject.tomlがあるなら、`poetry install`でdependenciesをインストール
  - `poetry add opencv-python` とかでパッケージをインストール
    - dev.dependenciesなら`-D`をつければヨシ
- 実行
  -  `poetry run python <python script>` でpoetryがvenvの中で実行してくれる


## 実行方法

1. image/input配下に画像ファイルを配置。
2. generator.pyのparametaを入力。
    - img_name: 拡張子を含まないファイル名(jpg)
    - resize_ratio: 800x800以下くらいになるような倍率をいれるといいかも
    - saturation_ratio: 彩度を上げるときの倍率、上げ過ぎると白飛びしたりするかも
3. 下記コマンドで実行。
  ```
  poetry run python src/generator.py
  ```
