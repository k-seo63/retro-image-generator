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
  - `poetry add opencv-python` とかでパッケージをインストール
- 実行
  -  `poetry run python <python script>` でpoetryがvenvの中で実行してくれる