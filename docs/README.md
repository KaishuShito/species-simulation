# Predator-Prey Simulation

2Dグリッド世界で草・草食動物・肉食動物をシミュレートし、個体数の推移を観察できるシステム。

## クイックスタート

```bash
# 依存インストール
pip install -r requirements.txt

# ヘッドレス実行（グラフ出力）
python simulation.py

# リアルタイム可視化
python simulation.py -v
```

## 目次

- [インストール](#インストール)
- [基本的な使い方](#基本的な使い方)
- [可視化モード](#可視化モード)
- [設定ファイル](#設定ファイル)
- [CLIオプション](#cliオプション)
- [シミュレーションの仕組み](#シミュレーションの仕組み)
- [拡張ポイント](#拡張ポイント)

---

## インストール

### 必要環境

- Python 3.10+
- macOS / Linux / Windows

### 依存パッケージ

```bash
pip install -r requirements.txt
```

| パッケージ | 用途 |
|-----------|------|
| numpy | グリッド処理 |
| matplotlib | グラフ出力 |
| pyyaml | 設定ファイル読み込み |
| pygame | リアルタイム可視化 |

---

## 基本的な使い方

### ヘッドレス実行

```bash
# デフォルト設定で1000ステップ実行
python simulation.py

# バランス版設定で実行
python simulation.py -c config_balanced.yaml

# ステップ数を指定
python simulation.py -s 2000

# シードを変更（別の初期状態）
python simulation.py --seed 123

# グラフ表示なし（保存のみ）
python simulation.py --no-show

# ログ出力を抑制
python simulation.py -q
```

### 出力

- コンソール: 100ステップごとの個体数ログ
- `population_graph.png`: 個体数推移グラフ

---

## 可視化モード

```bash
# 基本的な可視化起動
python simulation.py -v

# オプション調整
python simulation.py -v --cell-size 6 --fps 60 --speed 5
```

### 操作キー

| キー | 動作 |
|------|------|
| `SPACE` | 一時停止 / 再開 |
| `UP` | 速度を上げる（ステップ/フレーム +1） |
| `DOWN` | 速度を下げる |
| `S` | 一時停止中に1ステップ進める |
| `R` | シミュレーションをリセット |
| `G` | 現在のグラフを保存 |
| `ESC` / `Q` | 終了 |

### 画面構成

```
+---------------------------+---------------+
|                           | Population    |
|      Grid View            | Stats         |
|                           |               |
|  緑: 草                    | Controls      |
|  青丸: 草食動物            |               |
|  赤丸: 肉食動物            | Mini Graph    |
|                           |               |
+---------------------------+---------------+
```

---

## 設定ファイル

### config.yaml（デフォルト）

```yaml
world:
  width: 100          # グリッド幅
  height: 100         # グリッド高さ
  seed: 42            # 乱数シード（再現性）

grass:
  initial_density: 0.3    # 初期の草密度 (0-1)
  regrowth_rate: 0.05     # 再生確率/ステップ
  max_density: 0.5        # 最大密度

day_night:
  cycle_length: 100           # 1日のステップ数
  day_ratio: 0.6              # 昼の割合 (0-1)
  herbivore_day_activity: 1.0     # 草食動物の昼の活動倍率
  herbivore_night_activity: 0.5   # 草食動物の夜の活動倍率
  carnivore_day_activity: 0.7     # 肉食動物の昼の活動倍率
  carnivore_night_activity: 1.0   # 肉食動物の夜の活動倍率

herbivore:
  initial_count: 50       # 初期個体数
  max_count: 500          # 最大個体数（上限）
  initial_energy: 50      # 初期エネルギー
  max_energy: 100         # 最大エネルギー
  move_cost: 1            # 移動コスト
  eat_gain: 20            # 草を食べた時のエネルギー
  reproduce_threshold: 70 # 繁殖に必要なエネルギー
  reproduce_cost: 40      # 繁殖時に消費するエネルギー
  vision_range: 5         # 視野範囲
  flee_range: 4           # 逃避開始距離（0=逃げない）
  flee_speed: 1           # 逃走時の追加移動回数

carnivore:
  initial_count: 20
  max_count: 200
  initial_energy: 80
  max_energy: 150
  move_cost: 2            # 草食より高い
  eat_gain: 50            # 草食を食べた時のエネルギー
  reproduce_threshold: 100
  reproduce_cost: 60
  vision_range: 7         # 草食より広い
  flee_range: 0           # 肉食は逃げない
  flee_speed: 0

simulation:
  steps: 1000             # 実行ステップ数
  display_interval: 100   # ログ出力間隔
```

### config_balanced.yaml（安定版）

生態系が長期間安定するようにチューニングされた設定。

```bash
python simulation.py -c config_balanced.yaml
```

### カスタム設定の作成

```bash
cp config.yaml my_config.yaml
# my_config.yaml を編集
python simulation.py -c my_config.yaml
```

---

## CLIオプション

```
usage: simulation.py [-h] [-c CONFIG] [-s STEPS] [--seed SEED] [-o OUTPUT]
                     [--no-show] [-q] [-v] [--cell-size CELL_SIZE]
                     [--fps FPS] [--speed SPEED]

options:
  -h, --help            ヘルプを表示
  -c, --config CONFIG   設定ファイルパス (default: config.yaml)
  -s, --steps STEPS     ステップ数を上書き
  --seed SEED           乱数シードを上書き
  -o, --output OUTPUT   グラフ出力パス (default: population_graph.png)
  --no-show             グラフウィンドウを表示しない
  -q, --quiet           ステップログを抑制
  -v, --visual          Pygame可視化モードで実行
  --cell-size SIZE      セルサイズ（ピクセル）(default: 8)
  --fps FPS             目標FPS (default: 30)
  --speed SPEED         ステップ/フレーム (default: 1)
```

---

## シミュレーションの仕組み

### エンティティ

| 種類 | 行動 |
|------|------|
| 草 | 時間経過で再生（密度上限あり） |
| 草食動物 | 視野内の草を探して移動・捕食、捕食者から逃避 |
| 肉食動物 | 視野内の草食動物を探して捕食 |

### ライフサイクル

1. **昼夜判定**: 現在のステップから昼/夜を判定、活動倍率を適用
2. **逃避判定**: 草食動物は近くの肉食動物を検知すると逃げる
3. **移動**: 目標に向かって移動（エネルギー消費）
4. **捕食**: 目標と同じセルにいれば食べる（エネルギー獲得）
5. **繁殖**: エネルギーが閾値を超えると子を生成
6. **死亡**: エネルギーが0以下で死亡

### 昼夜サイクル

- **昼**: 草食動物が活発に活動、肉食動物はやや不活発
- **夜**: 肉食動物が活発に狩り、草食動物は動きが鈍くなる
- 活動倍率は視野範囲と行動確率に影響
- 可視化時は画面の明るさで昼夜を表現

### 逃避行動

草食動物は `flee_range` 内に肉食動物を検知すると：
1. 捕食者から遠ざかる方向に移動
2. `flee_speed` に応じて追加移動（エネルギーコスト軽減）
3. 逃避中は草を探さない（生存優先）

### パス探索

現在は貪欲法（GreedyPathFinder）を使用:
- 視野内で最も近い目標を選択
- 目標に向かって1マス移動

A*アルゴリズムへの差し替えインターフェースは用意済み。

### 個体数抑制

- `max_count`: 種ごとの最大個体数
- エネルギー閾値による繁殖制限
- 草の最大密度制限

---

## 拡張ポイント

### A*パス探索の実装

`simulation.py` の `AStarPathFinder` クラスを実装:

```python
class AStarPathFinder:
    def find_next_step(
        self,
        start: tuple[int, int],
        targets: list[tuple[int, int]],
        blocked: set[tuple[int, int]],
    ) -> Optional[tuple[int, int]]:
        # A*アルゴリズムを実装
        pass
```

### 新しい種の追加

1. `CreatureConfig` に新しい設定を追加
2. `World` クラスに新しいリストを追加
3. `_step_*` メソッドを追加

### 地形の追加

`World` クラスに障害物グリッドを追加し、移動時に考慮。

---

## ファイル構成

```
.
├── simulation.py       # メインシミュレーション
├── visualizer.py       # Pygame可視化
├── config.yaml         # デフォルト設定
├── config_balanced.yaml # 安定版設定
├── requirements.txt    # 依存パッケージ
├── docs/
│   └── README.md       # このドキュメント
└── population_graph.png # 出力グラフ（実行後）
```
