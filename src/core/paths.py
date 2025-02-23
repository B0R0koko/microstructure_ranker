from pathlib import Path

DATA_DIR: Path = Path("D:\data")
HIVE_TRADES: Path = DATA_DIR.joinpath("transformed").joinpath("trades")
FEATURE_DIR: Path = DATA_DIR.joinpath("features")
