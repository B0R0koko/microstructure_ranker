from pathlib import Path

DATA_DIR: Path = Path(r"D:\data")
SPOT_TRADES: Path = DATA_DIR.joinpath("transformed").joinpath("trades")
USDM_TRADES: Path = DATA_DIR.joinpath("transformed").joinpath("USDM")
FEATURE_DIR: Path = DATA_DIR.joinpath("features")
