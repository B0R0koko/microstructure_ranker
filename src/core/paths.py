from pathlib import Path

DATA_DIR: Path = Path(r"D:\data")

SPOT_TRADES: Path = DATA_DIR.joinpath("transformed").joinpath("trades")
USDM_TRADES: Path = DATA_DIR.joinpath("transformed").joinpath("USDM")
USDM_L1: Path = DATA_DIR.joinpath("transformed").joinpath("USDM-L1")
OKX_SPOT_TRADES: Path = DATA_DIR.joinpath("transformed").joinpath("OKX-SPOT")

FEATURE_DIR: Path = DATA_DIR.joinpath("features")
