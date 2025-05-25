from pathlib import Path

DATA_DIR: Path = Path(r"D:\data")
RAW_DATA_DIR: Path = DATA_DIR / "raw"
TRANSFORMED_DATA_DIR: Path = DATA_DIR / "transformed"

TEST_DATA_DIR: Path = DATA_DIR / "test"

# Raw zip files
BINANCE_SPOT_RAW_TRADES: Path = RAW_DATA_DIR / "binance" / "spot" / "trades"
BINANCE_USDM_RAW_TRADES: Path = RAW_DATA_DIR / "binance" / "usdm" / "trades"
BINANCE_USDM_RAW_L1: Path = RAW_DATA_DIR / "binance" / "usdm" / "l1"
OKX_SPOT_RAW_TRADES: Path = RAW_DATA_DIR / "okx" / "spot" / "trades"

# HIVE locations
BINANCE_SPOT_HIVE_TRADES: Path = TRANSFORMED_DATA_DIR / "binance" / "spot" / "trades"
BINANCE_USDM_HIVE_TRADES: Path = TRANSFORMED_DATA_DIR / "binance" / "usdm" / "trades"
OKX_SPOT_HIVE_TRADES: Path = TRANSFORMED_DATA_DIR / "okx" / "spot" / "trades"

FEATURE_DIR: Path = DATA_DIR.joinpath("features")
