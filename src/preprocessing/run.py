from pathlib import Path

from preprocessing.pipelines.binance_usdm_trades_to_hive import BinanceUSDM2Hive


def main():
    hive_uploader = BinanceUSDM2Hive(
        zipped_data_dir=Path("D:/data/zipped_data/USDM"),
        output_dir=Path("D:/data/transformed/USDM"),
    )

    hive_uploader.run_multiprocessing(processes=15)


if __name__ == "__main__":
    main()
