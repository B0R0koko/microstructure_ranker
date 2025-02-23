from pathlib import Path

from preprocessing.pipelines.load_trades_to_hive import Trades2HiveUploader


def main():
    hive_uploader = Trades2HiveUploader(
        zipped_data_dir=Path("D:/data/zipped_data/trades"),
        output_dir=Path("D:/data/transformed/trades-gzip"),
    )

    hive_uploader.run_multiprocessing(processes=15)


if __name__ == "__main__":
    main()
