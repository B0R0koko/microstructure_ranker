from pathlib import Path

from preprocessing.pipelines.load_trades_to_hive import Trades2HiveUploader


def main():
    hive_uploader = Trades2HiveUploader(
        zipped_data_dir=Path("D:/data/zipped_data/trades"),
        temp_dir=Path("D:/data/temp"),
        output_dir=Path("D:/data/transformed/trades"),
    )

    hive_uploader.run_multiprocessing()


if __name__ == "__main__":
    main()
