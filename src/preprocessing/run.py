from pathlib import Path

from preprocessing.pipelines.load_klines_to_hive import Klines2HiveUploader


def main():
    hive_uploader = Klines2HiveUploader(
        zipped_data_dir=Path("D:/data/zipped_data/1m"),
        temp_dir=Path("D:/data/temp"),
        output_dir=Path("D:/data/transformed/klines/1m"),
    )

    hive_uploader.run()


if __name__ == "__main__":
    main()
