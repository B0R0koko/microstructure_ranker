import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from core.currency_pair import CurrencyPair
from core.exchange import Exchange
from core.paths import get_root_dir
from core.utils import configure_logging
from feature_writer.Pumps.enums import PumpEvent


def load_pumps(path: Path) -> List[PumpEvent]:
    """
    path: Path - path to the JSON file with labeled known pump events
    returns: List[PumpEvent]
    """
    pump_events: List[PumpEvent] = []
    with open(path) as file:
        for event in json.load(file):
            pump_events.append(
                PumpEvent(
                    currency_pair=CurrencyPair.from_string(symbol=event["symbol"]),
                    time=datetime.strptime(event["time"], "%Y-%m-%d %H:%M:%S"),
                    exchange=Exchange.parse_from_lower(event["exchange"]),
                )
            )
    return pump_events


def main():
    configure_logging()
    path: Path = get_root_dir() / "src/feature_writer/Pumps/resources/pumps.json"
    logging.info("\n%s", load_pumps(path=path))


if __name__ == "__main__":
    main()
