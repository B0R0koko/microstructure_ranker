from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta, datetime, time
from enum import Enum
from typing import Optional, List

import pandas as pd


def get_last_day_month(date_to_round: date) -> date:
    """Gets the last date of the month"""
    day: int = monthrange(year=date_to_round.year, month=date_to_round.month)[1]
    return date(year=date_to_round.year, month=date_to_round.month, day=day)


def get_first_day_month(date_to_round: date) -> date:
    """Returns the first day of the month"""
    return date(year=date_to_round.year, month=date_to_round.month, day=1)


def generate_month_time_chunks(start_date: date, end_date: date) -> Optional[List[date]]:
    """Generate a list of months that lie entirely within given interval of start and end dates"""
    date_months: List[date] = [
        _date.date() for _date in pd.date_range(start_date, end_date, freq="MS", inclusive="both").tolist()
    ]
    # check if the last value is correct
    if not date_months:
        return

    if date_months[-1] == get_first_day_month(end_date):
        date_months.pop(-1)
    return date_months


def _convert_to_dates(dates: pd.DatetimeIndex) -> List[date]:
    return [el.date() for el in dates]


def get_seconds_slug(td: timedelta) -> str:
    if td.total_seconds() < 1:
        return f"{int(td.total_seconds() * 1000)}MS"
    assert td.total_seconds() % 1 == 0, "Above second timedeltas must be a multiple of 1 second"
    return f"{int(td.total_seconds())}S"


def generate_daily_time_chunks(start_date: date, end_date: date) -> Optional[List[date]]:
    days: List[date] = []

    if start_date != get_first_day_month(start_date):
        days.extend(
            _convert_to_dates(
                pd.date_range(start_date, get_last_day_month(start_date), freq="D", inclusive="both")
            )
        )

    if end_date != get_first_day_month(end_date):
        days.extend(
            _convert_to_dates(
                pd.date_range(get_first_day_month(end_date), end_date, freq="D", inclusive="both")
            )
        )

    return days


def start_of_the_day(day: date) -> datetime:
    """Converts date to datetime with 0:00 time"""
    return datetime.combine(date=day, time=time(hour=0, minute=0, second=0))


def end_of_the_day(day: date) -> datetime:
    """Converts date to datetime with 23:59:59:9999 time"""
    return start_of_the_day(day=day) + timedelta(days=1) - timedelta(microseconds=1)


def format_date(day: date) -> str:
    return day.strftime("%Y%m%d")


@dataclass
class Bounds:
    start_inclusive: datetime
    end_exclusive: datetime

    @classmethod
    def from_datetime_str(cls, start_inclusive: str, end_exclusive: str) -> "Bounds":
        return cls(
            start_inclusive=datetime.strptime(start_inclusive, "%Y-%m-%d %H:%M:%S"),
            end_exclusive=datetime.strptime(end_exclusive, "%Y-%m-%d %H:%M:%S"),
        )

    @classmethod
    def for_days(cls, start_inclusive: date, end_exclusive: date) -> "Bounds":
        """
        For instance if we pass start_inclusive = date(2024, 11, 1) and end_exclusive = date(2024, 12, 1),
        Final Bounds will have the following datetime (2024-11-01 0:00:00, 2024-11-30 23:59:59)
        """
        return cls(
            start_inclusive=start_of_the_day(day=start_inclusive),
            end_exclusive=end_of_the_day(day=end_exclusive - timedelta(days=1)),
        )

    @classmethod
    def for_day(cls, day: date) -> "Bounds":
        return cls(
            start_inclusive=start_of_the_day(day=day),
            end_exclusive=end_of_the_day(day=day),
        )

    @property
    def day0(self) -> date:
        return self.start_inclusive.date()

    @property
    def day1(self) -> date:
        return self.end_exclusive.date()

    def __str__(self) -> str:
        return f"Bounds: {self.start_inclusive} - {self.end_exclusive}"

    def generate_overlapping_bounds(self, step: timedelta, interval: timedelta) -> List["Bounds"]:
        """Returns a list of bounds created from parent Bounds interval with a certain interval size and step"""
        intervals: List["Bounds"] = []

        lb = self.start_inclusive

        while True:
            rb: datetime = lb + interval
            # create new overlapping sub-Bounds
            intervals.append(
                Bounds(start_inclusive=lb, end_exclusive=min(rb - timedelta(microseconds=1), self.end_exclusive))
            )
            lb += step

            if rb >= self.end_exclusive:
                break

        return intervals

    def contain_days(self, day: date) -> bool:
        return self.day0 <= day <= self.day1

    def create_offset_bounds(self, time_offset: "TimeOffset") -> "Bounds":
        """Returns Bounds for the interval which is used to compute target"""
        return Bounds(
            start_inclusive=self.end_exclusive,
            end_exclusive=self.end_exclusive + time_offset.value,
        )

    def expand_bounds(
            self, lb_timedelta: Optional[timedelta] = None, rb_timedelta: Optional[timedelta] = None
    ) -> "Bounds":
        return Bounds(
            start_inclusive=self.start_inclusive - lb_timedelta if lb_timedelta else self.start_inclusive,
            end_exclusive=self.end_exclusive + rb_timedelta if rb_timedelta else self.end_exclusive,
        )

    def date_range(self):
        for dt in pd.date_range(self.day0, self.day1, freq="1D", inclusive="both"):
            yield dt.date()

    def __eq__(self, other) -> bool:
        return self.start_inclusive == other.start_inclusive and self.end_exclusive == other.end_exclusive


class TimeOffset(Enum):
    HALF_SECOND: timedelta = timedelta(milliseconds=500)
    ONE_SECOND: timedelta = timedelta(seconds=1)
    FIVE_SECONDS: timedelta = timedelta(seconds=5)
    TEN_SECONDS: timedelta = timedelta(seconds=10)
    HALF_MINUTE: timedelta = timedelta(seconds=30)
    MINUTE: timedelta = timedelta(minutes=1)
    FIVE_MINUTES: timedelta = timedelta(minutes=5)
    FIFTEEN_MINUTES: timedelta = timedelta(minutes=15)
    HALF_HOUR: timedelta = timedelta(minutes=30)
    HOUR: timedelta = timedelta(hours=1)
    TWO_HOURS: timedelta = timedelta(hours=2)
    FOUR_HOURS: timedelta = timedelta(hours=4)
    EIGHT_HOURS: timedelta = timedelta(hours=8)
    TWELVE_HOURS: timedelta = timedelta(hours=12)
    DAY: timedelta = timedelta(days=1)
    THREE_DAYS: timedelta = timedelta(days=3)
    WEEK: timedelta = timedelta(days=7)
    TWO_WEEKS: timedelta = timedelta(days=14)


if __name__ == "__main__":
    bounds: Bounds = Bounds.for_days(
        start_inclusive=date(2025, 1, 1),
        end_exclusive=date(2025, 2, 1),
    )

    sub_bounds: List[Bounds] = bounds.generate_overlapping_bounds(step=timedelta(days=3), interval=timedelta(days=3))
    print(sub_bounds)
