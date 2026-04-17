import datetime


def iter_months(start: datetime, end: datetime):

    if start > end:
        raise ValueError("start_date must be <= end_date")

    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1


if __name__ == "__main__":
    for y, m in iter_months(
        datetime.datetime(2023, 5, 1), datetime.datetime(2024, 2, 1)
    ):
        print(y, m)
