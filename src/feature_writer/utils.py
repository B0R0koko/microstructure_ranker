import polars as pl


# Used to rename columns after pivoting
def flatten(col: str) -> str:
    # if it looks like {"a","b"}, strip the braces/quotes and join with _
    if col.startswith('{') and col.endswith('}'):
        inner = col.strip('{}')  # '"asset_hold_time","500MS"'
        parts = [p.strip('"') for p in inner.split(',')]
        return "_".join(parts)
    else:
        return col


def to_wide_format(df: pl.DataFrame) -> pl.DataFrame:
    """Converts data from long format to wide format"""
    return (
        df
        .pivot(values="value", index=["sampled_time", "currency_pair"], on=["feature", "window"])
        .rename(flatten)
    )
