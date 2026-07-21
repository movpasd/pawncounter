"""
Separate utility module for general-purpose functions which are otherwise unrelated to
the project
"""

import polars as pl


def white_mask() -> pl.Expr:
    return pl.lit(0xFFFFFFFFFFFFFFFF, dtype=pl.UInt128)


def black_mask() -> pl.Expr:
    return pl.lit(0xFFFFFFFFFFFFFFFF << 64, dtype=pl.UInt128)


def rank_mask(rank: int) -> pl.Expr:
    """
    u128 bitmask of the rank with given index (0-indexed)
    """
    white_mask = 0x0101010101010101 << rank
    return pl.lit(white_mask | (white_mask << 64), dtype=pl.UInt128)


def file_mask(file: int) -> pl.Expr:
    """
    u128 bitmask of the file with given index (0-indexed)
    """
    white_mask = 0xFF << file * 8
    return pl.lit(white_mask | (white_mask << 64), dtype=pl.UInt128)
