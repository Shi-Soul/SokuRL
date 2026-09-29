"""Decode exported resource values without changing the native frame format."""
from math import isfinite


def spirit_fraction(value, maximum):
    if type(value) is not int or not 0 <= value <= 0xFFFF:
        raise ValueError(f"invalid exported spirit word: {value}")
    if not isfinite(maximum) or maximum <= 0:
        raise ValueError(f"invalid spirit maximum: {maximum}")
    # currentSpirit is signed short; bridge ABI 7 exports its unsigned word.
    signed = value - 0x10000 if value & 0x8000 else value
    if signed > maximum:
        raise ValueError(f"spirit exceeds maximum: value={signed}, maximum={maximum}")
    return max(0, signed) / maximum
