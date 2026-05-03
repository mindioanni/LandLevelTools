
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
from models import ParsedPos


def _parse_epoch(date_text: str, time_text: str):
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(f"{date_text} {time_text}", fmt)
        except Exception:
            pass
    return None


def parse_pos(path: str | Path) -> ParsedPos:
    path = Path(path).expanduser().resolve()
    header = {}
    rows = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue

            if line.startswith("%"):
                text = line[1:].strip()
                if ":" in text:
                    key, val = text.split(":", 1)
                    header[key.strip()] = val.strip()
                continue

            parts = line.split()
            if len(parts) < 7:
                continue

            epoch = _parse_epoch(parts[0], parts[1])
            if epoch is None:
                continue

            def fval(i):
                try:
                    return float(parts[i])
                except Exception:
                    return None

            def ival(i):
                try:
                    return int(parts[i])
                except Exception:
                    return None

            rows.append({
                "time": epoch,
                "X": fval(2),
                "Y": fval(3),
                "Z": fval(4),
                "Q": ival(5),
                "ns": ival(6),
                "sdx": fval(7) if len(parts) > 7 else None,
                "sdy": fval(8) if len(parts) > 8 else None,
                "sdz": fval(9) if len(parts) > 9 else None,
                "sdxy": fval(10) if len(parts) > 10 else None,
                "sdyz": fval(11) if len(parts) > 11 else None,
                "sdzx": fval(12) if len(parts) > 12 else None,
                "age": fval(13) if len(parts) > 13 else None,
                "ratio": fval(14) if len(parts) > 14 else None,
            })

    df = pd.DataFrame(rows)
    return ParsedPos(path=path, header=header, dataframe=df)
