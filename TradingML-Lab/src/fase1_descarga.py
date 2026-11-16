"""Fase 1: descarga velas de MT5 a CSV y muestra un resumen.

Uso:  python src/fase1_descarga.py EURUSD M5 20000
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import mt5_connector as mc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main(symbol: str = "EURUSD", tf: str = "M5", n: int = 20_000) -> None:
    with mc.session():
        print("Terminal:", mc.terminal_summary())
        print("Ejemplos de símbolos:", mc.list_symbols("*USD*")[:15])
        df = mc.get_candles(symbol, tf, n)
    out = mc.save_csv(df, symbol, tf, ROOT / "data" / "raw")
    print(f"\n{len(df)} velas guardadas en {out}")
    print(df.head(3).to_string())
    print(df.tail(3).to_string())
    print("\nRango:", df["time"].min(), "->", df["time"].max())
    print("Valores nulos:", int(df.isna().sum().sum()))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if len(a) > 0 else "EURUSD", a[1] if len(a) > 1 else "M5", int(a[2]) if len(a) > 2 else 20_000)
