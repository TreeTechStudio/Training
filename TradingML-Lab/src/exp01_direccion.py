"""Experimento 01: ¿sube o baja el precio en los próximos H minutos?

Es exactamente la pregunta de una operación de Olymp ("Arriba"/"Abajo").

Regla de oro del negocio: si aciertas cobras el pago P (p. ej. 85 %) y si
fallas pierdes el 100 %. Para no perder dinero a la larga necesitas acertar:
    acierto_minimo = 1 / (1 + P)   ->   con P = 0.85 son 54.05 %
Por eso no basta con "más de 50 %": hay que superar ese número, fuera de muestra.

Uso: python src/exp01_direccion.py 5 0.85    (horizonte en minutos, pago)
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SEMILLA = 42
UMBRAL_CONFIANZA = 0.55  # declarado ANTES de ver resultados (no se ajusta)


def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """Solo usa información disponible al cierre de cada vela (sin mirar el futuro)."""
    c = df["close"]
    f = pd.DataFrame(index=df.index)
    for n in (1, 3, 5, 15, 30, 60):
        f[f"ret_{n}"] = c.pct_change(n)                       # cuánto se movió
    f["rango"] = (df["high"] - df["low"]) / c                 # tamaño de la vela
    f["cuerpo"] = (df["close"] - df["open"]) / c              # vela verde/roja
    r1 = c.pct_change()
    f["vol_15"] = r1.rolling(15).std()                        # nerviosismo reciente
    f["vol_60"] = r1.rolling(60).std()
    f["dist_ma20"] = c / c.rolling(20).mean() - 1             # lejos de su promedio
    f["dist_ma60"] = c / c.rolling(60).mean() - 1
    sube = r1.clip(lower=0).rolling(14).mean()
    baja = (-r1.clip(upper=0)).rolling(14).mean()
    f["rsi"] = 100 - 100 / (1 + sube / (baja + 1e-12))       # fuerza relativa
    v = df["volume"]
    f["vol_z"] = (v - v.rolling(60).mean()) / (v.rolling(60).std() + 1e-12)
    hora = df["time"].dt.hour + df["time"].dt.minute / 60
    f["hora_sin"] = np.sin(2 * np.pi * hora / 24)            # sesión: Asia/Londres/NY
    f["hora_cos"] = np.cos(2 * np.pi * hora / 24)
    f["dia_semana"] = df["time"].dt.dayofweek
    return f


def intervalo(acc: float, n: int) -> tuple[float, float]:
    """Intervalo de confianza 95 % del porcentaje de acierto."""
    e = 1.96 * np.sqrt(acc * (1 - acc) / max(n, 1))
    return acc - e, acc + e


def main(h: int = 5, pago: float = 0.85) -> None:
    df = pd.read_csv(ROOT / "data" / "raw" / "EURUSD_M1.csv", parse_dates=["time"])
    X = crear_features(df)
    futuro = df["close"].shift(-h)
    y = (futuro > df["close"]).astype(int)
    ok = X.notna().all(axis=1) & futuro.notna() & (futuro != df["close"])  # empate = devolución
    X, y = X[ok].reset_index(drop=True), y[ok].reset_index(drop=True)
    minimo = 1 / (1 + pago)
    print(f"Muestras: {len(X):,} | horizonte {h} min | pago {pago:.0%} -> acierto mínimo {minimo:.2%}")
    print(f"Proporción de 'sube' en los datos: {y.mean():.2%}\n")

    modelos = {
        "Regresión logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
        "Gradient Boosting (árboles)": HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=200, random_state=SEMILLA),
    }
    # Walk-forward: entreno con el pasado, pruebo con el bloque siguiente, avanzo.
    tscv = TimeSeriesSplit(n_splits=5, gap=h)
    filas = []
    for nombre, m in [("Línea base: repetir última vela", None)] + list(modelos.items()):
        for k, (tr, te) in enumerate(tscv.split(X)):
            if m is None:
                pred = (X["ret_1"].iloc[te] > 0).astype(int).to_numpy()
                prob = np.full(len(te), 0.5)
            else:
                m.fit(X.iloc[tr], y.iloc[tr])
                prob = m.predict_proba(X.iloc[te])[:, 1]
                pred = (prob > 0.5).astype(int)
            real = y.iloc[te].to_numpy()
            acc = (pred == real).mean()
            seguro = np.abs(prob - 0.5) >= (UMBRAL_CONFIANZA - 0.5)
            acc_s = (pred[seguro] == real[seguro]).mean() if seguro.sum() else np.nan
            filas.append({"modelo": nombre, "bloque": k + 1, "n": len(te), "acierto": acc,
                          "n_seguras": int(seguro.sum()), "acierto_seguras": acc_s})
            print(f"{nombre:34s} bloque {k+1}: acierto {acc:.2%} | "
                  f"con confianza≥{UMBRAL_CONFIANZA:.0%}: {acc_s:.2%} en {seguro.sum():,} casos")
    r = pd.DataFrame(filas)
    resumen = r.groupby("modelo", sort=False).agg(
        acierto=("acierto", "mean"), n=("n", "sum"),
        acierto_seguras=("acierto_seguras", "mean"), n_seguras=("n_seguras", "sum"))
    resumen["IC95"] = [f"{a:.2%}–{b:.2%}" for a, b in (intervalo(x, n) for x, n in zip(resumen.acierto, resumen.n))]
    resumen["supera_minimo"] = resumen["acierto"] > minimo
    print("\n=== RESUMEN (promedio de los 5 bloques fuera de muestra) ===")
    print(resumen.to_string(formatters={"acierto": "{:.2%}".format, "acierto_seguras": "{:.2%}".format}))

    rep = ROOT / "reports"
    sello = datetime.now().strftime("%Y-%m-%d %H:%M")
    resumen.assign(fecha=sello, horizonte=h, pago=pago, minimo=minimo).to_csv(
        rep / "experimentos.csv", mode="a", header=not (rep / "experimentos.csv").exists())
    r.to_csv(rep / f"exp01_h{h}_bloques.csv", index=False)
    if "Gradient Boosting (árboles)" in modelos:  # qué variables usó más el árbol
        from sklearn.inspection import permutation_importance
        m = modelos["Gradient Boosting (árboles)"]
        tr, te = list(tscv.split(X))[-1]
        imp = permutation_importance(m, X.iloc[te][:20000], y.iloc[te][:20000],
                                     n_repeats=3, random_state=SEMILLA, n_jobs=-1)
        orden = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False)
        print("\nVariables más útiles (cuánto empeora el acierto si se desordenan):")
        print((orden.head(8) * 100).round(3).astype(str).add(" pts").to_string())


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 5, float(a[1]) if len(a) > 1 else 0.85)
