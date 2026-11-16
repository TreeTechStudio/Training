"""Genera el INFORME DE EFICIENCIA (HTML + tablas CSV) con los datos de Olymp.

Uso:  python src/informe.py                 (todos los datos en data/pantalla)
      python src/informe.py ruta_a_datos    (otra carpeta, p. ej. para pruebas)

Lee:  bitacora_<activo>_h<seg>_p<pago>_<fecha>.csv   y   ticks_<activo>_<fecha>.csv
Crea: reports/informe_<fecha_hora>/informe.html  (+ tablas .csv y gráficas .png)
"""
from __future__ import annotations

import base64
import html
import io
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import estadistica as est  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ZONA = "America/Mexico_City"
MODELOS = ["repetir", "contraria", "ia"]
NOMBRES = {"repetir": "Repetir movimiento", "contraria": "Llevar la contraria", "ia": "IA (aprende)"}
# Paleta validada (azul, naranja, aqua: pasan separación para daltónicos) + estilos de línea
# distintos, porque el aqua tiene poco contraste sobre blanco: no se depende solo del color.
COLOR = {"repetir": "#2a78d6", "contraria": "#eb6834", "ia": "#1baf7a"}
ESTILO = {"repetir": "-", "contraria": "--", "ia": ":"}
TINTA, TINTA2, FONDO, REJILLA = "#0b0b0b", "#52514e", "#fcfcfb", "#e3e2dd"

plt.rcParams.update({
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
    "axes.edgecolor": REJILLA, "axes.labelcolor": TINTA2, "xtick.color": TINTA2, "ytick.color": TINTA2,
    "text.color": TINTA, "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


# ------------------------------------------------------------------ carga
def cargar(carpeta: Path) -> dict:
    """Devuelve {(activo, h, pago): {'bit': DataFrame, 'ticks': DataFrame|None}}."""
    grupos: dict = {}
    for f in sorted(carpeta.glob("bitacora_*.csv")):
        m = re.match(r"bitacora_(?P<a>.+)_h(?P<h>\d+)_p(?P<p>\d+)_(?P<d>\d{8})\.csv$", f.name)
        if not m:
            continue
        df = pd.read_csv(f, parse_dates=["hora_utc"])
        if df.empty:
            continue
        df["archivo"] = f.name
        k = (m["a"], int(m["h"]), int(m["p"]) / 100)
        grupos.setdefault(k, {"bit": [], "ticks": []})["bit"].append(df)
    for (a, _, _), g in grupos.items():
        for f in sorted(carpeta.glob(f"ticks_{a}_*.csv")):
            t = pd.read_csv(f, header=None, names=["hora_utc", "precio"], parse_dates=["hora_utc"])
            g["ticks"].append(t)
    out = {}
    for k, g in grupos.items():
        bit = pd.concat(g["bit"]).sort_values("hora_utc").reset_index(drop=True)
        ticks = pd.concat(g["ticks"]).sort_values("hora_utc").reset_index(drop=True) if g["ticks"] else None
        out[k] = {"bit": bit, "ticks": ticks}
    return out


# ------------------------------------------------------------------ cálculos
def tabla_resumen(bit: pd.DataFrame, h: int, pago: float) -> tuple[pd.DataFrame, int]:
    minimo = est.acierto_minimo(pago)
    resueltas = bit[bit["resultado"].isin(["sube", "baja"])].copy()
    empates = int((bit["resultado"] == "empate").sum())
    resueltas["real"] = (resueltas["resultado"] == "sube").astype(int)
    filas = []
    for m in MODELOS:
        d = resueltas[resueltas[m].notna()]
        ac = int((d[m] == d["real"]).sum())
        r_min = est.prueba_acierto(ac, len(d), h, minimo)
        r_50 = est.prueba_acierto(ac, len(d), h, 0.5)
        acc = r_min["acierto"]
        filas.append({
            "modelo": NOMBRES[m], "predicciones": len(d), "independientes≈": round(r_min["n_ind"]),
            "acierto": acc, "IC95_ingenuo": r_min["ic_ing"], "IC95_real": r_min["ic_ind"],
            "p_vs_50": r_50["p"], "p_vs_minimo": r_min["p"],
            "ganancia_por_1": est.valor_esperado(acc, pago) if len(d) else np.nan,
            "n_ind": r_min["n_ind"], "modelo_id": m,
        })
    t = pd.DataFrame(filas)
    t["p_ajustado"] = (t["p_vs_minimo"] * len(MODELOS)).clip(upper=1.0)  # Bonferroni: 3 modelos
    t["conclusion"] = [est.veredicto(r.p_ajustado, r.acierto, minimo, r.n_ind) for r in t.itertuples()]
    return t, empates


def por_bloques(bit: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    d = bit[bit["resultado"].isin(["sube", "baja"])].copy()
    d["real"] = (d["resultado"] == "sube").astype(int)
    d["bloque"] = pd.qcut(np.arange(len(d)), q=min(k, max(len(d) // 50, 1)), labels=False, duplicates="drop") + 1
    filas = []
    for b, g in d.groupby("bloque"):
        fila = {"bloque": int(b), "desde": g["hora_utc"].min().tz_convert(ZONA).strftime("%d/%m %H:%M"),
                "predicciones": len(g)}
        for m in MODELOS:
            x = g[g[m].notna()]
            fila[NOMBRES[m]] = (x[m] == x["real"]).mean() if len(x) else np.nan
        filas.append(fila)
    return pd.DataFrame(filas)


def por_hora(bit: pd.DataFrame) -> pd.DataFrame:
    d = bit[bit["resultado"].isin(["sube", "baja"])].copy()
    d["real"] = (d["resultado"] == "sube").astype(int)
    d["hora"] = d["hora_utc"].dt.tz_convert(ZONA).dt.hour
    filas = []
    for hh, g in d.groupby("hora"):
        fila = {"hora_local": f"{hh:02d}:00", "predicciones": len(g)}
        for m in MODELOS:
            x = g[g[m].notna()]
            fila[NOMBRES[m]] = (x[m] == x["real"]).mean() if len(x) else np.nan
        filas.append(fila)
    return pd.DataFrame(filas)


def por_confianza(bit: pd.DataFrame, h: int, minimo: float) -> pd.DataFrame:
    d = bit[bit["resultado"].isin(["sube", "baja"]) & bit["prob_ia"].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    d["real"] = (d["resultado"] == "sube").astype(int)
    d["conf"] = np.maximum(d["prob_ia"], 1 - d["prob_ia"])
    cortes = [0.5, 0.52, 0.55, 0.60, 1.01]
    nombres = ["50–52 %", "52–55 %", "55–60 %", "más de 60 %"]
    d["nivel"] = pd.cut(d["conf"], cortes, labels=nombres, right=False)
    filas = []
    for n in nombres:
        g = d[d["nivel"] == n]
        ac = int((g["ia"] == g["real"]).sum())
        r = est.prueba_acierto(ac, len(g), h, minimo)
        filas.append({"confianza_IA": n, "predicciones": len(g), "independientes≈": round(r["n_ind"]),
                      "acierto": r["acierto"], "IC95_real": r["ic_ind"]})
    return pd.DataFrame(filas)


def analisis_ticks(t: pd.DataFrame | None) -> dict | None:
    if t is None or len(t) < 100:
        return None
    seg = (t["hora_utc"] - t["hora_utc"].iloc[0]).dt.total_seconds().to_numpy()
    dt = np.diff(seg)
    r = np.diff(np.log(t["precio"].to_numpy()))
    ok = (dt > 0) & (dt < 3)            # solo retornos entre lecturas consecutivas (sin huecos)
    r = r[ok]
    r = r[np.isfinite(r)]
    ac = est.autocorrelacion(r, 10)
    return {
        "ticks": len(t), "horas": seg[-1] / 3600, "huecos": int((dt >= 3).sum()),
        "sin_cambio": float((r == 0).mean()) if len(r) else np.nan,
        "vol_1s": float(r.std()) if len(r) else np.nan,
        "asimetria": float(pd.Series(r).skew()), "curtosis": float(pd.Series(r).kurt()),
        "acf": ac, "banda": 1.96 / np.sqrt(max(len(r), 1)), "n_ret": len(r),
        "persistencia": float((np.sign(r[1:]) == np.sign(r[:-1]))[(r[1:] != 0) & (r[:-1] != 0)].mean())
        if len(r) > 2 else np.nan,
    }


def curva_capital(bit: pd.DataFrame, h: int, pago: float) -> dict:
    d = bit[bit["resultado"].isin(["sube", "baja"])].copy()
    d["real"] = (d["resultado"] == "sube").astype(int)
    out = {}
    for m in MODELOS:
        x = est.no_solapadas(d[d[m].notna()], h)
        gan = np.where(x[m].to_numpy() == x["real"].to_numpy(), pago, -1.0)
        out[m] = {"t": x["hora_utc"].dt.tz_convert(ZONA).to_numpy(), "curva": np.cumsum(gan)}
    return out


# ------------------------------------------------------------------ gráficas
def a_base64(fig) -> str:
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def etiquetar_finales(ax, pares, sep=0.07):
    """Etiqueta directa al final de cada línea; las separa si quedarían una sobre otra.

    pares: lista de (nombre, x_final, y_final). Llamar cuando los ejes ya tengan sus límites.
    """
    y0, y1 = ax.get_ylim()
    x0, x1 = ax.get_xlim()
    minsep = sep * (y1 - y0)
    prev = None
    for nombre, x, y in sorted(pares, key=lambda p: p[2]):
        ya = y if prev is None else max(y, prev + minsep)
        ax.text(x + 0.012 * (x1 - x0), ya, nombre, va="center", ha="left", color=TINTA2, fontsize=9,
                clip_on=False)
        prev = ya


def leyenda_con_referencias(ax, referencias):
    """Leyenda de las series + las líneas de referencia (en vez de texto sobre el gráfico)."""
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    for texto, estilo in referencias:
        handles.append(Line2D([0], [0], color=TINTA2, lw=1, ls=estilo))
        labels.append(texto)
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0, -0.2), ncol=3)


def graf_ventana(bit, h, minimo):
    d = bit[bit["resultado"].isin(["sube", "baja"])].copy()
    d["real"] = (d["resultado"] == "sube").astype(int)
    ventana = int(max(100, min(len(d) // 10, 2000)))
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    finales = []
    for m in MODELOS:
        x = d[d[m].notna()]
        if len(x) < ventana:
            continue
        s = (x[m] == x["real"]).astype(float).rolling(ventana).mean()
        ax.plot(range(len(x)), s, color=COLOR[m], ls=ESTILO[m], lw=2, label=NOMBRES[m])
        finales.append((NOMBRES[m], len(x) - 1, float(s.iloc[-1])))
    ax.axhline(0.5, color=TINTA2, lw=1)
    ax.axhline(minimo, color=TINTA2, lw=1, ls=(0, (6, 3)))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel(f"Predicciones (ventana móvil de {ventana})")
    ax.set_ylabel("Acierto")
    ax.set_title("Acierto a lo largo del tiempo")
    etiquetar_finales(ax, finales)
    leyenda_con_referencias(ax, [("50 % (azar)", "-"), (f"{minimo:.2%} (mínimo para no perder)", (0, (6, 3)))])
    return a_base64(fig)


def graf_capital(curvas, pago):
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    finales = []
    for m in MODELOS:
        c = curvas[m]
        if len(c["curva"]) < 2:
            continue
        ax.plot(range(len(c["curva"])), c["curva"], color=COLOR[m], ls=ESTILO[m], lw=2, label=NOMBRES[m])
        finales.append((NOMBRES[m], len(c["curva"]) - 1, float(c["curva"][-1])))
    if not finales:
        plt.close(fig)
        return None
    ax.axhline(0, color=TINTA2, lw=1)
    ax.set_xlabel("Operaciones sin solaparse (una a la vez)")
    ax.set_ylabel("Ganancia acumulada (en importes apostados)")
    ax.set_title(f"Si hubieras apostado 1 en cada señal (pago {pago:.0%})")
    etiquetar_finales(ax, finales)
    leyenda_con_referencias(ax, [("0 (ni ganas ni pierdes)", "-")])
    return a_base64(fig)


def graf_confianza(conf, minimo):
    if conf.empty or conf["predicciones"].sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    x = np.arange(len(conf))
    y = conf["acierto"].to_numpy(dtype=float)
    lo = np.array([c[0] for c in conf["IC95_real"]], dtype=float)
    hi = np.array([c[1] for c in conf["IC95_real"]], dtype=float)
    n_ind = conf["independientes≈"].to_numpy(dtype=float)
    ok = n_ind >= 30          # con menos muestras independientes, una barra engaña: se omite
    ax.bar(x[ok], np.nan_to_num(y[ok]), width=0.55, color=COLOR["ia"], zorder=3)
    ax.errorbar(x[ok], y[ok], yerr=[(y - lo)[ok], (hi - y)[ok]], fmt="none", ecolor=TINTA2, capsize=4, lw=1.2,
                zorder=4)
    ax.axhline(0.5, color=TINTA2, lw=1)
    ax.axhline(minimo, color=TINTA2, lw=1, ls=(0, (6, 3)))
    for i, n in enumerate(conf["predicciones"]):
        if ok[i]:
            ax.text(i, 0.31, f"n={n:,}", ha="center", color=FONDO, fontsize=8, zorder=5)
        else:
            ax.text(i, 0.40, f"pocos datos\n(n={n:,})", ha="center", va="center", color=TINTA2, fontsize=8)
    leyenda_con_referencias(ax, [("50 % (azar)", "-"), (f"{minimo:.2%} (mínimo para no perder)", (0, (6, 3)))])
    ax.set_xticks(x, conf["confianza_IA"])
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_ylim(0.3, 0.8)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Qué tan segura estaba la IA")
    ax.set_ylabel("Acierto")
    ax.set_title("¿Acierta más cuando está más segura?")
    return a_base64(fig)


def graf_acf(tk):
    if tk is None:
        return None
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    lags = np.arange(1, len(tk["acf"]) + 1)
    ax.axhspan(-tk["banda"], tk["banda"], color=REJILLA, alpha=0.8, lw=0, zorder=0)
    ax.bar(lags, tk["acf"], width=0.55, color=COLOR["repetir"], zorder=3)
    ax.axhline(0, color=TINTA2, lw=1)
    ax.set_xticks(lags)
    ax.set_xlabel("Retraso (segundos)")
    ax.set_ylabel("Correlación")
    ax.set_title("¿Un movimiento anticipa el siguiente? (zona gris = azar)")
    ax.grid(axis="x", visible=False)
    return a_base64(fig)


# ------------------------------------------------------------------ HTML
def pct(v, d=2):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.{d}f} %"


def pval(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else ("&lt;0.001" if v < 0.001 else f"{v:.3f}")


def tabla_html(filas: list[list[str]], cab: list[str]) -> str:
    th = "".join(f"<th>{html.escape(c)}</th>" for c in cab)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in f) + "</tr>" for f in filas)
    return f"<div class='tabla'><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>"


def seccion_activo(activo, h, pago, d, carpeta_out: Path) -> tuple[str, dict]:
    bit, ticks = d["bit"], d["ticks"]
    minimo = est.acierto_minimo(pago)
    t, empates = tabla_resumen(bit, h, pago)
    t.drop(columns=["modelo_id"]).to_csv(carpeta_out / f"resumen_{activo}.csv", index=False)
    ini, fin = bit["hora_utc"].min().tz_convert(ZONA), bit["hora_utc"].max().tz_convert(ZONA)
    horas = (fin - ini).total_seconds() / 3600
    filas = []
    for r in t.itertuples():
        ic = f"{pct(r.IC95_real[0], 1)} – {pct(r.IC95_real[1], 1)}"
        icg = f"{pct(r.IC95_ingenuo[0], 1)} – {pct(r.IC95_ingenuo[1], 1)}"
        sig = "ok" if (r.p_ajustado < 0.05 and r.acierto > minimo and r.n_ind >= 300) else ""
        filas.append([f"<b>{r.modelo}</b>", f"{r.predicciones:,}", f"{r.n_ind:,.0f}", pct(r.acierto), ic, icg,
                      pval(r.p_vs_50), pval(r.p_ajustado), f"{r.ganancia_por_1:+.3f}" if r.predicciones else "—",
                      f"<span class='{sig}'>{html.escape(r.conclusion)}</span>"])
    cab = ["Modelo", "Predicciones", "Independientes≈", "Acierto", "IC95 (real)", "IC95 (ingenuo, optimista)",
           "p vs 50 %", f"p vs {minimo:.1%} (ajustado)", "Ganancia por 1 apostado", "Qué se puede concluir"]
    s = [f"<section><h2>{html.escape(activo)} <small>· operaciones de {h} s · pago {pago:.0%} · "
         f"acierto mínimo {minimo:.2%}</small></h2>",
         f"<p class='meta'>Del {ini:%d/%m/%Y %H:%M} al {fin:%d/%m/%Y %H:%M} (hora local) · "
         f"{horas:.1f} h de datos · {empates} empates excluidos (Olymp devuelve el importe).</p>"]
    if t["n_ind"].max() < 300:
        s.append("<p class='aviso'>Pocos datos: con menos de ≈300 muestras independientes, ni un acierto alto "
                 "permite concluir nada. Deja correr más horas.</p>")
    s.append("<h3>Resumen por modelo</h3>" + tabla_html(filas, cab))
    s.append("<p class='nota'><b>Cómo leerlo:</b> las predicciones se hacen cada segundo pero cada operación dura "
             f"{h} s, así que se solapan: solo ≈ n/{h} son independientes. Por eso el IC95 <i>real</i> es más ancho "
             "que el ingenuo. El valor <i>p</i> es la probabilidad de ver un acierto así <u>por pura suerte</u>; "
             "se ajusta ×3 por haber probado 3 modelos (corrección de Bonferroni). Ganancia por 1: lo que ganas "
             "en promedio por cada unidad apostada (negativa = pérdida).</p>")
    g1 = graf_ventana(bit, h, minimo)
    s.append(f"<h3>Acierto en el tiempo</h3><img alt='Acierto móvil por modelo' src='data:image/png;base64,{g1}'>")
    g2 = graf_capital(curva_capital(bit, h, pago), pago)
    if g2:
        s.append(f"<h3>Curva de capital</h3><img alt='Ganancia acumulada' src='data:image/png;base64,{g2}'>"
                 "<p class='nota'>Solo cuenta operaciones que no se solapan (una abierta a la vez), "
                 "como podrías hacer en la vida real, sin reinversión.</p>")
    cap = curva_capital(bit, h, pago)
    filas_c = [[NOMBRES[m], f"{len(cap[m]['curva']):,}",
                f"{cap[m]['curva'][-1]:+.1f}" if len(cap[m]["curva"]) else "—",
                f"{est.max_drawdown(cap[m]['curva']):.1f}"] for m in MODELOS]
    s.append(tabla_html(filas_c, ["Modelo", "Operaciones", "Resultado final", "Peor caída (drawdown)"]))
    b = por_bloques(bit)
    b.to_csv(carpeta_out / f"bloques_{activo}.csv", index=False)
    if len(b) > 1:
        fb = [[str(r.bloque), r.desde, f"{r.predicciones:,}"] + [pct(getattr(r, c.replace(' ', '_').replace('(', '').replace(')', ''), np.nan), 1)
                                                                  for c in [NOMBRES[m] for m in MODELOS]]
              for r in b.rename(columns=lambda c: c.replace(" ", "_").replace("(", "").replace(")", "")).itertuples()]
        s.append("<h3>¿Es estable? Acierto por bloques de tiempo</h3>"
                 + tabla_html(fb, ["Bloque", "Desde", "Predicciones"] + [NOMBRES[m] for m in MODELOS])
                 + "<p class='nota'>Un modelo verdadero mantiene su acierto en todos los bloques. "
                   "Si solo brilla en uno, suele ser suerte.</p>")
    c = por_confianza(bit, h, minimo)
    if not c.empty:
        c.to_csv(carpeta_out / f"confianza_{activo}.csv", index=False)
        g3 = graf_confianza(c, minimo)
        fc = [[r.confianza_IA, f"{r.predicciones:,}", f"{r[3]:,.0f}",
               pct(r.acierto, 1), f"{pct(r.IC95_real[0], 1)} – {pct(r.IC95_real[1], 1)}"] for r in c.itertuples()]
        s.append("<h3>IA: ¿la confianza sirve?</h3>"
                 + (f"<img alt='Acierto por nivel de confianza' src='data:image/png;base64,{g3}'>" if g3 else "")
                 + tabla_html(fc, ["Confianza de la IA", "Predicciones", "Independientes≈", "Acierto", "IC95 (real)"]))
    hh = por_hora(bit)
    hh.to_csv(carpeta_out / f"horas_{activo}.csv", index=False)
    if len(hh) > 1:
        fh = [[r.hora_local, f"{r.predicciones:,}"] + [pct(getattr(r, k), 1) for k in
                                                       [NOMBRES[m].replace(' ', '_').replace('(', '').replace(')', '') for m in MODELOS]]
              for r in hh.rename(columns=lambda c: c.replace(" ", "_").replace("(", "").replace(")", "")).itertuples()]
        s.append("<h3>Acierto por hora del día (hora local)</h3>"
                 + tabla_html(fh, ["Hora", "Predicciones"] + [NOMBRES[m] for m in MODELOS])
                 + "<p class='nota'>Ojo: cada hora tiene pocos datos; mirar muchas horas y quedarse con la mejor "
                   "es una forma clásica de engañarse.</p>")
    tk = analisis_ticks(ticks)
    if tk:
        g4 = graf_acf(tk)
        s.append("<h3>El precio por dentro (ticks de 1 s)</h3>"
                 + tabla_html([[f"{tk['ticks']:,}", f"{tk['horas']:.1f} h", f"{tk['huecos']}", pct(tk["sin_cambio"], 1),
                                f"{tk['vol_1s'] * 1e4:.3f}", f"{tk['asimetria']:.2f}", f"{tk['curtosis']:.1f}",
                                pct(tk["persistencia"], 1)]],
                               ["Lecturas", "Duración", "Huecos (≥3 s)", "Sin cambio entre lecturas",
                                "Volatilidad 1 s (pb)", "Asimetría", "Curtosis", "Misma dirección que el anterior"])
                 + (f"<img alt='Autocorrelación' src='data:image/png;base64,{g4}'>" if g4 else "")
                 + "<p class='nota'>Si ninguna barra sale de la zona gris, los movimientos no se anticipan "
                   "unos a otros y ningún modelo basado solo en precios pasados tiene de dónde aprender. "
                   "Barras negativas fuera de la zona sugieren rebote; positivas, inercia.</p>")
    s.append("</section>")
    return "\n".join(s), {"activo": activo, "h": h, "pago": pago, "horas": horas, "n": len(bit),
                          "mejor": t.sort_values("acierto", ascending=False).iloc[0]}


CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e3e2dd;--acc:#2a78d6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,Segoe UI,sans-serif}
main{max-width:1060px;margin:0 auto;padding:32px 24px 64px}h1{font-size:26px;margin:0 0 4px}
h2{font-size:21px;margin:40px 0 4px;border-top:2px solid var(--ink);padding-top:16px}h2 small{font-weight:400;color:var(--ink2);font-size:14px}
h3{font-size:16px;margin:26px 0 8px}.meta,.nota{color:var(--ink2);font-size:13px;margin:6px 0 10px}
.aviso{background:#fff4d6;border-left:4px solid #eda100;padding:10px 14px;margin:12px 0}
img{max-width:100%;height:auto;display:block;margin:8px 0}
.tabla{overflow-x:auto;margin:8px 0}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border-bottom:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}
th{font-weight:600;color:var(--ink2);background:#f4f3ef}td .ok{font-weight:700}
.lim{background:#f4f3ef;padding:14px 18px;margin-top:36px;font-size:13px}.lim li{margin:4px 0}
@media print{main{padding:0}img{break-inside:avoid}}
"""

LIMITES = """
<div class='lim'><b>Límites de este estudio (para tu sección de discusión)</b><ul>
<li><b>Cuenta demo y operaciones en papel:</b> no hay latencia de ejecución, deslizamiento ni emociones.</li>
<li><b>Lectura por pantalla:</b> a ≈1 lectura por segundo, con posibles errores de OCR y huecos; el precio que ves puede
diferir de la liquidación real de Olymp.</li>
<li><b>Activos sintéticos (p. ej. Quickler):</b> los genera el broker; sus conclusiones no se trasladan a mercados reales.</li>
<li><b>Muestras solapadas:</b> corregidas con n/h, pero es una aproximación; un bootstrap por bloques sería más fino.</li>
<li><b>Pruebas múltiples:</b> 3 modelos con ajuste de Bonferroni; si pruebas más variantes, el umbral debe endurecerse.</li>
<li><b>Resultado pasado ≠ futuro.</b> Un acierto significativo exige replicarse en días distintos antes de creerlo.</li>
</ul></div>"""


def main(carpeta: Path | None = None) -> Path:
    carpeta = carpeta or (ROOT / "data" / "pantalla")
    datos = cargar(carpeta)
    sello = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "reports" / f"informe_{sello}"
    out.mkdir(parents=True, exist_ok=True)
    if not datos:
        (out / "informe.html").write_text(
            f"<meta charset='utf-8'><body style='font:16px system-ui;padding:40px'><h1>Sin datos todavía</h1>"
            f"<p>No encontré archivos bitacora_*.csv en {html.escape(str(carpeta))}. Ejecuta primero "
            "INICIAR_OLYMP_VIVO.bat y déjalo correr un buen rato.</p></body>", encoding="utf-8")
        print("Sin datos:", out / "informe.html")
        return out / "informe.html"
    secciones, resumenes = [], []
    for (activo, h, pago), d in sorted(datos.items()):
        s, r = seccion_activo(activo, h, pago, d, out)
        secciones.append(s)
        resumenes.append(r)
    comp = ""
    if len(resumenes) > 1:
        fc = [[html.escape(r["activo"]), f"{r['h']} s", f"{r['pago']:.0%}", f"{r['horas']:.1f} h", f"{r['n']:,}",
               html.escape(r["mejor"]["modelo"]), pct(r["mejor"]["acierto"]), pct(est.acierto_minimo(r["pago"]))]
              for r in resumenes]
        comp = ("<h2>Comparación entre activos</h2>"
                + tabla_html(fc, ["Activo", "Duración", "Pago", "Datos", "Predicciones", "Mejor modelo",
                                  "Su acierto", "Mínimo para no perder"])
                + "<p class='nota'>Ojo: «mejor modelo» elegido a posteriori infla el resultado; "
                  "usa las secciones de abajo para juzgar con significancia.</p>")
    doc = (f"<!doctype html><html lang='es'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
           f"<title>Informe de eficiencia</title><style>{CSS}</style><main>"
           f"<h1>Informe de eficiencia · Olymp Trade (cuenta demo)</h1>"
           f"<p class='meta'>Generado el {datetime.now():%d/%m/%Y %H:%M} · TradingML-Lab</p>"
           f"{comp}{''.join(secciones)}{LIMITES}</main></html>")
    (out / "informe.html").write_text(doc, encoding="utf-8")
    print("Informe listo:", out / "informe.html")
    return out / "informe.html"


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
