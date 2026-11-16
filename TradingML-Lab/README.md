# TradingML-Lab

Laboratorio educativo de ML aplicado a trading con MetaTrader 5. Solo lectura de datos, sin órdenes.

## Entorno (Python 3.12; el paquete MetaTrader5 no soporta 3.15)
```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Primer uso
1. Abre MetaTrader 5 e inicia sesión en una cuenta DEMO (esquina inferior derecha debe mostrar datos de conexión, no "sin conexión").
2. `python src/fase1_descarga.py EURUSD M5 20000`

Plan completo: `docs/PLAN_MAESTRO.md`.
