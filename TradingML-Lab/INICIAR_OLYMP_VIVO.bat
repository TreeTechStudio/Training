@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo  TradingML-Lab - Prediccion en vivo sobre Olymp (solo observa)
echo.
echo  ANTES DE EMPEZAR:
echo   1. Olymp MAXIMIZADO en la pantalla PRINCIPAL (la de la laptop).
echo   2. Tu trabajo, en el segundo monitor. Nada encima de Olymp.
echo   3. Laptop conectada a la corriente y sin suspension.
echo   Ctrl+C para parar. Luego: GENERAR_INFORME.bat
echo ============================================================
set /p ACT=Nombre del activo que muestra Olymp (ej. Quickler, EURUSD) [Quickler]: 
if "%ACT%"=="" set ACT=Quickler
set /p DUR=Duracion de la operacion en segundos [5]: 
if "%DUR%"=="" set DUR=5
set /p PAGO=Pago del activo (0.85 = 85%%) [0.85]: 
if "%PAGO%"=="" set PAGO=0.85
.venv\Scripts\python.exe src\olymp_vivo.py %DUR% %PAGO% %ACT%
pause
