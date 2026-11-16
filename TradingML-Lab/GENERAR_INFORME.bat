@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo  TradingML-Lab - Generando informe de eficiencia...
echo ============================================================
.venv\Scripts\python.exe src\informe.py
for /f "delims=" %%d in ('dir /b /ad /o-n reports\informe_*') do (
  start "" "reports\%%d\informe.html"
  goto :listo
)
:listo
pause
