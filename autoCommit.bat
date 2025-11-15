@echo off
cd /d "%~dp0"
git pull https://github.com/Drybones3363/TradeGamba.git
git push --set-upstream origin main
git add -A
git commit -m "Auto-commit"
pause