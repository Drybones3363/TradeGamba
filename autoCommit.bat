@echo off
cd /d "%~dp0"

git init
git remote add origin https://github.com/drybones3363/TradeGamba.git
git add .
git commit -m "Initial commit"
git push -u origin main


git add -A
git commit -m "Auto-commit"
git push
pause