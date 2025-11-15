@echo off
cd /d "%~dp0"
git add -A
git commit -m "Auto-commit"
git push
pause