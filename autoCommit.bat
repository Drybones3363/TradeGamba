@echo off
cd /d "%~dp0"

git push --set-upstream origin main
git add -A
git commit -m "Auto-commit"
pause