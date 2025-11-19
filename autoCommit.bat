
@echo off
cd /d "%~dp0"
git branch -M main
git add -A
git commit -m "Auto-commit"
git push --set-upstream origin main --force
pause
