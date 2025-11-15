
@echo off
cd /d "%~dp0"
git push --set-upstream origin master
git add -A
git branch -M master
git commit -m "Auto-commit"
pause
