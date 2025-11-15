
@echo off
cd /d "%~dp0"
git branch -M master
git add -A
git commit -m "Auto-commit"
git push --set-upstream origin master
pause
