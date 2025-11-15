@echo off
cd /d "%~dp0"

git push --set-upstream origin master
git add -A
git branch -M main
git commit -m "Auto-commit"
git push -u origin main
pause