@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 观域 GeoScope 正在启动
echo 请打开 http://127.0.0.1:4180
python server.py
pause
