@echo off
chcp 65001 >nul
cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
  echo 未找到 Python。请先安装 Python 3.10 或更高版本。
  pause
  exit /b 1
)

python -c "from PIL import Image" >nul 2>&1
if errorlevel 1 (
  echo 尚未安装运行组件。请先双击“安装依赖.bat”。
  pause
  exit /b 1
)

echo GeoScope v0.4.0 正在启动...
echo 浏览器将打开 http://127.0.0.1:4180/?v=0.4.0
start "" "http://127.0.0.1:4180/?v=0.4.0"
python server.py
pause
