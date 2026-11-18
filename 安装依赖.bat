@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在安装 GeoScope 演示版所需组件...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo 安装失败。请确认已安装 Python，并检查网络连接。
  pause
  exit /b 1
)
echo.
echo 安装完成。接下来请双击“启动Demo.bat”。
pause
