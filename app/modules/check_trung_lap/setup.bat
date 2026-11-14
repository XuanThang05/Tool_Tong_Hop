@echo off
chcp 65001 >nul
cd /d "%~dp0"

where py >nul 2>nul
if %ERRORLEVEL%==0 (
  set PY=py -3
) else (
  set PY=python
)

echo [1/4] Tao moi truong .venv ...
%PY% -m venv .venv
if errorlevel 1 (
  echo Khong tao duoc venv. Cai Python 3.11+ roi chay lai.
  pause
  exit /b 1
)

call .venv\Scripts\activate.bat
python -m pip install -U pip wheel
echo [2/4] Cai PyTorch ...
nvidia-smi >nul 2>nul
if %ERRORLEVEL%==0 (
  echo Phat hien NVIDIA GPU, cai torch CUDA 12.4
  pip install torch --index-url https://download.pytorch.org/whl/cu124
) else (
  echo Khong thay GPU, cai torch CPU (ban nhe)
  pip install torch
)
echo [3/4] Cai thu vien app ...
pip install -r requirements.txt pytest
echo [4/4] Test nhe ...
python -m pytest -q
if errorlevel 1 (
  echo Test loi. Van co the thu run.bat, nhung hay bao lai loi nay.
  pause
  exit /b 1
)
echo.
echo Xong. Chay run.bat de mo app. Lan dau se tai model tieng Viet (~120MB).
pause
