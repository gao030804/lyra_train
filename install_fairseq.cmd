@echo off
call "E:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
set DISTUTILS_USE_SDK=1
set MSSdk=1
set MAX_JOBS=1
set USERPROFILE=C:\Users\12540
set HOME=C:\Users\12540
set PIP_CACHE_DIR=E:\lyra\.pip-cache
where cl
where link
E:\lyra\.venv\Scripts\python.exe -m pip install E:\lyra\.vendor\fairseq-0.12.2 --no-build-isolation --timeout 1000 -v
