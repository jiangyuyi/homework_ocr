@echo off
chcp 65001 >nul 2>&1
setlocal EnableExtensions
title 作业手写提取 - 正在启动

rem ============================================================
rem  作业手写提取 - 一键启动（Windows）
rem
rem  双击本文件即可；第一次运行会自动下载识别模型。
rem  启动失败时不要关窗口：原因会同时写进同目录的
rem  启动错误.log，把那个文件发给协助你的人即可。
rem
rem  测试用开关：设置环境变量 HOMEWORK_OCR_DRYRUN=1 后，
rem  只做环境探测、打印将要执行的命令，然后停住等你按键。
rem ============================================================

cd /d "%~dp0"
if errorlevel 1 goto :fail_cd

set "LOG=%~dp0启动错误.log"
set "PY="
set "RC="
set "SAW_STORE_ALIAS="
set "BROKEN_VENV="
set "REASON="
set "FIX1="
set "FIX2="
set "FIX3="

rem ---- 1/5 找 Python：先试项目自带的 .venv ----
if not exist ".venv\Scripts\python.exe" goto :scan_wellknown
".venv\Scripts\python.exe" -c "import sys" >nul 2>&1
if errorlevel 1 goto :broken_venv
set "PY=.venv\Scripts\python.exe"
goto :have_python

:broken_venv
set "BROKEN_VENV=1"
set "PY="

rem ---- 2/5 再扫安装器的固定位置（不依赖 PATH）----
rem 这一步是为什么重装系统之后还能双击就用：
rem Python 装完会落在固定目录下，但**已经开着的终端**里 PATH 还是旧的，
rem 这时 python / py 全都找不到。直接按绝对路径找就绕开了这个问题。
:scan_wellknown
if defined PY goto :have_python
for %%V in (314 313 312 311 310) do if not defined PY call :probe_abs "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
for %%V in (314 313 312 311 310) do if not defined PY call :probe_abs "%ProgramFiles%\Python%%V\python.exe"
call :probe_abs "%LOCALAPPDATA%\Programs\Python\Launcher\py.exe"
call :probe_abs "%WINDIR%\py.exe"
if defined PY goto :have_python

rem ---- 3/5 最后才扫 PATH 上的 python / py / python3 ----
:scan_path
call :try_python python
if defined PY goto :have_python
call :try_python py
if defined PY goto :have_python
call :try_python python3
if defined PY goto :have_python
goto :no_python

:have_python
echo.
echo   使用 Python：%PY%
if defined BROKEN_VENV echo   提示：项目自带的 .venv 无法运行，已改用系统 Python。
echo.

rem ---- 4/5 首次运行先下载模型 ----
if exist "models" goto :run_gui
echo   首次运行：需要先下载识别模型（要联网，大约几分钟）。
echo.
if defined HOMEWORK_OCR_DRYRUN goto :dryrun_fetch
"%PY%" -m homework_ocr fetch-models --out models
set "RC=%errorlevel%"
if not "%RC%"=="0" goto :fail_fetch
echo.
echo   模型已就绪。
echo.

rem ---- 5/5 启动图形界面 ----
:run_gui
if defined HOMEWORK_OCR_DRYRUN goto :dryrun_gui
"%PY%" -m homework_ocr gui %*
set "RC=%errorlevel%"
if not "%RC%"=="0" goto :fail_gui
echo.
echo   已退出。
echo.
pause
endlocal & exit /b 0

rem ============================================================
rem  失败分支：一律打印中文原因 + 写日志 + 停住等按键
rem ============================================================

:fail_cd
set "REASON=无法进入程序所在目录。"
set "FIX1=请确认 启动.bat 还在原来的文件夹里，并且该磁盘或共享位置已连接。"
goto :fatal

:no_python
set "REASON=找不到可用的 Python。"
set "FIX1=请到 https://www.python.org/downloads/ 安装 Python 3.10 或更高版本。"
set "FIX2=安装时务必勾选 Add Python to PATH。"
if defined SAW_STORE_ALIAS set "FIX3=检测到 Microsoft 应用商店的 python 占位别名，它不是真正的 Python，已自动跳过。"
goto :fatal

:fail_fetch
set "REASON=模型下载失败，退出码 %RC%。"
set "FIX1=请检查网络，然后重新双击本文件。"
set "FIX2=如果所在网络需要代理，请先在浏览器里配好代理再试。"
goto :fatal

:fail_gui
set "REASON=程序启动失败，退出码 %RC%。"
set "FIX1=请把上面的错误信息原样发给协助你的人。"
set "FIX2=如果提示 No module named homework_ocr，说明这个 Python 环境里还没装项目。"
goto :fatal

:fatal
echo.
echo   ==============================================
echo    启动失败
echo   ==============================================
echo.
echo   原因：%REASON%
if defined FIX1 echo   %FIX1%
if defined FIX2 echo   %FIX2%
if defined FIX3 echo   %FIX3%
echo.
echo   完整日志：%LOG%
echo   把这个日志文件发给协助你的人就能定位问题。
echo.
>>"%LOG%" echo.
call :log "-------- %DATE% %TIME% --------"
call :log "工作目录：%CD%"
if defined PY call :log "使用的 Python：%PY%"
call :log "原因：%REASON%"
if defined FIX1 call :log "%FIX1%"
if defined FIX2 call :log "%FIX2%"
if defined FIX3 call :log "%FIX3%"
echo.
pause
endlocal & exit /b 1

rem ============================================================
rem  测试模式
rem ============================================================

:dryrun_fetch
echo   测试模式：将执行  %PY% -m homework_ocr fetch-models --out models
goto :run_gui

:dryrun_gui
echo   测试模式：将执行  %PY% -m homework_ocr gui %*
echo.
echo   测试模式：探测结束，没有真正启动界面。
echo.
pause
endlocal & exit /b 0

rem ============================================================
rem  子程序
rem ============================================================

rem 把一行文字追加到日志文件
:log
>>"%LOG%" echo %~1
goto :eof

rem 探测一个按绝对路径给出的 Python；同样跳过商店占位别名
:probe_abs
if defined PY goto :eof
if not exist "%~1" goto :eof
echo "%~1" | find /i "WindowsApps" >nul 2>&1
if errorlevel 1 goto :probe_abs_real
set "SAW_STORE_ALIAS=1"
goto :eof
:probe_abs_real
"%~1" -c "import sys" >nul 2>&1
if errorlevel 1 goto :eof
set "PY=%~1"
goto :eof

rem 探测一个 Python 命令；where 可能返回多个结果，逐个试，
rem 并跳过 Microsoft 应用商店的假别名（它不是真正的 Python）
:try_python
if defined PY goto :eof
for /f "delims=" %%I in ('where %~1 2^>nul') do call :probe_candidate "%%I"
goto :eof

:probe_candidate
if defined PY goto :eof
echo "%~1" | find /i "WindowsApps" >nul 2>&1
if errorlevel 1 goto :probe_real
set "SAW_STORE_ALIAS=1"
goto :eof
:probe_real
"%~1" -c "import sys" >nul 2>&1
if errorlevel 1 goto :eof
set "PY=%~1"
goto :eof
