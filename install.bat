@echo off
REM SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
REM Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
REM
REM Install this pack's dependencies into ComfyUI's own environment.
REM
REM   install.bat                         - everything, into the detected interpreter
REM   install.bat --dry-run               - print the commands, change nothing
REM   install.bat --groups sfm            - just the rig solve
REM   install.bat --python C:\ComfyUI\python_embeded\python.exe
REM
REM All this wrapper does is find ComfyUI's python; scripts\install.py does the
REM work and takes the flags. Getting the interpreter right is the point: every
REM package goes into that environment, so the wrong one installs a working set
REM of dependencies somewhere ComfyUI will never look.
REM
REM The trainer group builds CUDA extensions from source and needs matching MSVC
REM Build Tools; run it from a Developer Command Prompt. The other three groups
REM are wheels and have no such requirement.

setlocal enabledelayedexpansion

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

REM --python is consumed here; everything else is forwarded untouched.
set "PYTHON="
set "ARGS="
:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--python" (
    if "%~2"=="" (
        echo install.bat: --python needs a path>&2
        exit /b 2
    )
    set "PYTHON=%~2"
    shift
    shift
    goto parse
)
set "ARGS=!ARGS! %1"
shift
goto parse
:parsed

REM Preference order, most explicit first. An activated venv or conda env is
REM trusted over bare `python`, which may be an unrelated system install.
if not defined PYTHON if defined COMFYUI_PYTHON set "PYTHON=%COMFYUI_PYTHON%"
if not defined PYTHON if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" set "PYTHON=%VIRTUAL_ENV%\Scripts\python.exe"
if not defined PYTHON if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PYTHON=%CONDA_PREFIX%\python.exe"

REM A portable ComfyUI keeps its interpreter beside the ComfyUI directory; this
REM pack lives in ComfyUI\custom_nodes\comfyui-cumuli, so look two levels up.
if not defined PYTHON if exist "%ROOT%\..\..\..\python_embeded\python.exe" set "PYTHON=%ROOT%\..\..\..\python_embeded\python.exe"
if not defined PYTHON if exist "%ROOT%\..\..\venv\Scripts\python.exe" set "PYTHON=%ROOT%\..\..\venv\Scripts\python.exe"

if not defined PYTHON (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        if not defined PYTHON set "PYTHON=%%P"
    )
)

if not defined PYTHON (
    echo install.bat: could not find a python interpreter.>&2
    echo.>&2
    echo Point it at ComfyUI's own interpreter, which is the only one that matters here:>&2
    echo.>&2
    echo     install.bat --python C:\ComfyUI\python_embeded\python.exe>&2
    echo     install.bat --python C:\ComfyUI\venv\Scripts\python.exe>&2
    echo.>&2
    echo or set COMFYUI_PYTHON, or activate the environment first.>&2
    exit /b 1
)

if not exist "%PYTHON%" (
    where "%PYTHON%" >nul 2>&1
    if errorlevel 1 (
        echo install.bat: interpreter not found: %PYTHON%>&2
        exit /b 1
    )
)

"%PYTHON%" "%ROOT%\scripts\install.py"!ARGS!
set "RC=%errorlevel%"

REM Double-clicked from Explorer there is no parent console, so the window would
REM close the instant this returns -- taking the SMPL-X instructions and any
REM error with it. cmdcmdline carries /c only when it was launched that way.
echo %cmdcmdline% | find /i "/c" >nul
if not errorlevel 1 (
    echo.
    pause
)
exit /b %RC%
