@echo off
setlocal

set PROJECT=%~1

if "%PROJECT%"=="" (
    echo Usage: manage_projects.bat ^<project_name^>
    echo Available projects: weapon_detection, face_detection, project_3, project_4, project_5
    exit /b 1
)

:: Set base directory
set BASE_DIR=%cd%

:: Activate virtual environment if it exists
if exist "%BASE_DIR%\.venv\Scripts\activate.bat" (
    call "%BASE_DIR%\.venv\Scripts\activate.bat"
)

echo Requested project: %PROJECT%

:: Stop existing python instances to prevent port/camera conflicts
echo Stopping existing project instances...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq manage_projects*" /T >nul 2>&1
:: Note: The above taskkill might be too broad if other python scripts are running. 
:: A safer alternative if needed is to find the exact PID of app.py and kill it.

timeout /t 1 /nobreak >nul

if "%PROJECT%"=="weapon_detection" (
    echo Launching Weapon Detection Project...
    python app.py --mode weapons_only --host 0.0.0.0 --port 8000
    goto :eof
)

if "%PROJECT%"=="face_detection" (
    echo Launching Face Detection Project...
    python app.py --mode faces_only --host 0.0.0.0 --port 8000
    goto :eof
)

if "%PROJECT%"=="project_3" (
    echo Launching Project 3...
    echo Execution command for project 3 not implemented yet.
    goto :eof
)

if "%PROJECT%"=="project_4" (
    echo Launching Project 4...
    echo Execution command for project 4 not implemented yet.
    goto :eof
)

if "%PROJECT%"=="project_5" (
    echo Launching Project 5...
    echo Execution command for project 5 not implemented yet.
    goto :eof
)

echo Error: Unknown project '%PROJECT%'
echo Valid options: weapon_detection, face_detection, project_3, project_4, project_5
exit /b 1
