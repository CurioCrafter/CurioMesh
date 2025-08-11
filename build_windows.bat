@echo off
setlocal

set BUILD_DIR=%~dp0core_build
cmake -S "%~dp0core" -B "%BUILD_DIR%" -A x64 -DPROJECT_NAME=curio_core
cmake --build "%BUILD_DIR%" --config Release

echo.
echo Build finished. Copy the produced curio_core.*.pyd into this folder if not already placed by CMake.
endlocal


