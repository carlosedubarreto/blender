@echo off
setlocal EnableDelayedExpansion

:: -------------------------------------------------------------
:: Blender Fast Incremental Build Script (VS2022 + Ninja)
:: Supports: full (default) and lite
:: Usage:
::   fast_build.bat            (incremental full build)
::   fast_build.bat run        (full build + launch)
::   fast_build.bat lite       (incremental lite build)
::   fast_build.bat lite run   (lite build + launch)
:: -------------------------------------------------------------

:: Ensure MSVC 64-bit build environment and standard C++ headers are loaded
if not defined DevEnvDir (
    if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    )
)

:: Set OptiX SDK path for Cycles
if exist "C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0" (
    set "OPTIX_ROOT_DIR=C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0"
)

:: 1. Clear conflicting compiler overrides
set CMAKE_C_COMPILER_LAUNCHER=
set CMAKE_CXX_COMPILER_LAUNCHER=
set _CL_=


:: 2. Parse build flavor (default to full)
set "FLAVOR=full"
set "LAUNCH=0"

for %%A in (%*) do (
    if /i "%%A"=="lite" set "FLAVOR=lite"
    if /i "%%A"=="full" set "FLAVOR=full"
    if /i "%%A"=="run"  set "LAUNCH=1"
)

:: 3. Set directory paths
set "SOURCE_DIR=%~dp0"
set "BLENDER_LIB_DIR=%SOURCE_DIR%..\lib\windows_x64"

if /i "%FLAVOR%"=="lite" (
    set "BUILD_DIR=%SOURCE_DIR%..\build_windows_Lite_x64_vc17_Release"
    set "MAKE_TARGET=lite ninja nobuild"
) else (
    set "BUILD_DIR=%SOURCE_DIR%..\build_windows_x64_vc17_Release"
    set "MAKE_TARGET=ninja nobuild"
)

:: 4. Ensure Ninja is in PATH
where ninja >nul 2>nul
if %ERRORLEVEL% neq 0 (
    if exist "%BLENDER_LIB_DIR%\ninja\ninja.exe" (
        set "PATH=%BLENDER_LIB_DIR%\ninja;%PATH%"
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" (
        set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
    )
)

where ninja >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] ninja.exe could not be found in PATH, Blender lib, or Visual Studio.
    exit /b 1
)

:: 5. Generate build folder via make.bat if not initialized yet
if not exist "%BUILD_DIR%\build.ninja" (
    echo [INFO] Generating Ninja project for %FLAVOR% build...
    call "%SOURCE_DIR%make.bat" 2022 %MAKE_TARGET%
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] CMake configuration failed.
        exit /b %ERRORLEVEL%
    )
)

:: 6. Run the incremental build
cd /d "%BUILD_DIR%"
echo [INFO] Running Ninja build (%FLAVOR%)...
ninja

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Build failed! Check compiler errors above.
    exit /b %ERRORLEVEL%
)

echo [SUCCESS] %FLAVOR% build completed successfully.
echo Output binary: %BUILD_DIR%\bin\blender.exe

:: 7. Launch if requested
if "%LAUNCH%"=="1" (
    echo [INFO] Launching Blender...
    start "" "%BUILD_DIR%\bin\blender.exe"
)

endlocal