@echo off
setlocal
cd /d "%~dp0"

for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
if not defined VSINSTALL (
    echo Could not locate Visual Studio C++ Build Tools.
    exit /b 1
)

set "VULKAN_INCLUDE="
if exist "external\vulkan\include\vulkan\vulkan.h" set "VULKAN_INCLUDE=/Iexternal\vulkan\include"
if not defined VULKAN_INCLUDE if defined VULKAN_SDK if exist "%VULKAN_SDK%\Include\vulkan\vulkan.h" set "VULKAN_INCLUDE=/I"%VULKAN_SDK%\Include""

if not defined VULKAN_INCLUDE (
    echo Vulkan headers not found.
    echo Install the Vulkan SDK or clone Vulkan-Headers to external\vulkan.
    exit /b 1
)

call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%

if not exist build mkdir build
rc /nologo /fo build\version.res src\version.rc
if errorlevel 1 exit /b %errorlevel%

cl /nologo /LD /EHsc /O2 /MD /W3 /std:c++20 ^
  /Iexternal\reshade\include ^
  /Iexternal\ngx ^
  %VULKAN_INCLUDE% ^
  /Fobuild\ ^
  /Fdbuild\ ^
  src\dlss5-feed.cpp ^
  /link ^
  /OUT:build\dlss5-feed.addon64 ^
  build\version.res ^
  external\ngx\libs\nvsdk_ngx_d.lib ^
  kernel32.lib ^
  user32.lib ^
  advapi32.lib ^
  ole32.lib

exit /b %errorlevel%
