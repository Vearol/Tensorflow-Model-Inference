@echo off

cd zlib
if %platform%==x86 x86.bat
if %platform%==X64 x64.bat
