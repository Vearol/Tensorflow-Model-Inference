@echo off

pushd vendors\zlib
nmake -f win32\Makefile.msc
popd
