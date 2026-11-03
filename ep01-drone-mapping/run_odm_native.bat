@echo off
rem GeoAI Lab - Episode 1: run native Windows ODM 3.6.2 with absolute paths (cmd current-dir lookup is disabled on this machine)
setlocal
call "E:\GeoAI-Lab-Tutorials\tools\ODM\win32env.bat"
python -X utf8 "E:\GeoAI-Lab-Tutorials\tools\ODM\run.py" --project-path "E:\GeoAI-Lab-Tutorials\data\drone" aukerman --dsm --dtm --orthophoto-resolution 5 --dem-resolution 10 --pc-quality medium --feature-quality high --max-concurrency 10
endlocal
