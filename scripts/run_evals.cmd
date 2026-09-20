@echo off
rem Run the F0 then F1 evaluations, resuming either from its saved rows.
rem Launched detached so it survives the terminal or agent session that started it:
rem   powershell -c "Start-Process -WindowStyle Minimized scripts\run_evals.cmd"
cd /d "%~dp0.."
uv run --no-sync python scripts\evaluate.py --arm p3-f0-base --model p3-f0-base >> results\p3-f0-base.log 2>&1
echo F0_EXIT %ERRORLEVEL% >> results\p3-f0-base.log
uv run --no-sync python scripts\evaluate.py --arm p3-f1-qlora --model p3-f1-qlora >> results\p3-f1-qlora.log 2>&1
echo F1_EXIT %ERRORLEVEL% >> results\p3-f1-qlora.log
