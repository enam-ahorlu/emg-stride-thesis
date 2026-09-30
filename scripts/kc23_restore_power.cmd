@echo off
rem Restores the ONE power setting changed for the KC23 queue (25 September 2026).
rem The Balanced scheme's "Sleep after" was 900 s on AC and 600 s on battery; it was set to 0 (never) so that
rem switching to Balanced could not put the machine to sleep while the queue has work.
rem Run this when the programme ends. See KC23_POWER_SETTINGS.md.
powercfg /setacvalueindex 381b4222-f694-41f0-9685-ff5bb260df2e SUB_SLEEP STANDBYIDLE 900
powercfg /setdcvalueindex 381b4222-f694-41f0-9685-ff5bb260df2e SUB_SLEEP STANDBYIDLE 600
echo Balanced scheme sleep timeouts restored: 15 min on AC, 10 min on battery.
