# KC23 power settings changed (25 September 2026)

Machine: Acer laptop, Windows 11, classic S3 standby (no Modern Standby). Two power schemes exist.

## What was already true (nothing changed)

The ACTIVE scheme, "Acer" (6b1d62b8-4045-4ef0-8836-50a12fbee8b4), already had **Sleep after = 0 (never)** on both
AC and battery. Its "Hibernate after" (12 h AC, 3 h battery) only counts once the machine has slept, so with sleep
disabled it never fires. Nothing was changed on this scheme.

## What was changed (exactly one setting, on the inactive scheme)

| Scheme | Setting | Before | After |
|---|---|---|---|
| Balanced (381b4222-f694-41f0-9685-ff5bb260df2e) | Sleep after, on AC | 900 s (15 min) | 0 (never) |
| Balanced (381b4222-f694-41f0-9685-ff5bb260df2e) | Sleep after, on battery | 600 s (10 min) | 0 (never) |

Reason: if the scheme is switched to Balanced (by Acer's software, by Windows, or by hand) the machine would sleep
after 10 to 15 minutes idle and pause every job.

Commands used:

    powercfg /setacvalueindex 381b4222-f694-41f0-9685-ff5bb260df2e SUB_SLEEP STANDBYIDLE 0
    powercfg /setdcvalueindex 381b4222-f694-41f0-9685-ff5bb260df2e SUB_SLEEP STANDBYIDLE 0

## To revert when the programme ends

Run `kc23_restore_power.cmd` (it sets 900 s on AC and 600 s on battery), or by hand:

    powercfg /setacvalueindex 381b4222-f694-41f0-9685-ff5bb260df2e SUB_SLEEP STANDBYIDLE 900
    powercfg /setdcvalueindex 381b4222-f694-41f0-9685-ff5bb260df2e SUB_SLEEP STANDBYIDLE 600

## Not a power-plan setting, and not persistent

`kc23_queue.py` also asks Windows to keep the system awake (SetThreadExecutionState, ES_SYSTEM_REQUIRED) for as long
as the runner is alive with work. It holds under any scheme, is released when the runner exits or crashes, and needs
no reverting. The display can still turn off.

## What still puts the laptop to sleep

Closing the lid, and the Start menu Sleep command. Neither is changed. Keep the lid open (or docked with the lid
action set to Do nothing) while the queue runs. Keep it on AC power for long runs.
