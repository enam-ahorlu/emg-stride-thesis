# scripts/kc23_queue_task.ps1
# Registers, inspects and removes the Task Scheduler entry that keeps src/kc23_queue.py running unattended
# (added 25 September 2026). Usage, from the repository root:
#   powershell -ExecutionPolicy Bypass -File scripts/kc23_queue_task.ps1 -Action Register
#   powershell -ExecutionPolicy Bypass -File scripts/kc23_queue_task.ps1 -Action Status
#   powershell -ExecutionPolicy Bypass -File scripts/kc23_queue_task.ps1 -Action Unregister
#
# What the task does
#   - Runs `.venv\Scripts\pythonw.exe -u src/kc23_queue.py --redirect-output` from this folder (26 September 2026: no console at all,
#     so there is no window to close; output goes to logs\kc23\queue_stdout.log and queue_stderr.log; Ctrl+C and Ctrl+Break
#     are ignored). It used to run python.exe under cmd.exe, which Task Scheduler hosted in a Windows Terminal window; closing that
#     window ended the runner with exit code 0xC000013A and killed its running job. Jobs now start in their own hidden console and
#     process group (kc23_queue.spawn_detached).
#   - Triggers: at logon of the current user, and every 5 minutes indefinitely. The runner takes a lock file
#     (logs\kc23\queue.lock), so an extra launch while a runner is alive exits at once with code 3 and touches
#     nothing. If the runner has crashed or been killed (an app restart, a closed terminal), the next launch takes the
#     stale lock, ADOPTS any job still alive from running.json, and re-queues a dead one, which finishes through
#     --resume and expected_outputs.
#   - Runs as the current user, only while logged on (interactive token), so the GPU is available. After a reboot it
#     starts at the next logon. It does not run while logged off (that needs a stored password and a session in which
#     CUDA is not guaranteed).
#   - Not stopped on battery, not stopped when the machine goes idle, no execution time limit, at most one instance.
#   - The task lives outside the Claude app's process tree, so closing or restarting the app does not stop it or the
#     jobs it launches. (Jobs launched BEFORE the task took over stay in the old tree until they finish.)
#
# What it never does: retry a job that failed (queue_state.json remembers it; use `src/kc23_queue.py --retry JOB_ID`), or
# release a halted stage (use `src/kc23_queue.py --clear-halt STAGE` after a decision).
param(
    [Parameter(Mandatory = $true)][ValidateSet("Register", "Unregister", "Status", "Start")][string]$Action
)

$TaskName = "KC23Queue"
$Here = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)   # repository root (this script is in scripts/)
$Py = Join-Path $Here ".venv\Scripts\python.exe"
$Pyw = Join-Path $Here ".venv\Scripts\pythonw.exe"
$User = "$env:USERDOMAIN\$env:USERNAME"

function Get-TaskXml {
    # -u: unbuffered; --redirect-output: the runner opens its own log files (pythonw has no stdout)
    $args1 = '-u src/kc23_queue.py --redirect-output'
    $start = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss")
    @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Keeps the KC23 experiment queue (src/kc23_queue.py) running. One runner at a time (lock file); adopts live jobs; never retries a failed job.</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <UserId>$User</UserId>
    </LogonTrigger>
    <TimeTrigger>
      <StartBoundary>$start</StartBoundary>
      <Enabled>true</Enabled>
      <Repetition>
        <Interval>PT5M</Interval>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
    </TimeTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>$User</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>true</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>6</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>$([System.Security.SecurityElement]::Escape($Pyw))</Command>
      <Arguments>$([System.Security.SecurityElement]::Escape($args1))</Arguments>
      <WorkingDirectory>$Here</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"@
}

switch ($Action) {
    "Register" {
        if (-not (Test-Path $Pyw)) { throw "pythonw not found at $Pyw" }
        New-Item -ItemType Directory -Force -Path (Join-Path $Here "logs\kc23") | Out-Null
        Register-ScheduledTask -TaskName $TaskName -Xml (Get-TaskXml) -Force | Out-Null
        Write-Host "Registered task $TaskName for $User."
        Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo | Format-List TaskName, LastRunTime, NextRunTime, LastTaskResult
    }
    "Unregister" {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed task $TaskName. A runner that is already running is NOT stopped (it holds queue.lock)."
    }
    "Status" {
        Get-ScheduledTask -TaskName $TaskName | Format-List TaskName, State
        Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo | Format-List LastRunTime, NextRunTime, LastTaskResult
        $lock = Join-Path $Here "logs\kc23\queue.lock"
        if (Test-Path $lock) { Write-Host "queue.lock:"; Get-Content $lock } else { Write-Host "no queue.lock (no runner)" }
    }
    "Start" {
        Start-ScheduledTask -TaskName $TaskName
        Write-Host "Started $TaskName (a second runner exits at once if one is already alive)."
    }
}
