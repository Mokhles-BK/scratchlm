$avail = (Get-Counter '\Memory\Available MBytes' -ErrorAction SilentlyContinue).CounterSamples.CookedValue
Write-Host ("Available RAM: {0:N1} MB" -f $avail)
Write-Host ""
Write-Host "Top 15 processes by working set:"
Get-Process | Sort-Object WorkingSet -Descending | Select-Object -First 15 |
    ForEach-Object {
        "{0,8:N1} MB  {1}" -f ($_.WorkingSet / 1MB), $_.ProcessName
    }