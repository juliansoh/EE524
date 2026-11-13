$ErrorActionPreference = 'Stop'
$culture = [Globalization.CultureInfo]::InvariantCulture
$rows = foreach ($file in Get-ChildItem "$PSScriptRoot\*.raw.csv") {
    $raw = Import-Csv $file.FullName | Where-Object { $_.'Kernel Name' }
    if (@($raw).Count -ne 1) { throw "Expected one launch in $($file.Name)" }
    $details = Import-Csv ($file.FullName -replace '\.raw\.csv$', '.details.csv')
    function Metric([string]$name) {
        $value = $raw.$name
        if ([string]::IsNullOrWhiteSpace($value)) { throw "Missing metric $name" }
        [double]::Parse($value, [Globalization.NumberStyles]::Float -bor [Globalization.NumberStyles]::AllowThousands, $culture)
    }
    function Detail([string]$name) {
        $value = ($details | Where-Object { $_.'Metric Name' -eq $name } | Select-Object -First 1).'Metric Value'
        if ([string]::IsNullOrWhiteSpace($value)) { throw "Missing detail $name" }
        [double]::Parse($value, [Globalization.NumberStyles]::Float -bor [Globalization.NumberStyles]::AllowThousands, $culture)
    }
    $name = $file.BaseName -replace '\.raw$', ''
    $k = if ($name -match '-k(\d+)') { [int]$Matches[1] } else { 100 }
    $j = if ($k -eq 2500) { 2000 } else { $k }
    $l = if ($k -eq 2500) { 3000 } else { $k }
    $variant = if ($name -match 'tile(\d+)') { "tile$($Matches[1])" } elseif ($name -match 'Serial') { 'serial-label-naive' } else { 'naive' }
    $seconds = (Metric 'gpu__time_duration.sum') / 1e9
    $fpRate = (Metric 'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed') +
        (Metric 'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed') +
        (Metric 'derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2')
    $fpPerSecond = $fpRate * (Metric 'smsp__cycles_elapsed.avg.per_second')
    $bytesPerSecond = Metric 'dram__bytes.sum.per_second'
    $peakFP = (Metric 'derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2') * (Metric 'sm__cycles_elapsed.avg.per_second')
    $peakBW = (Metric 'dram__bytes.sum.peak_sustained') * (Metric 'dram__cycles_elapsed.avg.per_second')
    [pscustomobject]@{
        Report = $name
        FMA = if ($name -like 'NoFFMA-*') { 'disabled' } else { 'enabled' }
        K = $k
        Variant = $variant
        Duration_ms = $seconds * 1000
        Useful_GFLOPs = (2.0 * $j * $k * $l) / $seconds / 1e9
        HW_GFLOPs = $fpPerSecond / 1e9
        HW_AI = $fpPerSecond / $bytesPerSecond
        DRAM_bytes = (Metric 'dram__bytes_read.sum') + (Metric 'dram__bytes_write.sum')
        DRAM_GBs = $bytesPerSecond / 1e9
        FP32_peak_GFLOPs = $peakFP / 1e9
        DRAM_peak_GBs = $peakBW / 1e9
        Ridge_AI = $peakFP / $peakBW
        FP32_peak_pct = $fpPerSecond / $peakFP * 100
        Compute_pct = Detail 'Compute (SM) Throughput'
        DRAM_pct = Detail 'DRAM Throughput'
        L1_pct = Detail 'L1/TEX Cache Throughput'
        Occupancy_pct = Detail 'Achieved Occupancy'
        Theoretical_occupancy_pct = Detail 'Theoretical Occupancy'
        Registers = Detail 'Registers Per Thread'
        Shared_bytes = Detail 'Static Shared Memory Per Block'
        Block_threads = Detail 'Block Size'
        Grid_blocks = Detail 'Grid Size'
        Waves_per_SM = Detail 'Waves Per SM'
        FFMA_per_cycle = Metric 'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed'
        FADD_per_cycle = Metric 'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed'
        FMUL_per_cycle = Metric 'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed'
    }
}
$rows | Export-Csv "$PSScriptRoot\summary.csv" -NoTypeInformation
$rows | Select-Object K, Variant,
    @{n='ms';e={'{0:F6}' -f $_.Duration_ms}},
    @{n='HW_GF/s';e={'{0:F2}' -f $_.HW_GFLOPs}},
    @{n='AI';e={'{0:F3}' -f $_.HW_AI}},
    Compute_pct, Occupancy_pct, Registers, Shared_bytes, Waves_per_SM |
    Format-Table -AutoSize
