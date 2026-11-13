$ErrorActionPreference = 'Stop'
$culture = [Globalization.CultureInfo]::InvariantCulture
$root = Split-Path $PSScriptRoot -Parent
$data = @(Import-Csv "$PSScriptRoot\summary.csv")
$ordered = @()
foreach ($k in 100, 1000, 2500) {
    foreach ($variant in 'naive', 'tile8', 'tile12', 'tile16', 'tile32') {
        $ordered += $data | Where-Object { [int]$_.K -eq $k -and $_.Variant -eq $variant -and $_.FMA -eq 'enabled' }
    }
}
$ordered += $data | Where-Object Variant -eq 'serial-label-naive'
foreach ($k in 100, 1000, 2500) {
    foreach ($variant in 'naive', 'tile8', 'tile12', 'tile16', 'tile32') {
        $ordered += $data | Where-Object { [int]$_.K -eq $k -and $_.Variant -eq $variant -and $_.FMA -eq 'disabled' }
    }
}
if ($ordered.Count -ne 31) { throw 'Expected all 31 reports' }

function Num([double]$value) { $value.ToString('0.###', $culture) }
function X([double]$ai) { 90 + [Math]::Log10($ai) / 2 * 680 }
function Y([double]$rate) { 565 - ([Math]::Log10($rate) - 2) / 2 * 475 }
$svg = [Text.StringBuilder]::new()
[void]$svg.AppendLine('<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="780" viewBox="0 0 1100 780" role="img" aria-labelledby="title desc">')
[void]$svg.AppendLine('<title id="title">Combined MMULT FP32 Roofline, Tesla T4</title><desc id="desc">All 31 report results, including 15 FMA-disabled runs. Logarithmic arithmetic intensity and hardware performance axes. One common FMA-capable reference roof uses the original largest tile32 run. IDs correspond to the report tables.</desc>')
[void]$svg.AppendLine('<rect width="1100" height="780" fill="white"/><g font-family="Arial,sans-serif" fill="#172033">')
[void]$svg.AppendLine('<text x="90" y="32" font-size="24" font-weight="bold">MMULT: all recorded GPU results</text>')
[void]$svg.AppendLine('<text x="90" y="57" font-size="15">Tesla T4 | FP32 FLOPs include padded work | DRAM traffic denominator</text>')
foreach ($ai in 1, 2, 5, 10, 20, 50, 100) {
    $x = Num (X $ai)
    [void]$svg.AppendLine("<line x1=`"$x`" y1=`"90`" x2=`"$x`" y2=`"565`" stroke=`"#e0e5ec`"/><text x=`"$x`" y=`"590`" text-anchor=`"middle`" font-size=`"15`">$ai</text>")
}
foreach ($rate in 100, 200, 500, 1000, 2000, 5000, 10000) {
    $y = Num (Y $rate)
    [void]$svg.AppendLine("<line x1=`"90`" y1=`"$y`" x2=`"770`" y2=`"$y`" stroke=`"#e0e5ec`"/><text x=`"77`" y=`"$(Num ((Y $rate) + 5))`" text-anchor=`"end`" font-size=`"15`">$rate</text>")
}
$reference = $data | Where-Object { $_.K -eq '2500' -and $_.Variant -eq 'tile32' -and $_.FMA -eq 'enabled' }
$ridge = [double]$reference.Ridge_AI
$peak = [double]$reference.FP32_peak_GFLOPs
$bandwidth = [double]$reference.DRAM_peak_GBs
$roofPoints = "90,$(Num (Y $bandwidth)) $(Num (X $ridge)),$(Num (Y $peak)) 770,$(Num (Y $peak))"
[void]$svg.AppendLine("<polyline points=`"$roofPoints`" fill=`"none`" stroke=`"#172033`" stroke-width=`"3`"/>")
$rx = Num (X $ridge)
[void]$svg.AppendLine("<line x1=`"$rx`" y1=`"90`" x2=`"$rx`" y2=`"565`" stroke=`"#64748b`" stroke-dasharray=`"6 5`"/>")
[void]$svg.AppendLine('<text x="480" y="104" font-size="14">FP32 reference: 7817 GFLOP/s</text>')
[void]$svg.AppendLine('<text x="130" y="255" font-size="14" transform="rotate(-32 130 255)">DRAM reference: 313.28 GB/s</text>')
[void]$svg.AppendLine("<text x=`"$(Num ((X $ridge) + 8))`" y=`"548`" font-size=`"13`">ridge 24.953</text>")
$colors = @{ naive = '#444444'; tile8 = '#c23d32'; tile12 = '#986200'; tile16 = '#007f70'; tile32 = '#2259b5'; 'serial-label-naive' = '#444444' }
$offsets = @(
    @(-27,20), @(2,22), @(-14,-14), @(-6,-15), @(12,6),
    @(-24,17), @(12,10), @(-21,-14), @(10,-13), @(-25,-14),
    @(-27,15), @(-22,20), @(-23,-14), @(-26,-13), @(-24,-15), @(0,0)
)
for ($i = 0; $i -lt $ordered.Count; $i++) {
    $row = $ordered[$i]
    $x = X ([double]$row.HW_AI)
    $y = Y ([double]$row.HW_GFLOPs)
    $sx = Num $x
    $sy = Num $y
    $color = $colors[$row.Variant]
    $disabled = $row.FMA -eq 'disabled'
    $fill = if ($disabled) { 'none' } else { $color }
    $stroke = if ($disabled) { $color } else { 'white' }
    $id = '{0:D2}' -f ($i + 1)
    $tip = "$id $($row.Variant), k=$($row.K), FMA=$($row.FMA), AI=$(Num ([double]$row.HW_AI)), GFLOP/s=$(Num ([double]$row.HW_GFLOPs))"
    [void]$svg.AppendLine("<g><title>$tip</title>")
    if ($i -eq 15) {
        [void]$svg.AppendLine("<circle cx=`"$sx`" cy=`"$sy`" r=`"10`" fill=`"none`" stroke=`"$color`" stroke-width=`"1.5`"/>")
    } elseif ($row.K -eq '100') {
        $radius = if ($disabled) { 9 } else { 6 }
        [void]$svg.AppendLine("<circle cx=`"$sx`" cy=`"$sy`" r=`"$radius`" fill=`"$fill`" stroke=`"$stroke`" stroke-width=`"1.5`"/>")
    } elseif ($row.K -eq '1000') {
        $radius = if ($disabled) { 10 } else { 7 }
        $points = "$(Num $x),$(Num ($y-$radius)) $(Num ($x-$radius)),$(Num ($y+$radius-1)) $(Num ($x+$radius)),$(Num ($y+$radius-1))"
        [void]$svg.AppendLine("<polygon points=`"$points`" fill=`"$fill`" stroke=`"$stroke`" stroke-width=`"1.5`"/>")
    } else {
        $radius = if ($disabled) { 9 } else { 6 }
        $size = $radius * 2
        [void]$svg.AppendLine("<rect x=`"$(Num ($x-$radius))`" y=`"$(Num ($y-$radius))`" width=`"$size`" height=`"$size`" fill=`"$fill`" stroke=`"$stroke`" stroke-width=`"1.5`"/>")
    }
    if ($i -ne 15) {
        $label = if ($i -eq 0) { '01/16' } else { $id }
        $offset = if ($disabled) { @(9,27) } else { $offsets[$i] }
        [void]$svg.AppendLine("<text x=`"$(Num ($x+$offset[0]))`" y=`"$(Num ($y+$offset[1]))`" font-size=`"12`" font-weight=`"bold`">$label</text>")
    }
    [void]$svg.AppendLine('</g>')
    $legendY = 96 + $i * 18
    $case = switch ($row.K) { '100' { '100x100' } '1000' { '1000x1000' } '2500' { '2000x3000' } }
    $variantLabel = if ($i -eq 15) { 'Serial file / naive' } else { $row.Variant }
    $suffix = if ($disabled) { ' NoFMA' } else { '' }
    [void]$svg.AppendLine("<text x=`"805`" y=`"$legendY`" font-size=`"12`" fill=`"$color`">$id $case : $variantLabel$suffix</text>")
}
[void]$svg.AppendLine('<path d="M90 90V565H770" fill="none" stroke="#172033" stroke-width="2"/>')
[void]$svg.AppendLine('<text x="430" y="624" text-anchor="middle" font-size="18">Hardware arithmetic intensity (FLOP/byte), log scale</text>')
[void]$svg.AppendLine('<text x="24" y="330" text-anchor="middle" font-size="18" transform="rotate(-90 24 330)">Hardware performance (GFLOP/s), log scale</text>')
[void]$svg.AppendLine('<text x="805" y="674" font-size="13">Circle / triangle / square:</text><text x="805" y="694" font-size="13">k=100 / 1000 / 2500</text>')
[void]$svg.AppendLine('<text x="90" y="664" font-size="14">Filled: FMA enabled. Hollow: FMA disabled. Nearly coincident points retain separate IDs.</text>')
[void]$svg.AppendLine('<text x="90" y="690" font-size="14">Common roof is FMA-capable FP32 peak, not an instruction-mix-adjusted NoFMA ceiling.</text>')
[void]$svg.AppendLine('<text x="90" y="750" font-size="13">All 31 points lie below the reference roof. The Serial-labeled report is another GPU naive run, not CPU timing.</text>')
[void]$svg.AppendLine('</g></svg>')
$svg.ToString() | Set-Content "$PSScriptRoot\MMULT-roofline.svg" -Encoding utf8

$markdown = Get-Content "$root\MMULT-Analysis.md" -Raw
$body = (ConvertFrom-Markdown -InputObject $markdown).Html
$body = $body -replace '<img[^>]*src="MMULT-analysis[\\/]+MMULT-roofline.svg"[^>]*>', $svg.ToString()
$style = @'
@page { size: Letter; margin: 0.6in 0.55in; }
body { font: 10pt/1.38 "Segoe UI", Arial, sans-serif; color: #172033; max-width: 100%; }
h1 { font-size: 23pt; line-height: 1.15; margin-bottom: 14pt; }
h2 { font-size: 15pt; margin-top: 21pt; padding-bottom: 4pt; border-bottom: 1px solid #b9c9d8; }
h3 { font-size: 11.5pt; margin-top: 15pt; }
h1,h2,h3 { break-after: avoid; }
p { margin: 7pt 0; orphans: 3; widows: 3; }
table { border-collapse: collapse; width: 100%; margin: 10pt 0 13pt; font-size: 8pt; line-height: 1.28; }
th,td { border: 1px solid #b8c5d3; padding: 5px 6px; vertical-align: top; }
th { background: #e8eef5; text-align: left; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
code { font: 8.5pt Consolas, monospace; overflow-wrap: anywhere; }
pre { background: #f0f3f6; padding: 9px; white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.25; }
pre code { font-size: 8pt; }
svg { width: 100%; height: auto; display: block; break-inside: avoid; }
a { color: #2259b5; text-decoration: none; }
ul { padding-left: 17pt; }
li { margin: 4pt 0; }
em { color: #405068; }
@media print { h2 { break-inside: avoid; } }
'@
$html = "<!doctype html><html lang=`"en`"><head><meta charset=`"utf-8`"><title>MMULT Variants: Nsight Compute Analysis</title><style>$style</style></head><body>$body</body></html>"
$html | Set-Content "$PSScriptRoot\MMULT-Analysis.html" -Encoding utf8
[xml]$svg.ToString() | Out-Null
if ($body -match '<img') { throw 'Roofline image was not inlined' }
Write-Output 'Rendered HTML and validated SVG; all 31 results included.'
