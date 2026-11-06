$md='E:\Dev\EE524\AU26_GPUCompute\Homework-3\Divergence-vs-NoDivergence-Analysis.md'
$html='E:\Dev\EE524\AU26_GPUCompute\Homework-3\Divergence-vs-NoDivergence-Analysis.html'
$pdf='E:\Dev\EE524\AU26_GPUCompute\Homework-3\Divergence-vs-NoDivergence-Analysis.pdf'

$lines = Get-Content $md
$sb = New-Object System.Text.StringBuilder

$header = @"
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Divergence Analysis</title>
<style>
body{font-family:Segoe UI,Arial,sans-serif;line-height:1.45;color:#1f2937;max-width:900px;margin:40px auto;padding:0 24px;}
h1{font-size:30px;border-bottom:2px solid #e5e7eb;padding-bottom:8px;margin-top:0;}
h2{font-size:22px;margin-top:28px;color:#111827;}
h3{font-size:18px;margin-top:22px;color:#111827;}
p{margin:10px 0;}
ul{margin:8px 0 14px 24px;}
li{margin:4px 0;}
hr{border:none;border-top:1px solid #e5e7eb;margin:22px 0;}
code{background:#f3f4f6;padding:1px 5px;border-radius:4px;}
</style>
</head>
<body>
"@
[void]$sb.AppendLine($header)

$inList = $false
foreach($line in $lines){
  $t = $line.TrimEnd()
  if($t -match '^---+$'){
	if($inList){ [void]$sb.AppendLine('</ul>'); $inList=$false }
	[void]$sb.AppendLine('<hr/>')
	continue
  }
  if($t -match '^###\s+(.*)$'){
	if($inList){ [void]$sb.AppendLine('</ul>'); $inList=$false }
	[void]$sb.AppendLine('<h3>' + $Matches[1] + '</h3>')
	continue
  }
  if($t -match '^##\s+(.*)$'){
	if($inList){ [void]$sb.AppendLine('</ul>'); $inList=$false }
	[void]$sb.AppendLine('<h2>' + $Matches[1] + '</h2>')
	continue
  }
  if($t -match '^#\s+(.*)$'){
	if($inList){ [void]$sb.AppendLine('</ul>'); $inList=$false }
	[void]$sb.AppendLine('<h1>' + $Matches[1] + '</h1>')
	continue
  }
  if($t -match '^\-\s+(.*)$'){
	if(-not $inList){ [void]$sb.AppendLine('<ul>'); $inList=$true }
	[void]$sb.AppendLine('<li>' + $Matches[1] + '</li>')
	continue
  }
  if([string]::IsNullOrWhiteSpace($t)){
	if($inList){ [void]$sb.AppendLine('</ul>'); $inList=$false }
	[void]$sb.AppendLine('<p></p>')
	continue
  }
  if($inList){ [void]$sb.AppendLine('</ul>'); $inList=$false }
  [void]$sb.AppendLine('<p>' + $t + '</p>')
}
if($inList){ [void]$sb.AppendLine('</ul>') }
[void]$sb.AppendLine('</body></html>')

Set-Content -Path $html -Value $sb.ToString() -Encoding UTF8

$edge='C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
$uri = 'file:///' + ($html -replace '\\','/')
& $edge --headless --disable-gpu --print-to-pdf="$pdf" "$uri"

Get-Item $pdf | Select-Object FullName,Length,LastWriteTime
