$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$port = 8321
$listener = New-Object System.Net.HttpListener
$listener.Prefixes.Add("http://localhost:$port/")
$listener.Start()
Write-Host "Serving $root at http://localhost:$port/"

$mime = @{
  ".html"="text/html; charset=utf-8"; ".css"="text/css"; ".js"="application/javascript"
  ".png"="image/png"; ".jpg"="image/jpeg"; ".jpeg"="image/jpeg"; ".gif"="image/gif"
  ".svg"="image/svg+xml"; ".mp4"="video/mp4"; ".webm"="video/webm"; ".pdf"="application/pdf"
  ".ico"="image/x-icon"; ".woff"="font/woff"; ".woff2"="font/woff2"; ".ttf"="font/ttf"; ".json"="application/json"
}

while ($listener.IsListening) {
  try {
    $ctx = $listener.GetContext()
    $req = $ctx.Request
    $res = $ctx.Response
    $path = [System.Uri]::UnescapeDataString($req.Url.AbsolutePath)

    # POST /save?name=<relpath> : body is base64 image data, decoded and written under root
    if ($req.HttpMethod -eq "POST" -and $path -eq "/save") {
      $name = $req.QueryString["name"]
      $reader = New-Object System.IO.StreamReader($req.InputStream, $req.ContentEncoding)
      $b64 = $reader.ReadToEnd(); $reader.Close()
      $comma = $b64.IndexOf(",")
      if ($comma -ge 0) { $b64 = $b64.Substring($comma + 1) }
      $outPath = Join-Path $root ($name -replace "/", "\")
      $dir = Split-Path $outPath -Parent
      if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
      [System.IO.File]::WriteAllBytes($outPath, [System.Convert]::FromBase64String($b64))
      $res.StatusCode = 200
      $res.AddHeader("Access-Control-Allow-Origin", "*")
      $bytes = [System.Text.Encoding]::UTF8.GetBytes("saved $name")
      $res.OutputStream.Write($bytes, 0, $bytes.Length)
      $res.Close()
      continue
    }

    # POST /__savehtml?name=<relpath.html> : body is raw UTF-8 HTML, written under root (defaults to index.html)
    if ($req.HttpMethod -eq "POST" -and $path -eq "/__savehtml") {
      $name = $req.QueryString["name"]; if (-not $name) { $name = "index.html" }
      $reader = New-Object System.IO.StreamReader($req.InputStream, [System.Text.Encoding]::UTF8)
      $html = $reader.ReadToEnd(); $reader.Close()
      $outPath = Join-Path $root ($name -replace "/", "\")
      $fullRoot = (Resolve-Path $root).Path
      $dir = Split-Path $outPath -Parent
      $okDir = (Test-Path $dir) -and ((Resolve-Path $dir).Path.StartsWith($fullRoot))
      $res.AddHeader("Access-Control-Allow-Origin", "*")
      if ($okDir -and $outPath.ToLower().EndsWith(".html") -and $html.Length -gt 200) {
        if (Test-Path $outPath) { Copy-Item $outPath "$outPath.bak" -Force }
        [System.IO.File]::WriteAllText($outPath, $html, (New-Object System.Text.UTF8Encoding($false)))
        $res.StatusCode = 200
        $bytes = [System.Text.Encoding]::UTF8.GetBytes("saved $name")
      } else {
        $res.StatusCode = 403
        $bytes = [System.Text.Encoding]::UTF8.GetBytes("forbidden")
      }
      $res.OutputStream.Write($bytes, 0, $bytes.Length)
      $res.Close()
      continue
    }

    if ($path -eq "/") { $path = "/index.html" }
    $file = Join-Path $root ($path -replace "/", "\")
    $fullRoot = (Resolve-Path $root).Path

    if (-not (Test-Path $file -PathType Leaf) -or -not ((Resolve-Path $file).Path.StartsWith($fullRoot))) {
      $res.StatusCode = 404
      $bytes = [System.Text.Encoding]::UTF8.GetBytes("404 Not Found: $path")
      $res.OutputStream.Write($bytes, 0, $bytes.Length)
      $res.Close()
      continue
    }

    $ext = [System.IO.Path]::GetExtension($file).ToLower()
    $ct = $mime[$ext]; if (-not $ct) { $ct = "application/octet-stream" }
    $res.ContentType = $ct
    $res.AddHeader("Accept-Ranges", "bytes")

    $fs = [System.IO.File]::OpenRead($file)
    $len = $fs.Length
    $start = 0; $end = $len - 1
    $range = $req.Headers["Range"]
    if ($range -and $range -match "bytes=(\d*)-(\d*)") {
      if ($Matches[1]) { $start = [long]$Matches[1] }
      if ($Matches[2]) { $end = [long]$Matches[2] } elseif ($Matches[1]) { $end = $len - 1 }
      if ($start -gt $end -or $start -ge $len) {
        $res.StatusCode = 416
        $fs.Close(); $res.Close(); continue
      }
      $res.StatusCode = 206
      $res.AddHeader("Content-Range", "bytes $start-$end/$len")
    }
    $count = $end - $start + 1
    $res.ContentLength64 = $count
    $fs.Seek($start, [System.IO.SeekOrigin]::Begin) | Out-Null
    $buf = New-Object byte[] 65536
    $remaining = $count
    try {
      while ($remaining -gt 0) {
        $toRead = [Math]::Min($buf.Length, $remaining)
        $n = $fs.Read($buf, 0, $toRead)
        if ($n -le 0) { break }
        $res.OutputStream.Write($buf, 0, $n)
        $remaining -= $n
      }
    } catch { }
    $fs.Close()
    $res.Close()
  } catch { }
}
