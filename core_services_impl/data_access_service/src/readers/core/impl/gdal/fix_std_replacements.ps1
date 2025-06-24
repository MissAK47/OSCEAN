# GDAL模块 std -> boost 替换脚本
param(
    [string]$SourceFile = "gdal_format_handler_backup2.cpp",
    [string]$OutputFile = "gdal_format_handler.cpp"
)

Write-Host "开始修复 $SourceFile -> $OutputFile"

# 读取文件内容
$content = Get-Content $SourceFile -Raw -Encoding UTF8

# 替换所有std用法
$content = $content -replace 'std::optional<DataChunk>', 'boost::optional<DataChunk>'
$content = $content -replace 'std::optional<oscean::core_services::VariableMeta>', 'boost::optional<oscean::core_services::VariableMeta>'
$content = $content -replace 'std::optional<oscean::core_services::CRSInfo>', 'boost::optional<oscean::core_services::CRSInfo>'
$content = $content -replace 'std::future<void>', 'boost::future<void>'
$content = $content -replace 'std::future<bool>', 'boost::future<bool>'
$content = $content -replace 'std::future<std::vector<DataChunk>>', 'boost::future<std::vector<DataChunk>>'
$content = $content -replace 'std::vector<std::future<boost::optional<DataChunk>>>', 'std::vector<boost::future<boost::optional<DataChunk>>>'
$content = $content -replace 'std::async\(std::launch::async', 'boost::async(boost::launch::async'
$content = $content -replace 'std::nullopt', 'boost::none'

# 添加必要的boost包含（如果还没有）
if ($content -notmatch '#include <boost/optional\.hpp>') {
    $content = $content -replace '(#include <boost/thread/future\.hpp>)', "$1`n#include <boost/optional.hpp>"
}

# 保存文件（使用UTF8编码）
[System.IO.File]::WriteAllText((Join-Path $PWD $OutputFile), $content, [System.Text.Encoding]::UTF8)

Write-Host "替换完成: $OutputFile"
Write-Host "已替换的项目:"
Write-Host "- std::optional -> boost::optional"
Write-Host "- std::future -> boost::future"
Write-Host "- std::async -> boost::async"
Write-Host "- std::nullopt -> boost::none" 