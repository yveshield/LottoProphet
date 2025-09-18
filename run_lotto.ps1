# 文件名：run_lotto.ps1

# 切换到目标目录
Set-Location "D:\Dev\LottoProphet"

# 启动后台任务运行 Python 脚本
& .\.venv\Scripts\Activate.ps1
python .\main.py app
