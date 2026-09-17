#!/bin/bash
# 全新服务器部署记录脚本（在云服务器上以 root 运行）
# 前置：阿里云轻量 Ubuntu 24.04，已放行 8501 端口
set -e
echo "== 1. swap 2G =="
[ -f /swapfile ] || { fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile && echo '/swapfile none swap sw 0 0' >> /etc/fstab; }
echo "== 2. Docker（阿里云镜像） =="
command -v docker >/dev/null || curl -fsSL https://get.docker.com | sh -s docker --mirror Aliyun
echo "== 3. 代码与知识库 =="
mkdir -p /opt/phone-app   # 解压 deploy.tar.gz 到此目录（含 .env）
echo "== 4. Python 依赖 =="
cd /opt/phone-app && python3 -m venv .venv && .venv/bin/pip install -q -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
echo "== 5. Reader 容器 =="
[ -f /tmp/reader.tar ] && docker load -i /tmp/reader.tar
docker run -d --name reader --restart unless-stopped -p 127.0.0.1:3001:8081 ghcr.io/jina-ai/reader:main
echo "== 6. Streamlit 服务 =="
cat > /etc/systemd/system/phone-app.service <<'UNIT'
[Unit]
Description=Phone Review App (Streamlit)
After=network-online.target docker.service
[Service]
WorkingDirectory=/opt/phone-app
ExecStart=/opt/phone-app/.venv/bin/streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload && systemctl enable --now phone-app
echo "✅ 部署完成：http://服务器IP:8501"
