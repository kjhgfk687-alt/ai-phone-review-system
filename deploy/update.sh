#!/bin/bash
# 一键更新云端代码/知识库（本地运行）：bash deploy/update.sh
SERVER=root@112.74.101.54
APP=/opt/phone-app
cd "$(dirname "$0")/.."
tar czf /tmp/app-update.tar.gz \
  streamlit_app.py cshi.py model_finder.py registry.py resolvers.py \
  knowledge_store.py knowledge_base.py scoring.py rag.py \
  phone_registry.yaml phone_knowledge.yaml requirements.txt .env \
  knowledge/params knowledge/docs assets
scp -q /tmp/app-update.tar.gz $SERVER:/tmp/
ssh $SERVER "cd $APP && tar xzf /tmp/app-update.tar.gz && systemctl restart phone-app && systemctl is-active phone-app && echo '✅ 云端已更新并重启'"
