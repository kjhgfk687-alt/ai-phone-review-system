#!/bin/bash
# 把云端运行期数据（注册表/参数知识库/文档语料）拉回本地（本地运行）：bash deploy/backup.sh
SERVER=root@112.74.101.54
APP=/opt/phone-app
cd "$(dirname "$0")/.."
scp -q $SERVER:$APP/phone_registry.yaml ./phone_registry.yaml
ssh $SERVER "cd $APP/knowledge && tar czf /tmp/kb.tar.gz params docs"
scp -q $SERVER:/tmp/kb.tar.gz /tmp/kb.tar.gz
tar xzf /tmp/kb.tar.gz -C knowledge/
echo "✅ 云端数据已回拉到本地，请 git 提交存档"
