#!/bin/bash

set -e

echo "拉取最新代码..."#服务器需要自动更新代码
git pull

echo "检查虚拟环境..."

if [ ! -d "agent" ]; then
    python3 -m venv agent
fi

source agent/bin/activate

echo "安装依赖..."

pip install -r requirements.txt

echo "启动项目..."

python agent.py