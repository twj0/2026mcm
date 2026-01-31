#!/bin/bash

# 自动同步 examples 文件夹内所有子文件夹的 git 仓库
# Usage: ./sync_all.sh

set -e  # 遇到错误时退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "开始同步 examples 文件夹内的所有 git 仓库"
echo "=========================================="
echo ""

# 统计信息
total=0
success=0
failed=0
skipped=0

# 遍历所有子文件夹
for dir in */; do
    # 移除末尾的斜杠
    dir_name="${dir%/}"

    # 跳过非目录
    if [ ! -d "$dir_name" ]; then
        continue
    fi

    total=$((total + 1))

    echo "[$total] 处理: $dir_name"

    # 检查是否是 git 仓库
    if [ ! -d "$dir_name/.git" ]; then
        echo "  ⊘ 跳过 (不是 git 仓库)"
        skipped=$((skipped + 1))
        echo ""
        continue
    fi

    # 进入目录并执行 git pull
    cd "$dir_name"

    # 检查是否有远程仓库
    if ! git remote -v | grep -q "origin"; then
        echo "  ⊘ 跳过 (没有配置 origin 远程仓库)"
        skipped=$((skipped + 1))
        cd ..
        echo ""
        continue
    fi

    # 获取当前分支
    current_branch=$(git branch --show-current)

    if [ -z "$current_branch" ]; then
        echo "  ⊘ 跳过 (处于 detached HEAD 状态)"
        skipped=$((skipped + 1))
        cd ..
        echo ""
        continue
    fi

    echo "  分支: $current_branch"

    # 执行 git pull
    if git pull origin "$current_branch" 2>&1; then
        echo "  ✓ 同步成功"
        success=$((success + 1))
    else
        echo "  ✗ 同步失败"
        failed=$((failed + 1))
    fi

    cd ..
    echo ""
done

echo "=========================================="
echo "同步完成"
echo "=========================================="
echo "总计: $total 个目录"
echo "成功: $success"
echo "失败: $failed"
echo "跳过: $skipped"
echo "=========================================="
