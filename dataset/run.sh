#!/bin/bash

# 数据处理脚本优化版本
# 功能: 执行完整的数据处理流程 (snippet -> gen_label -> import_data)

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查输入数据
check_input_data() {
    log_info "检查输入数据..."
    
    if [ ! -f "./dataset/output/results.json" ]; then
        log_error "输入数据文件 ./dataset/output/results.json 不存在"
        exit 1
    fi
    
    log_success "输入数据文件存在"
}

# 执行数据处理步骤
run_step() {
    local step_name="$1"
    local command="$2"
    local description="$3"
    
    log_info "开始执行: $description"
    echo "=================================="
    
    if eval "$command"; then
        log_success "$description 完成"
        echo "=================================="
        echo
    else
        log_error "$description 失败"
        echo "=================================="
        echo
        exit 1
    fi
}

# 主函数
main() {
    echo "🚀 Pxplore 数据处理脚本"
    echo "================================"
    echo
    
    # 环境检查
    check_input_data
    
    echo "✅ 环境检查完成，开始数据处理..."
    echo
    
    # 执行数据处理步骤
    run_step "parse_snippets" "python -m dataset.parse_snippets" "步骤1: 生成知识片段"
    run_step "gen_label" "python -m dataset.gen_label" "步骤2: 生成标签"
    run_step "import_data" "python -m dataset.import_data" "步骤3: 导入向量数据库"
    
    echo "🎉 所有数据处理步骤已完成！"
    log_success "数据处理流程执行成功"
}

main
