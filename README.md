# Pxplore - 个性化知识推荐服务

## 📖 项目简介

Pxplore 是一个基于大模型的个性化知识推荐服务系统，专注于为学生提供智能化的学习内容推荐和教学风格适配。该系统通过分析学生的学习行为数据、交互历史和认知水平，自动推荐最适合的教学内容片段，并将其适配为符合学生学习偏好的个性化表述。

## ✨ 核心功能

### 🧑‍🎓 学生画像建模 (Student Profiling)
- **语言分析**：基于学生在线课堂讨论日志，分析学生的语言表达能力和认知模式
- **行为分析**：解析学生的页面交互、回顾循环、测验结果等学习行为数据
- **认知层级评估**：采用布鲁姆认知分类法等教育理论，评估学生的认知发展水平
- **动态画像更新**：实时更新学生的学习状态和能力模型

### 📚 智能内容推荐 (Content Recommendation)
- **混合检索算法**：结合密集向量检索和 BM25 算法，实现精准的内容匹配
- **个性化排序**：基于学生画像和学习上下文，智能选择最适合的教学内容片段
- **多维度评估**：考虑内容相关性、认知层级匹配度、逻辑连贯性等多个维度
- **实时推荐**：支持异步任务处理，快速响应推荐请求

### 🎨 教学风格适配 (Style Adaptation)
- **自然语言生成**：将推荐的教学内容转化为亲切自然的讲师风格
- **上下文衔接**：确保新内容与历史学习内容的逻辑连贯性
- **个性化表述**：根据学生特点调整语言风格和表达方式
- **结构化输出**：生成包含开场白、主体内容和收尾语的完整教学脚本

### 🎯 会话管理 (Session Management)
- **多轮对话支持**：维护学生与系统的长期交互历史
- **状态持久化**：保存学生的学习进度和偏好设置
- **智能应答**：基于上下文生成恰当的教学回应

## 🏗️ 系统架构

```
Pxplore/
├── app.py                 # 应用入口
├── base.py               # 数据模型定义
├── config.py             # 配置文件
├── requirements.txt      # 依赖包列表
├── utils.py             # 工具函数
├── data/                # 基础数据结构
│   ├── profile.py       # 学生画像数据结构
│   ├── session.py       # 会话数据结构
│   ├── snippet.py       # 内容片段数据结构
│   └── task.py          # 任务管理数据结构
├── dataset/             # 原始数据处理脚本
│   ├── app.py           # 数据处理应用
│   ├── gen_label.py     # 标签生成
│   ├── import_data.py   # 数据导入
│   ├── parse_snippets.py # 内容片段解析
│   ├── output/          # 处理结果输出
│   └── prompts/         # 数据处理提示词
├── model/               # 推荐算法优化
│   ├── calculate_consistency.py  # 一致性计算
│   ├── calculate_reward.py       # 奖励计算
│   ├── data_preprocessing.py     # 数据预处理
│   ├── evaluation.py            # 模型评估
│   ├── prepare_data.py          # 数据准备
│   ├── data/                    # 训练数据
│   ├── prompts/                 # 模型提示词
│   └── test/                    # 测试脚本
└── service/             # 核心服务模块
    ├── llm/             # 大语言模型服务
    │   ├── base.py      # LLM 基础类
    │   ├── openai.py    # OpenAI 接口
    │   └── qwen.py      # Qwen 模型接口
    ├── scripts/         # 主要服务脚本
    │   ├── hybrid_retriever.py    # 混合检索器
    │   ├── session_controller.py  # 会话控制器
    │   ├── slide_segmentation.py  # 幻灯片分割
    │   ├── snippet_recommender.py # 内容推荐器
    │   ├── student_profiling.py   # 学生画像分析
    │   ├── style_adaptation.py    # 风格适配器
    │   ├── buffer/                # 缓存目录
    │   └── prompts/               # 服务提示词
    ├── test/            # 服务测试
    └── utils/           # 服务工具
        ├── data_transformer.py    # 数据转换
        ├── dense_embedding.py     # 密集嵌入
        └── episodes_processor.py  # 学习事件处理
```

## 🚀 快速开始

### 环境要求

- Python 3.12+
- FastAPI
- 支持的 LLM 服务（OpenAI GPT 或 Qwen）

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/Pxplore/pxplore-algo.git
cd pxplore-algo
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置环境**
根据需要配置 `config.py` 中的 LLM 服务参数

4. **启动服务**
```bash
python app.py
```

服务将在 `http://0.0.0.0:8899` 启动

## 📋 API 接口

### 学生画像分析

**启动分析任务**
```http
POST /student_profile
Content-Type: application/json

{
  "behavioral_data": {
    "discussion_threads": [...],
    "page_interactions": [...],
    "review_loops": [...],
    "quizzes": [...]
  }
}
```

**查询分析状态**
```http
GET /student_profile/status/{task_id}
```

### 内容推荐

**启动推荐任务**
```http
POST /recommend
Content-Type: application/json

{
  "student_profile": {...},
  "interaction_history": "学习交互历史",
  "title": "当前主题",
  "model": "gpt-4o"
}
```

**查询推荐结果**
```http
GET /recommend/status/{task_id}
```

### 风格适配

**启动适配任务**
```http
POST /style_adapt
Content-Type: application/json

{
  "student_profile": {...},
  "interaction_history": "学习交互历史",
  "title": "当前主题",
  "recommend_id": "推荐内容ID",
  "recommend_reason": "推荐理由"
}
```

**查询适配结果**
```http
GET /style_adapt/status/{task_id}
```

### 会话管理

**处理消息**
```http
POST /session/handle_message
Content-Type: application/json

{
  "session_id": "会话ID",
  "scripts": [...],
  "history": [...],
  "message": "用户消息"
}
```

**获取会话数据**
```http
GET /session/{session_id}
```

## 🔧 核心算法

### 混合检索算法
- **密集向量检索**：使用 Qdrant 向量数据库进行语义相似度匹配
- **稀疏检索**：采用 BM25 算法进行关键词匹配
- **混合评分**：`hybrid_score = α × bm25_score + (1-α) × dense_score`

### 学生认知层级评估
- **布鲁姆分类法**：评估学生在知识、理解、应用、分析、综合、评价六个层次的能力
- **动态窗口分析**：采用自适应窗口大小分析学生的认知趋势
- **线性回归趋势**：通过回归分析判断学生认知水平的变化趋势

### 个性化内容适配
- **上下文感知**：结合历史学习内容和当前学习状态
- **风格迁移**：将标准教学内容转化为个性化表述
- **结构化生成**：生成包含过渡语、主体内容和收尾语的完整脚本

## 📊 数据格式

### 学生行为数据
```json
{
  "discussion_threads": [
    {
      "thread_id": "线程ID",
      "messages": [
        {
          "author_type": "student|teacher|ai",
          "content": "消息内容",
          "timestamp": "时间戳"
        }
      ]
    }
  ],
  "page_interactions": [...],
  "review_loops": [...],
  "quizzes": [...]
}
```

### 推荐结果格式
```json
{
  "selected_candidate": {
    "id": "内容片段ID",
    "bloom_level": "认知层级",
    "summary": "内容摘要",
    "content": "具体内容"
  },
  "reason": "推荐理由"
}
```

### 风格适配结果
```json
{
  "start_speech": "过渡开场白",
  "new_scripts": [
    "优化后的教学内容句子1",
    "优化后的教学内容句子2"
  ],
  "end_speech": "温暖的收尾语"
}
```
