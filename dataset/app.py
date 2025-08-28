import streamlit as st
import json
import os
from io import BytesIO
import requests
from pdf2image import convert_from_bytes

# --- 配置页面 ---
st.set_page_config(
    page_title="课程内容切片标注平台",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 数据存储（简易，实际应用可对接数据库或更复杂文件结构）---
ANNOTATIONS_FILE = "./output/result.json"

# --- 模拟加载讲稿数据 ---
@st.cache_data
def load_lecture_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

lecture_data = load_lecture_data("./output/data.json")

# 过滤掉没有模块的chapter和course
filtered_courses = []
for course in lecture_data:
    filtered_chapters = []
    for chapter in course["children"]:
        seen_modules = set()
        unique_modules = []
        for module in chapter.get("children", []):
            key = (module.get("module_name"), module.get("module_id"))
            if key not in seen_modules:
                seen_modules.add(key)
                unique_modules.append(module)
        if unique_modules:
            new_chapter = chapter.copy()
            new_chapter["children"] = unique_modules
            filtered_chapters.append(new_chapter)
    if filtered_chapters:
        new_course = course.copy()
        new_course["children"] = filtered_chapters
        filtered_courses.append(new_course)

if not filtered_courses:
    st.error("没有包含模块的课程数据！")
    st.stop()

# --- 侧边栏：选择课程、章节、模块 ---
st.sidebar.header("讲稿内容导航")

# 选择课程
course_options = [course["course_name"] for course in filtered_courses]
if "selected_course_idx" not in st.session_state:
    st.session_state.selected_course_idx = 0
selected_course = st.sidebar.selectbox(
    "选择课程", course_options, index=st.session_state.selected_course_idx, key="course_selector"
)
st.session_state.selected_course_idx = course_options.index(selected_course)
current_course = filtered_courses[st.session_state.selected_course_idx]

# 选择章节
chapter_options = [chapter["chapter_name"] for chapter in current_course["children"]]
if "selected_chapter_idx" not in st.session_state or st.session_state.selected_chapter_idx >= len(chapter_options):
    st.session_state.selected_chapter_idx = 0
if chapter_options:
    selected_chapter = st.sidebar.selectbox(
        "选择章节", chapter_options, index=st.session_state.selected_chapter_idx, key="chapter_selector"
    )
    st.session_state.selected_chapter_idx = chapter_options.index(selected_chapter)
    current_chapter = current_course["children"][st.session_state.selected_chapter_idx]
else:
    selected_chapter = None
    current_chapter = None

# 选择模块
marked_module_ids = set()
if os.path.exists(ANNOTATIONS_FILE):
    try:
        with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
            all_data = json.load(f)
        def collect_marked_modules(data, course_id=None):
            if isinstance(data, dict):
                if "module_id" in data and course_id is not None:
                    marked_module_ids.add((course_id, data["module_id"]))
                for k, v in data.items():
                    if k == "course_id":
                        course_id = v
                    collect_marked_modules(v, course_id)
            elif isinstance(data, list):
                for x in data:
                    collect_marked_modules(x, course_id)
        collect_marked_modules(all_data)
    except Exception:
        pass

module_options = []
module_id_to_name = {}
current_course_id = current_course["course_id"]
for module in current_chapter["children"] if current_chapter else []:
    name = module["module_name"]
    mid = module["module_id"]
    if (current_course_id, mid) in marked_module_ids:
        display_name = f"（已完成标注）{name}"
    else:
        display_name = name
    module_options.append(mid)
    module_id_to_name[mid] = display_name

if "selected_module_idx" not in st.session_state or st.session_state.selected_module_idx >= len(module_options):
    st.session_state.selected_module_idx = 0
if module_options:
    def module_format_func(mid):
        return module_id_to_name.get(mid, mid)
    selected_module_id = st.sidebar.selectbox(
        "选择模块", module_options, key="module_selector", format_func=module_format_func
    )
    # 直接根据selected_module_id查找current_module
    current_module = None
    for m in current_chapter["children"]:
        if m["module_id"] == selected_module_id:
            current_module = m
            break
else:
    selected_module_id = None
    current_module = None

# --- 主内容区：仅显示当前模块信息 ---
st.title("📚 讲稿切片标注平台")
st.write("---")

# 新增分组逻辑
if current_module:
    # 用 session_state 临时存储未提交的分组
    session_key = f"knowledge_snippet_{current_module['module_id']}"
    if session_key not in st.session_state:
        st.session_state[session_key] = current_module.get("knowledge_snippet", []).copy() if current_module.get("knowledge_snippet") else []

    # 已保存分组直接取当前模块的 knowledge_snippet 字段（初始为空，只有本会话内添加的）
    saved_groups = current_module.get("knowledge_snippet", [])
    if saved_groups:
        st.markdown("#### 已保存分组：")
        for group in saved_groups:
            st.write(f"start: {group['start_index']}  end: {group['end_index']}  备注: {group.get('description', '')}")

    # 显示所有未保存的分组，并支持移除
    st.markdown("#### 当前分组：")
    to_remove = []
    for idx, group in enumerate(st.session_state[session_key]):
        col1, col2, col3 = st.columns([4, 4, 2])
        with col1:
            st.write(f"start: {group['start_index']}")
        with col2:
            st.write(f"end: {group['end_index']}")
        with col3:
            if st.button("移除", key=f"remove_group_{idx}"):
                st.session_state[session_key].pop(idx)
                st.rerun()
    # 实际移除
    for idx in sorted(to_remove, reverse=True):
        st.session_state[session_key].pop(idx)

    # 新增分组操作区（始终显示）
    with st.form(key="add_group_form", clear_on_submit=True):
        start_index = st.number_input("起始序号 (start_index)", min_value=1, step=1, key="start_index_input_form")
        end_index = st.number_input("结束序号 (end_index)", min_value=1, step=1, key="end_index_input_form")
        description = st.text_input("备注（可选）", key="description_input_form")
        submitted = st.form_submit_button("保存分组")
        if submitted:
            if start_index > end_index:
                st.error("起始序号不能大于结束序号！")
            else:
                st.session_state[session_key].append({
                    "start_index": int(start_index),
                    "end_index": int(end_index),
                    "description": description if description else ""
                })
                st.success(f"分组已添加: {start_index} - {end_index}")
                st.rerun()

    # 确认保存按钮
    if st.button("确认保存", key="confirm_save_groups"):
        module_id = current_module.get("module_id")
        new_snippet = st.session_state[session_key]
        if os.path.exists(ANNOTATIONS_FILE):
            with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
                try:
                    save_data = json.load(f)
                except Exception:
                    save_data = []
        else:
            save_data = []
        # 查找对应 course
        course_obj = filtered_courses[st.session_state.selected_course_idx]
        chapter_obj = course_obj["children"][st.session_state.selected_chapter_idx]
        # 复制当前 current_module 并加上 knowledge_snippet
        module_obj = current_module.copy()
        module_obj["knowledge_snippet"] = new_snippet
        course_id = course_obj["course_id"]
        chapter_id = chapter_obj["chapter_id"]
        # 查找或插入 course
        course_found = False
        for c in save_data:
            if c.get("course_id") == course_id:
                course_found = True
                # 查找或插入 chapter
                chapter_found = False
                for ch in c["children"]:
                    if ch.get("chapter_id") == chapter_id:
                        chapter_found = True
                        # 查找或插入 module
                        module_found = False
                        for idx, m in enumerate(ch["children"]):
                            if m.get("module_id") == module_id:
                                ch["children"][idx] = module_obj
                                module_found = True
                                break
                        if not module_found:
                            ch["children"].append(module_obj)
                        break
                if not chapter_found:
                    # 新 chapter
                    new_chapter = chapter_obj.copy()
                    new_chapter["children"] = [module_obj]
                    c["children"].append(new_chapter)
                break
        if not course_found:
            # 新 course
            new_course = course_obj.copy()
            new_chapter = chapter_obj.copy()
            new_chapter["children"] = [module_obj]
            new_course["children"] = [new_chapter]
            save_data.append(new_course)
        # 写入文件
        with open(ANNOTATIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=4)
        st.success("分组已保存")
        current_module["knowledge_snippet"] = new_snippet

if current_module:
    # 判断是否已完成标注
    is_marked = current_module.get("module_id") in marked_module_ids
    title_suffix = "（已完成标注）" if is_marked else ""
    st.subheader(f"{current_module['module_name']} {title_suffix}")
    # 显示所有 children 的 index、agenda_id 和 children[1]["script"]
    for i, item in enumerate(current_module.get("children", [])):
        st.markdown(f"**序号 {i+1}**")
        file_url = None
        script = None
        children_list = item.get("children", [])
        if len(children_list) > 0 and isinstance(children_list[0], dict):
            file_url = children_list[0].get("file_url")
        if len(children_list) > 1 and isinstance(children_list[1], dict):
            script = children_list[1].get("script")
        if file_url:
            try:
                response = requests.get(file_url)
                if response.status_code == 200:
                    images = convert_from_bytes(response.content, first_page=1, last_page=1)
                    if images:
                        img_byte_arr = BytesIO()
                        images[0].save(img_byte_arr, format='PNG')
                        img_byte_arr.seek(0)
                        st.image(img_byte_arr, use_container_width=True)
                    else:
                        st.write(f"file_url: {file_url}")
                else:
                    st.write(f"file_url: {file_url}")
            except Exception as e:
                st.write(f"file_url: {file_url}")
        else:
            st.write("file_url: 无")
        if script:
            st.write(f"{script}")
        else:
            st.write("【无讲稿内容】")
        st.write("---")
else:
    st.info("当前选择下暂无模块")
# 其余标注逻辑留空

# 侧边栏底部添加一个刷新按钮
st.sidebar.button("刷新页面", on_click=lambda: st.session_state.clear() or st.rerun())