from pymongo import MongoClient
from config import MONGO
from bson.objectid import ObjectId
from typing import Dict, Any
from datetime import datetime
from tqdm import tqdm
import json

collection = MongoClient(MONGO.HOST, MONGO.PORT).pxplore.lecture_snippets

def add_snippet(snippet):
    if "children" not in snippet or len(snippet["children"]) == 0:
        print(f"Snippet has no children: {snippet['module_name']}")
        return None

    existing_snippets = get_snippets_by_module(snippet["module_id"])
    for item in existing_snippets:
        if len(item["children"]) == len(snippet["children"]) and item["children"][0]["index"] == snippet["children"][0]["index"] and item["children"][-1]["index"] == snippet["children"][-1]["index"]:
            print(f"Snippet already exists: {item['_id']}")
            return item['_id']
    
    snippet["created_at"] = datetime.now().isoformat()
    _id = collection.insert_one(snippet).inserted_id
    return _id
    

def get_snippet(snippet_id):
    return collection.find_one({"_id": ObjectId(snippet_id)}, {"_id": 0, "created_at": 0})

def get_all_snippets():
    return list(collection.find({}))

def get_snippets_by_course(course_id):
    return list(collection.find({"course_id": course_id}))

def get_snippets_by_chapter(chapter_id):
    return list(collection.find({"chapter_id": chapter_id}))

def get_snippets_by_module(module_id):
    return list(collection.find({"module_id": module_id}))

def add_label(snippet_id, label):
    collection.update_one({"_id": snippet_id}, {"$set": {"label": label}})

def parse_snippet(snippet: Dict[str, Any]) -> str:
	content_list = [item['children'][1]['script'].replace('\n', '').strip() for item in snippet['children']]
	return "\n".join(content_list)

def parse_data(course, chapter, module):
    snippet_data = []
    for idx, snippet in enumerate(module['knowledge_snippet'], 1):
        start = snippet['start_index'] - 1  # 转为0基
        end = snippet['end_index']          # 切片右开
        snippet_data.append({
            "course_name": course['course_name'],
            "course_id": course['course_id'],
            "chapter_name": chapter['chapter_name'],
            "chapter_id": chapter['chapter_id'],
            "module_name": module['module_name'],
            "module_id": module['module_id'],
            "ppt_file_id": module['ppt_file_id'],
            "ppt": module['ppt'],
            "children": module['children'][start:end]
        })
    return snippet_data

def remove_snippet(snippet_id):
    collection.delete_one({"_id": snippet_id})

if __name__ == "__main__":
    data = json.load(open("data/data_674d49431c976493ceaa1e91_课程数据-20250830112118.json", "r"))
    
    # 更新过期的文件下载链接
    print("开始更新过期的文件下载链接...")
    
    def find_matching_course_data(course_data, course_id, chapter_id, module_id):
        """根据course_id, chapter_id, module_id查找匹配的课程数据"""
        for course in course_data:
            if course.get('course_id') == course_id:
                for chapter in course.get('children', []):
                    if chapter.get('chapter_id') == chapter_id:
                        for module in chapter.get('children', []):
                            if module.get('module_id') == module_id:
                                return module
        return None
    
    def update_snippet_urls(snippet, course_data):
        """更新snippet中的过期URL"""
        course_id = snippet.get('course_id')
        chapter_id = snippet.get('chapter_id')
        module_id = snippet.get('module_id')
        
        if not all([course_id, chapter_id, module_id]):
            print(f"跳过snippet {snippet['_id']}: 缺少必要的ID字段")
            return False
        
        # 查找匹配的课程数据
        module_data = find_matching_course_data(course_data, course_id, chapter_id, module_id)
        if not module_data:
            print(f"未找到匹配的课程数据: course_id={course_id}, chapter_id={chapter_id}, module_id={module_id}")
            return False
        
        updated = False
        updates = {}
        
        # 更新PPT链接
        if module_data.get('ppt') and snippet.get('ppt') != module_data['ppt']:
            updates['ppt'] = module_data['ppt']
            updated = True

        # 更新children中的file_url
        if 'children' in snippet and 'children' in module_data:
            for snippet_child in snippet['children']:
                # 通过agenda_id在module_data中查找匹配的child
                matching_module_child = None
                for module_child in module_data['children']:
                    if snippet_child.get('agenda_id') == module_child.get('agenda_id'):
                        matching_module_child = module_child
                        break
                
                if matching_module_child and 'children' in snippet_child and 'children' in matching_module_child:
                    # 通过script内容判断是否匹配，而不是依赖file_id
                    if (len(snippet_child['children']) > 1 and 
                        len(matching_module_child['children']) > 1 and
                        snippet_child['children'][1].get('script') == matching_module_child['children'][1].get('script')):
                        
                        if 'children' not in updates:
                            updates['children'] = snippet['children'].copy()
                        
                        # 找到snippet_child在snippet['children']中的索引
                        for i, child in enumerate(snippet['children']):
                            if child.get('agenda_id') == snippet_child.get('agenda_id'):
                                # 更新file_url
                                updates['children'][i]['children'][0]['file_url'] = matching_module_child['children'][0]['file_url']
                                # 如果file_id不同，也更新file_id
                                if snippet_child['children'][0].get('file_id') != matching_module_child['children'][0].get('file_id'):
                                    updates['children'][i]['children'][0]['file_id'] = matching_module_child['children'][0]['file_id']
                                updated = True
                                break
        
        # 执行数据库更新
        if updated and updates:
            try:
                result = collection.update_one(
                    {'_id': snippet['_id']},
                    {'$set': updates}
                )
                if result.modified_count > 0:
                    return True
                else:
                    return False
            except Exception as e:
                print(f"更新出错 {snippet['_id']}: {e}")
                return False
        
        return updated
    
    # 更新所有snippets
    snippets = get_all_snippets()
    print(f"找到 {len(snippets)} 个snippets")
    
    updated_count = 0
    total_count = len(snippets)
    
    for snippet in tqdm(snippets):
        if update_snippet_urls(snippet, data):
            updated_count += 1
    
    print(f"\n更新完成! 总共更新了 {updated_count}/{total_count} 个snippets")
    
