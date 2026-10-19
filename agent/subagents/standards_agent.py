# 规范知识库子智能体（M8 扩展新增）
# 与 network_search_agent / database_query_agent / knowledge_base_agent 同构：
# 4 键字典（name / description / system_prompt / tools），
# 三个文本值来自 prompt/prompts.yml 的 sub_agents.standards 段

from agent.prompts import sub_agents_content
from tools.standards_tool import search_standards

standards_agent = {
    "name": sub_agents_content['standards']['name'],
    "description": sub_agents_content['standards']['description'],
    "system_prompt": sub_agents_content['standards']['system_prompt'],
    "tools": [search_standards],
}
