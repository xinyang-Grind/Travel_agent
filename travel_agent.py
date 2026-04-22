import os
import re
import requests
from openai import OpenAI
from tavily import TavilyClient

# ========== 1. 系统提示词 ==========
AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对Thought和Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示:
- 每次只输出一对Thought-Action
- Action必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束

请开始吧！
"""

# ========== 2. 工具函数 ==========
def get_weather(city: str) -> str:
    """查询真实天气"""
    url = f"https://wttr.in/{city}?format=j1"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_C']
        
        return f"{city}当前天气:{weather_desc}，气温{temp_c}摄氏度"
        
    except requests.exceptions.RequestException as e:
        return f"错误:查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        return f"错误:解析天气数据失败，可能是城市名称无效 - {e}"

def get_attraction(city: str, weather: str) -> str:
    """搜索旅游景点推荐"""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误:未配置TAVILY_API_KEY环境变量。"
    
    tavily = TavilyClient(api_key=api_key)
    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"
    
    try:
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        
        if response.get("answer"):
            return response["answer"]
        
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")
        
        if not formatted_results:
            return "抱歉，没有找到相关的旅游景点推荐。"
        
        return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_results[:5])
    
    except Exception as e:
        return f"错误:执行Tavily搜索时出现问题 - {e}"

# 工具字典
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}

# ========== 3. LLM客户端 ==========
class OpenAICompatibleClient:
    """通用LLM客户端"""
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def generate(self, prompt: str, system_prompt: str) -> str:
        print("🤔 正在调用大语言模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
                temperature=0.7
            )
            answer = response.choices[0].message.content
            print("✅ 大语言模型响应成功")
            return answer
        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return "错误:调用语言模型服务时出错。"

# ========== 4. 主程序 ==========
def main():
    # ----- 配置区（请修改这里）-----
    # 选项A: 使用 DeepSeek（便宜且好用）
    API_KEY = "sk-Rf06g4odBHDXsH705SguC2DlDAp067tZKTACZgNXxYhjKaZp"  # 从 platform.deepseek.com 获取
    BASE_URL = "http://49.51.196.114:3000/v1"
    MODEL_ID = "gpt-5-mini"
    
    # 选项B: 使用 OpenAI
    # API_KEY = "sk-xxx"
    # BASE_URL = "https://api.openai.com/v1"
    # MODEL_ID = "gpt-3.5-turbo"
    
    # 选项C: 使用本地 Ollama（完全免费）
    # API_KEY = "ollama"
    # BASE_URL = "http://localhost:11434/v1"
    # MODEL_ID = "qwen2.5:7b"  # 需要先运行 ollama pull qwen2.5:7b
    
    # Tavily API Key
    TAVILY_API_KEY = "your-tavily-api-key"
    os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY
    # -----------------------------
    
    # 检查必要的配置
    if API_KEY == "your-deepseek-api-key" and "your" in API_KEY:
        print("⚠️  请先配置 API_KEY！")
        print("\n获取免费 API 的方法：")
        print("1. DeepSeek: https://platform.deepseek.com/ (注册送500万tokens)")
        print("2. Tavily: https://app.tavily.com/ (注册送1000次搜索)")
        print("3. 或使用本地 Ollama (完全免费)")
        return
    
    # 初始化 LLM
    llm = OpenAICompatibleClient(
        model=MODEL_ID,
        api_key=API_KEY,
        base_url=BASE_URL
    )
    
    # 用户输入
    user_prompt = input("🏖️  请输入你的旅行需求: ")
    # 或者直接使用示例
    # user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
    
    prompt_history = [f"用户请求: {user_prompt}"]
    
    print(f"\n用户输入: {user_prompt}")
    print("=" * 50)
    
    # 主循环
    for i in range(5):
        print(f"\n--- 循环 {i+1} ---")
        
        # 构建 Prompt
        full_prompt = "\n".join(prompt_history)
        
        # 调用 LLM
        llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
        
        # 截断多余的输出
        match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)
        if match:
            truncated = match.group(1).strip()
            if truncated != llm_output.strip():
                llm_output = truncated
                print("✂️  已截断多余的输出")
        
        print(f"\n模型输出:\n{llm_output}")
        prompt_history.append(llm_output)
        
        # 解析 Action
        action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
        if not action_match:
            observation = "错误: 未能解析到 Action 字段。请确保格式为 'Thought: ... Action: ...'"
            print(f"\nObservation: {observation}")
            prompt_history.append(f"Observation: {observation}")
            continue
        
        action_str = action_match.group(1).strip()
        
        # 检查是否完成
        if action_str.startswith("Finish"):
            final_answer = re.search(r"Finish\[(.*)\]", action_str)
            if final_answer:
                final_answer = final_answer.group(1)
            else:
                final_answer = action_str.replace("Finish[", "").rstrip("]")
            print(f"\n🎉 任务完成！\n\n最终答案: {final_answer}")
            break
        
        # 解析工具调用
        tool_match = re.search(r"(\w+)\(.*?\)", action_str)
        if not tool_match:
            observation = f"错误: 无法解析工具调用格式 '{action_str}'"
            print(f"\nObservation: {observation}")
            prompt_history.append(f"Observation: {observation}")
            continue
        
        tool_name = tool_match.group(1)
        
        # 提取参数
        args_match = re.search(r"\((.*)\)", action_str)
        kwargs = {}
        if args_match:
            args_str = args_match.group(1)
            # 解析 key="value" 格式
            for key, value in re.findall(r'(\w+)="([^"]*)"', args_str):
                kwargs[key] = value
        
        # 执行工具
        if tool_name in available_tools:
            print(f"🔧 执行工具: {tool_name}{kwargs}")
            observation = available_tools[tool_name](**kwargs)
        else:
            observation = f"错误: 未定义的工具 '{tool_name}'"
        
        print(f"\nObservation: {observation}")
        print("=" * 50)
        prompt_history.append(f"Observation: {observation}")

if __name__ == "__main__":
    main()