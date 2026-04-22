from openai import OpenAI

class OpenAICompatibleClient:
    """
    一个用于调用任何兼容OpenAI接口的LLM服务的客户端。
    """
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用LLM API来生成回应。"""
        print("正在调用大语言模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功。")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return "错误:调用语言模型服务时出错。"


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 1. 创建客户端实例（需要替换成真实的 API 信息）
    client = OpenAICompatibleClient(
        model="gpt-5-mini",  # 或 "deepseek-chat" 等
        api_key="sk-Rf06g4odBHDXsH705SguC2DlDAp067tZKTACZgNXxYhjKaZp",  # 你的 API Key
        base_url="http://49.51.196.114:3000/v1"  # API 地址
    )
    
    # 2. 准备提示词
    system_prompt = "你是一个有用的助手，请用中文回答。"
    user_prompt = "介绍一下北京"
    
    # 3. 调用生成
    response = client.generate(user_prompt, system_prompt)
    
    # 4. 打印结果
    print(f"\n模型回复:\n{response}")