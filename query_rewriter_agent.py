import os
from camel.agents import ChatAgent
from camel.types import ModelPlatformType, ModelType
from camel.models import ModelFactory
from prompt import query_rewriter_system_prompt_chinese,query_rewriter_system_prompt

os.environ["QWEN_API_KEY"] = "sk-4c461e0b5a3f4de1a2ad3ce436f2b0cc"

def query_rewriter_agent():
    model = ModelFactory.create(
        model_platform=ModelPlatformType.QWEN,
        model_type=ModelType.QWEN_2_5_14B
    )
    
    agent = ChatAgent(
        system_message=query_rewriter_system_prompt_chinese,
        model=model,
    )
    
    return agent

if __name__ == "__main__":
    agent = query_rewriter_agent()
    # 测试用例
    test_queries = [
        "如果我驾驶证被扣分了，会影响后代嘛",
        "那如果我酒驾呢",
    ]
    
    for query in test_queries:
        response = agent.step(query)
        print(f"原始问题: {query}")
        print(f"重写后: {response.msgs[0].content}\n") 