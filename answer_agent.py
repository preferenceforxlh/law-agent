import os
from camel.agents import ChatAgent
from camel.types import ModelPlatformType, ModelType
from camel.models import ModelFactory
from prompt import answer_prompt

#os.environ["QWEN_API_KEY"] = "sk-4c461e0b5a3f4de1a2ad3ce436f2b0cc"

def answer_agent():

    # model = ModelFactory.create(
    #     model_platform=ModelPlatformType.QWEN,
    #     model_type=ModelType.QWEN_2_5_14B
    # )

    model = ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type="/root/work/law-qwen-7b",
        url="http://localhost:8000/v1",
        api_key="lvjiahui",
    )
    
    agent = ChatAgent(
        system_message=answer_prompt,
        model=model,
    )
    
    return agent