import os
from camel.agents import ChatAgent
from camel.types import ModelPlatformType, ModelType
from camel.models import ModelFactory
from prompt import context_decision_system_prompt

os.environ["QWEN_API_KEY"] = "sk-4c461e0b5a3f4de1a2ad3ce436f2b0cc"

def context_decision_agent():
    model = ModelFactory.create(
        model_platform=ModelPlatformType.QWEN,
        model_type=ModelType.QWEN_2_5_14B
    )
    
    agent = ChatAgent(
        system_message=context_decision_system_prompt,
        model=model,
    )
    
    return agent