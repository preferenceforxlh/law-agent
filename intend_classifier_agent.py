import os
from camel.agents import ChatAgent
from camel.types import ModelPlatformType, ModelType
from camel.models import ModelFactory
from prompt import legal_intent_classifier_system_prompt_chinese
os.environ["QWEN_API_KEY"] = "sk-4c461e0b5a3f4de1a2ad3ce436f2b0cc"

def legal_intent_classifier_agent():
    model = ModelFactory.create(
        model_platform=ModelPlatformType.QWEN,
        model_type=ModelType.QWEN_2_5_14B
    )
    
    agent = ChatAgent(
        system_message=legal_intent_classifier_system_prompt_chinese,
        model=model,
    )
    
    return agent

if __name__ == "__main__":
    agent = legal_intent_classifier_agent()
    user_msg = "如果我被骗了怎么办"
    response = agent.step(user_msg)
    print(response.msgs[0].content)