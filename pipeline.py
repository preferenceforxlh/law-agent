import os
import torch
from bm25_retriever import BM25Retriever
from chroma_retriever import ChromaRetriever
from intend_classifier_agent import legal_intent_classifier_agent
from query_rewriter_agent import query_rewriter_agent
from context_decision_agent import context_decision_agent
from answer_agent import answer_agent
from duckduckgo_search import DDGS
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ["http_proxy"] = "http://127.0.0.1:17043"
os.environ["https_proxy"] = "http://127.0.0.1:17043"
os.environ["QWEN_API_KEY"] = "sk-4c461e0b5a3f4de1a2ad3ce436f2b0cc"


query = """
上下文：
{context}
问题：
{user_query}
"""
max_steps = 3
context = []
ranked_context = []

def pipeline(user_query: str):
    intent_classifier = legal_intent_classifier_agent()
    query_rewriter = query_rewriter_agent()
    context_decision = context_decision_agent()
    answer = answer_agent()
    bm25_retriever = BM25Retriever()
    chroma_retriever = ChromaRetriever()
    ddgs = DDGS()

    tokenizer = AutoTokenizer.from_pretrained("/root/work/model/bge-rerank")
    model = AutoModelForSequenceClassification.from_pretrained("/root/work/model/bge-rerank").to("cuda")
    model.eval()


    print(f"用户问题: {user_query}")
    print("开始判断用户意图")
    intent_response = intent_classifier.step(user_query)
    if intent_response.msgs[0].content == "no":
        return "请输入法律相关问题"
    else:
        print("正在进行query改写")
        query_response = query_rewriter.step(user_query)
        rewrite_query = query_response.msgs[0].content
        print(f"query改写结果:\n {rewrite_query}")
        print("正在进行上下文检索")
        bm25_context = bm25_retriever.query(rewrite_query,top_k=5)
        chroma_context = chroma_retriever.query(rewrite_query,top_k=5)

        context1 = [data["text"] for data in bm25_context]
        context2 = [data["text"] for data in chroma_context]
        context.extend(context1)
        context.extend(context2)
        
        context_str = "\n\n".join(context)
        input_text = query.format(context=context_str, user_query=rewrite_query)
        
        print("正在进行上下文决策")
        for i in range(max_steps):
            response = context_decision.step(input_text)
            print(f"step {i}: {response}")
            if response.msgs[0].content == "no":
                print("正在进行DuckDuckGo搜索")
                results = ddgs.text(rewrite_query)
                context.extend([result["body"] for result in results])
                input_text = query.format(context="\n\n".join(context), user_query=rewrite_query)
            else:
                break
        
        print("正在进行rerank排序")
        pairs = [[rewrite_query, context] for context in context]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(model.device)
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
            values, indices = scores.cpu().topk(5)
            indices = indices.tolist()
            ranked_context = [context[i] for i in indices]
        
        print("正在进行法律解答")
        final_query = query.format(context="\n\n".join(ranked_context), user_query=rewrite_query)
        final_response = answer.step(final_query)
        print(f"final_response:\n {final_response.msgs[0].content}")
        return 

if __name__ == "__main__":
    pipeline("如果我被骗了怎么办")