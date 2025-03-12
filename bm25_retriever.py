import os
import json 
import bm25s
import numpy as np
from datasets import load_dataset
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter,MarkdownHeaderTextSplitter

class BM25Retriever:
    """基于BM25算法的检索器实现
    
    支持txt、md、json等格式文档的存储和检索
    使用langchain进行文档分块
    """
    
    def __init__(self):
        """初始化BM25检索器"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=256,
            chunk_overlap=32,
        )
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
        )
        self.retriever  = bm25s.BM25()
        self.reload_retriever = None
        self.chunks = []
    
    def process(self,root_path: str) -> None:
        # 遍历root_path目录下的所有文件
        print(f"*** Processing directory ***: {root_path}")
        for root,dirs,files in os.walk("/root/work/data/"):
            for file in files:
                file_path = os.path.join(root, file)
                print(f"*** Processing file ***: {file_path}")
                self.add_document(file_path)

        corpus_tokens = bm25s.tokenize(self.chunks, stopwords="zh")
        self.retriever.index(corpus_tokens)
        print("*** save index ***")
        self.retriever.save("/root/work/index_bm25",corpus = self.chunks)
    
    def _load_document(self, file_path: str) -> Any:
        """加载文档内容
        支持.txt、.md、.json格式
        """
        if file_path.endswith('.json'):
            return load_dataset("json",data_files=file_path,split="train")
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def add_document(
        self,
        file_path: str,
    ) -> None:
        """添加文档并构建BM25索引
        Args:
            file_path: 文档路径
        """
        data = self._load_document(file_path)
        # .json
        if file_path.endswith(".json"):
            chunks = [sub_data["content"] for sub_data in data]
        else:
            # .txt
            if file_path.endswith(".txt"):
                chunks = self.text_splitter.create_documents([data])
                chunks = [chunk.page_content for chunk in chunks]
            # .md
            else:
                chunks = self.markdown_splitter.split_text(data)
                chunks = self.text_splitter.split_documents(chunks)
                chunks = [chunk.page_content for chunk in chunks]
        
        print(f"*** load {len(chunks)} chunks from {file_path} ***")
        self.chunks.extend(chunks)
    
    def query(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """检索相关文档片段
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
        Returns:
            包含相关度分数、文档内容和元数据的结果列表
        """
        if not self.reload_retriever:
            self.reload_retriever = bm25s.BM25.load("/root/work/index_bm25",load_corpus=True)
            
        # 计算BM25分数
        query_tokens = bm25s.tokenize(query)
        results, scores = self.reload_retriever.retrieve(query_tokens, k=top_k)
        
        context = []
        for i in range(results.shape[1]):
            doc, score = results[0, i], scores[0, i]
            context.append({
                "id":doc["id"],
                "text":doc["text"],
                "score":score
            })
        return context

if __name__ == "__main__":
    client = BM25Retriever()
    # client.process("root/work/data/")
    result = client.query("我对老板态度不好，把我辞退了，能要补偿吗？")
    print(result)
    