import os
import uuid
import chromadb
from typing import List, Dict, Any
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter,MarkdownHeaderTextSplitter
from datasets import load_dataset

class ChromaRetriever:
    """基于ChromaDB的检索器实现
    
    支持txt、md、json等格式文档的存储和检索
    使用langchain进行文档分块
    """
    
    def __init__(
        self,
        collection_name: str = "law_data",
    ):
        """初始化ChromaDB检索器
        
        Args:
            collection_name: ChromaDB集合名称
        """
        self.client = chromadb.PersistentClient(
            path="/root/work/chroma_db"
        )

        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="/root/work/model/bge-large-zh-v1.5"
        )
        
        #self.collection = self.client.create_collection(collection_name,embedding_function=sentence_transformer_ef)
        self.collection = self.client.get_collection(collection_name,embedding_function=sentence_transformer_ef)
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
    
    def process(self,root_path: str) -> None:
        # 遍历root_path目录下的所有文件
        for root,dirs,files in os.walk(root_path):
            for file in files:
                file_path = os.path.join(root,file)
                print(f"*** Processing file ***: {file_path}")
                self.add_document(file_path)
    
    def _load_document(self, file_path: str) -> Any:
        """加载文档内容
        
        支持.txt、.md、.json格式
        """
        if file_path.endswith('.json'):
            return load_dataset("json",data_files=file_path,split="train")
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def get_source(self,path) -> str:
        """
        get the file type of the file
        """
        return path.split(".")[-1]

    def add_document(
        self,
        file_path: str,
    ) -> None:
        """添加文档到ChromaDB
        
        Args:
            file_paths: 文档路径
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

        # 为每个分块添加文档来源信息
        chunk_metadata = [{
            "source": self.get_source(file_path),
        } for i in range(len(chunks))]
        
        # 为每个分块生成唯一ID
        ids = [str(uuid.uuid4()) for i in range(len(chunks))]
        
        self.collection.add(
            documents=chunks,
            metadatas=chunk_metadata,
            ids=ids
        )
    
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
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )
        
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity_score': results['distances'][0][i]
            })
            
        return formatted_results 


if __name__ == "__main__":
    client = ChromaRetriever()
    #client.process("/root/work/data/")
    result = client.query("如果同学爸爸强奸了别人家的孩子，会受到什么惩罚")
    print(result)