import os
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import PyPDF2
import markdown
import html2text
import json
from tqdm import tqdm
import tiktoken
import re
from bs4 import BeautifulSoup
from IPython.display import display, Code, Markdown

load_dotenv("api_key.env", override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

class BaseEmbeddings:
    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api

    def get_embedding(self, text: str, model: str) -> List[float]:
        raise NotImplementedError

    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        compute the cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude


class OpenAIEmbedding(BaseEmbeddings):
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
            )

    def get_embedding(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        if self.is_api:
            text = text.replace("\n", " ")
            return self.client.embeddings.create(input=[text], model=model).data[0].embedding
        else:
            raise NotImplementedError


class ReadFiles:
    def __init__(self, path: str) -> None:
        self.path = path
        self.file_list = self.get_files()

    def get_files(self):
        file_list = []
        for filepath, dirnames, filenames in os.walk(self._path):
            for filename in filenames:
                if filename.endswith('.md'):
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith('.txt'):
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith('.pdf'):
                    file_list.append(os.path.join(filepath, filename))
        return file_list

    def get_content(self, max_token_len: int = 600, cover_content: int = 150):
        docs = []
        for file in self.file_list:
            content = self.read_file_content(file)
            chunk_content = self.get_chunk(content, max_token_len=max_token_len, cover_content=cover_content)
            docs.extend(chunk_content)
        return docs

    @classmethod
    def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150):
        chunk_text = []
        curr_len = 0
        curr_chunk = ''
        lines = text.split('\n')

        for line in lines:
            line = line.replace(' ', '')
            line_len = len(enc.encode(line))
            if line_len > max_token_len:
                num_chunks = (line_len + token_len - 1) // token_len
                for i in range(num_chunks):
                    start = i * token_len
                    end = start + token_len
                    while not line[start: end].rstrip().isspace():
                        start += 1
                        end += 1
                        if start >= line_len:
                            break
                    curr_chunk = curr_chunk[-cover_content:]
                    chunk_text.append(curr_chunk)
                start = (num_chunks - 1) * token_len
                curr_chunk = curr_chunk[-cover_content:] + line[start: end]
                chunk_text.append(curr_chunk)
            if curr_len + line_len <= max_token_len:
                curr_chunk += line + '\n'
                curr_len += line_len + 1
            else:
                chunk_text.append(curr_chunk)
                curr_chunk = curr_chunk[-cover_content:] + line
                curr_len = line_len + cover_content

        if curr_chunk:
            chunk_text.append(curr_chunk)

        return chunk_text

    @classmethod
    def read_file_content(cls, file_path: str):
        if file_path.endswith('.pdf'):
            return cls.read_pdf(file_path)
        elif file_path, endswith('.md'):
            return cls.read_markdown(file_path)
        elif file_path.endswith('.txt'):
            return cls.read_text(file_path)
        else:
            raise ValueError("Unsupported file type")

    @classmethod
    def read_pdf(cls, file_path: str):
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
            return text

    @classmethod
    def read_markdown(cls, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            md_text = file.read()
            html_text = markdown.markdown(md_text)
            soup = BeautifulSoup(html_text, 'html.parser')
            plain_text = soup.get_text()
            text = re.sub(r'http\S+', '', plain_text)
            return text

    @classmethod
    def read_text(cls, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

class Documents:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def get_content(self):
        with open(self.path, mode='r', encoding='utf-8') as file:
            content = json.load(f)
        return content

class VectorStore:
    def __init__(self, document: List[str] = None) -> None:
        if document is None:
            document = []
        self.document = document
        self.vectors = []

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        self.vectors = [EmbeddingModel.get_embedding(doc) for doc in self.document]
        return self.vectors

    def persist(self, path: str = 'storage'):
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(os.path.join(path, 'vectors.npy'), self.vectors)
        with open(os.path.join(path, 'documents.txt'), 'w') as f:
            for doc in self.document:
                f.write(f"{doc}\n")

    def load_vector(self, path: str = 'storage'):
        self.vectors = np.load(os.path.join(path, 'vectors.npy')).tolist()
        with open(os.path.join(path, 'documents.txt'), 'r') as f:
            self.document = [line.strip() for line in f.readlines()]

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude

    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
        query_vector = EmbeddingModel.get_embedding(query)
        similarities = [self.get_similarity(query_vector, vector) for vector in self.vectors]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [self.document[idx] for idx in top_k_indices]

documents = [
    "Machine learning is a branch of artificial intelligence.",
    "Deep learning is a special method of machine learning.",
    "Supervised learning is a way of training models.",
    "Reinforcement learning learns through rewards and punishments.",
    "Unsupervised learning does not rely on labeled data."
]

class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        """
        :param prompt: user's query
        :param history: conversation history
        :param content: the provided contextual information
        """
        pass

    def load_model(self):
        pass

class GPTchat(BaseModel):
    def __init__(self, api_key: str, base_url: str = OPENAI_BASE_URL):
        super().__init__()
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, prompt: str, history: List = [], content: str = '') -> str:
        full_prompt = PROMPT_TEMPLATE['GPT_PROMPT_TEMPLATE'].format(question=prompt)
        response = self.client.chat.completions.create(
            model='gpt-5',
            messages=[
                {'role': 'user', 'content': full_prompt}
            ]
        )
        return response.choices[0].message.content


PROMPT_TEMPLATE = dict(
    GPT_PROMPT_TEMPLATE=
    """
    Here is a reference passage that may or may not be related to the question.  
    If you think the reference passage is relevant to the question, first summarize its content.  
    If you think the reference passage is irrelevant to the question, then use your own original knowledge to answer the user's question.  
    Always answer in English.  

    Question: {question}

    Reference context:
    ...
    {context}
    ...

    Useful answer:
    """
)


