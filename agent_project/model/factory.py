from abc import abstractmethod, ABC
from typing import Optional
from langchain_community.chat_models import ChatTongyi
from langchain_community.document_compressors import DashScopeRerank
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
import sys
sys.path.insert(0, r"D:\Desktop\experiment\llm\AI\agent_project")
from utils.config_handler import rag_conf,chroma_conf

class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self)->Optional[Embeddings|BaseChatModel]:
        pass

class ChatModelFactory(BaseModelFactory):
    def generator(self) ->Optional[Embeddings|BaseChatModel]:
        return ChatTongyi(model=rag_conf["chat_model_name"])

class EmbeddingsFactory(BaseModelFactory):
    def generator(self) ->Optional[Embeddings|BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])

chat_model=ChatModelFactory().generator()
embed_model=EmbeddingsFactory().generator()
