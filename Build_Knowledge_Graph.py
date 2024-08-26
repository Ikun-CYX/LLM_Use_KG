from typing import Optional
from langchain_community.graphs import Neo4jGraph

from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import ChatZhipuAI
from langchain.llms import Tongyi

from langchain_experimental.graph_transformers import LLMGraphTransformer

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from tqdm import tqdm
import traceback
from parse_args import parse_args_build_knowledge_graph
from utils import Graph_Node_Relation
Graph_Node_Relation = Graph_Node_Relation()

class Build_Knowledge_Graph(object):
    def __init__(
            self,
            url: Optional[str] = None,
            username: Optional[str] = None,
            password: Optional[str] = None,
            database: Optional[str] = None,
            llm: Optional[str] = "Local",
            file_path: Optional[str] = None,
            error_print_path: Optional[str] = None,
            node_information_print_path: Optional[str] = None,
            whether_add_context: Optional[bool] = False,
    ) -> None:
        self.url = url
        self.username = username
        self.password = password
        self.database = database
        self.graph = Neo4jGraph(
            url=self.url,
            username=self.username,
            password=self.password,
            database=self.database
        )

        self.file_path = file_path
        self.error_print_path = error_print_path
        self.node_information_print_path = node_information_print_path

        if llm == "Local":
            self.llm = ChatOllama(model="qwen2:7b")
        elif llm == "ZhiPu":
            zhipuai_api_key = "e812ac1a0ece30cfb0153d9e2446612c.NmAoKm5xfwIHAj2H"
            self.llm = ChatZhipuAI(
                temperature=0,
                api_key=zhipuai_api_key,
                model="GLM-4-0520"
            )
        elif llm == "TongYi":
            self.llm = Tongyi()
        else:
            raise Exception("Invalid LLM")

        # LLMGraphTransformer模块 构建图谱
        self.llm_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=Graph_Node_Relation.allowed_nodes,
            allowed_relationships=Graph_Node_Relation.allowed_relationships,
        )
        self.add_context = whether_add_context

    def Process_Document(self):
        raw_documents = PyPDFLoader(self.file_path).load()

        # 将文本分割成每个包含20个tokens的块，并且这些块之间没有重叠
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=20, chunk_overlap=0
        )
        # Chunk the document
        documents = text_splitter.split_documents(raw_documents)
        return documents

    def Node_Extraction_And_Knowledge_Graph(self):

        documents = self.Process_Document()

        bar = tqdm(total=len(documents))
        for idx, doc in enumerate(documents):
            document = doc.copy()
            if self.add_context:
                if idx == 0:
                    document.page_content = doc.page_content + documents[idx+1].page_content
                elif idx == len(documents)-1:
                    document.page_content = documents[idx-1].page_content + doc.page_content
                else:
                    document.page_content = documents[idx-1].page_content + doc.page_content + documents[idx+1].page_content

        try:
            graph_documents_filtered = self.llm_transformer.process_response(document)

            graph_documents_filtereds = [graph_documents_filtered]
            with open(r'D:\demo\llama3_test_llama_index\node\node5.txt', 'a', encoding='utf-8') as f_err:
                for i, graph_doc in enumerate(graph_documents_filtereds):
                    print(f"Graph Document {idx}:")
                    f_err.write(f"Graph Document {idx}:")
                    f_err.write("\n")
                    print(f"Nodes:\n")
                    for j, node in enumerate(graph_doc.nodes):
                        print(f"Node[{j}]: {node}")
                        f_err.write(f"Node[{j}]: {node}")
                        f_err.write("\n")
                    for k, relationships in enumerate(graph_doc.relationships):
                        print(f"Relationships[{k}]: {relationships}")
                        f_err.write(f"Relationships[{k}]: {relationships}")
                        f_err.write("\n")
                    f_err.write("\n\n\n")

            self.graph.add_graph_documents(
                [graph_documents_filtered],
                baseEntityLabel=True,
                include_source=True
            )
        except:
            cc = "pdf的第" + str(idx) + "页有问题，请重新检查\n"
            print(cc)
            with open('D:\demo\llama3_test_llama_index\error\error_log.txt', 'a', encoding='utf-8') as f_err:
                f_err.write(cc)
                f_err.write("Exception occurred:\n")
                traceback.print_exc(file=f_err)
                f_err.write("\n")
        bar.update(1)
        bar.close()

def main():
    config = vars(parse_args_build_knowledge_graph())
    BKG = Build_Knowledge_Graph(
        url=config["url"],
        username=config["username"],
        password=config["password"],
        database=config["database"],
        llm=config["llm"],
        file_path=config["file_path"],
        error_print_path=config["error_print_path"],
        node_information_print_path=config["node_information_print_path"],
        whether_add_context=config["whether_add_context"],
    )
    BKG.Node_Extraction_And_Knowledge_Graph()

if __name__ == '__main__':
    main()