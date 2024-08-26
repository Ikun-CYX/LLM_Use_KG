import ast
from typing import Optional
from langchain_community.graphs import Neo4jGraph

from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import ChatZhipuAI
from langchain.llms import Tongyi
from langchain.prompts import ChatPromptTemplate

from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory

class Slotcraft_QA(object):
    def __init__(
            self,
            url: Optional[str] = None,
            username: Optional[str] = None,
            password: Optional[str] = None,
            database: Optional[str] = None,
            llm: Optional[str] = "Local",
            whether_relevance_query: Optional[bool] = True,
            remember_historical_conversations: Optional[bool] = True,
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
        if llm == "Local":
            self.llm = ChatOllama(model="llama3.1")
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
        self.Whether_relevance_query = whether_relevance_query
        self.Remember_Historical_Conversations = remember_historical_conversations
        if self.Remember_Historical_Conversations:
            self.memory = ConversationBufferMemory(memory_key="customize_chat_history")

    def output_structured_extraction(self, entities) -> str:
        if not isinstance(entities, str):
            entities = entities.content
        return entities
    def Extract_Keywords(self, question: str) -> str:
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "您是一位提取关键词语的高手，正在从以下给的的句子中提取出关键的词语，并且给出关键词的英文"
                    "同时还要给出每个词语的英文"
                    "请以列表的格式输出"
                    "例如，我给你一个输入“分支相关的组件有哪些？”"
                    "最理想的输出是：['分支','组件','branch','component']"
                    "重要说明：不要添加任何解释和文字!!。",
                ),
                (
                    "human",
                    "使用给定的格式从以下内容中提取信息："
                    "输入：{question}。"
                    "输入：",
                ),
            ]
        )
        entity_chain = chat_prompt | self.llm
        entities = entity_chain.invoke({"question": question})
        entities = self.output_structured_extraction(entities)
        print(entities)
        return entities

    def generate_full_text_query(self, input: str) -> str:
        """
        Generate a full-text search query for a given input string.

        This function constructs a query string suitable for a full-text search.
        It processes the input string by splitting it into words and appending a
        similarity threshold (~2 changed characters) to each word, then combines
        them using the AND operator. Useful for mapping entities from user questions
        to database values, and allows for some misspelings.
        """
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()

    def structured_retriever(self, entities: str,
                                   question: str) -> str:
        """
        Collects the neighborhood of entities mentioned
        in the question
        """
        result = ""
        entities = ast.literal_eval(entities)
        query = """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node,score
                CALL {
                  WITH node
                  MATCH (node)-[r:!MENTIONS]->(neighbor)
                  RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                  UNION ALL
                  WITH node
                  MATCH (node)<-[r:!MENTIONS]-(neighbor)
                  RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 50
                """
        for entity in entities:
            response = self.graph.query(query,{"query": entity},)
            result += "\n".join([el['output'] for el in response])
        response = self.graph.query(query, {"query": question}, )
        result += "\n".join([el['output'] for el in response])
        return result

    def relevance_query(self, structured_data: str,
                              question: str) -> str:

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "您是一位提取相关信息的高手，我会给您一个问题和若干对三元组"
                    "您的任务是从我给出的若干对三元组中寻找出相关问题最贴近的二十对左右三元组"
                    "您输出的相关三元组务必严格遵守，我给您的若干三元组相同的格式"
                    
                    "如果三元组内容，和问题根本毫不相关，请输出未查询到与问题相关的三元组，同时我也希望您不要胡说八道。"
                    "最后用中文输出。"
                    
                    "重要说明：不要添加任何解释和文字。",
                ),
                (
                    "human",
                    "使用给定的格式从以下内容中提取信息，输入如下。"
                    "问题：{question}。"
                    "输入三元组对：{structured_data}。",
                ),
            ]
        )
        entity_chain = chat_prompt | self.llm
        entities = entity_chain.invoke({
            "question": question,
            "structured_data": structured_data
        })
        entities = self.output_structured_extraction(entities)
        return entities

    def retriever(self, question: str,
                        Whether_relevance_query: bool) -> str:
        print(f"Search query: {question}")
        entities = self.Extract_Keywords(question)
        structured_data = self.structured_retriever(entities, question)
        print(structured_data)

        if Whether_relevance_query == True:
            structured_data = self.relevance_query(structured_data, question)
        print(entities)
        final_data = f"""Structured data:
        {structured_data}
        """
        print(f"final_data: {final_data}")
        return final_data

    def questions_and_answers_without_memory(self, question: str) -> str:

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "slotcraft是一个用来快速制作slot游戏的计算机软件系统"
                    "而您我的朋友，您精通slotcraft的使用，同时又很会设计slot游戏（老虎机游戏）"
                    "我是个爱好开发slot游戏的人，但是对于slotcraft的使用又不是那么熟悉，"
                    "希望您能认真分析我给出关于slotcraft三元组对上下文，然后回答我的问题"
                    "我给您的三元组可以这么解读，例：“ChgSymbolVals的sourceType - TYPE - SourceType”,"
                    "其中‘“ChgSymbolVals的sourceType’为头实体，“SourceType”是尾实体，“TYPE”是关系"
                    "所以这个三元组的意思是：ChgSymbolVals的sourceType类型是SourceType"

                    "如果三元组中上下文内容，和问题根本毫不相关，请不要参考三元组作答，同时我也希望您不要胡说八道。"
                    "最后用中文输出。"
                    "重要说明：不要添加任何解释和文字。",
                ),
                (
                    "human",
                    "输入如下。"
                    "三元组对上下文：{context}。"
                    "问题：{question}。"
                    "回答:",
                ),
            ]
        )

        chain = prompt | self.llm | StrOutputParser()
        output = chain.invoke({
            "context": self.retriever(question, self.Whether_relevance_query),
            "question": question
        })
        output = self.output_structured_extraction(output)
        return output

    def questions_and_answers_with_memory(self, question: str) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "slotcraft是一个用来快速制作slot游戏的计算机软件系统"
                    "而您我的朋友，您是一位精通slotcraft的使用的对话机器人，同时又很会设计slot游戏（老虎机游戏）"
                    "我是个爱好开发slot游戏的人，但是对于slotcraft的使用又不是那么熟悉，"
                    "希望您能认真分析我给出关于slotcraft三元组对上下文，然后回答我的问题"
                    "我给您的三元组可以这么解读，例：“ChgSymbolVals的sourceType - TYPE - SourceType”,"
                    "其中‘“ChgSymbolVals的sourceType’为头实体，“SourceType”是尾实体，“TYPE”是关系"
                    "所以这个三元组的意思是：ChgSymbolVals的sourceType类型是SourceType"
                    
                    "如果三元组中上下文内容，和问题根本毫不相关，请不要参考三元组作答，同时我也希望您不要胡说八道。"
                    "最后用中文输出。"
                    
                    "以下<history>标签中是我俩的历史对话记录，您可以参考历史对话记录"
                    "历史对话:<history>{customize_chat_history}</history>"
                    
                    "重要说明：不要添加任何解释性文字。",
                ),
                (
                    "human",
                    "输入如下。"
                    "三元组对上下文：{context}。"
                    "问题：{question}。"
                    "回答:",
                ),
            ]
        )
        print(self.memory.load_memory_variables({}))
        chain = prompt | self.llm | StrOutputParser()
        output = chain.invoke({
            "context": self.retriever(question, self.Whether_relevance_query),
            "question": question,
            "customize_chat_history": self.memory.load_memory_variables({})
        })
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(output)

        return output

    def questions_and_answers(self, question: str) -> str:
        if self.Remember_Historical_Conversations:
            return self.questions_and_answers_with_memory(question)
        else:
            return self.questions_and_answers_without_memory(question)