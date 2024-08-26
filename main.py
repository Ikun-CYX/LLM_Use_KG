from parse_args import parse_args_slotcraft_QA
from slotcraft_QA import Slotcraft_QA
def main():

    config = vars(parse_args_slotcraft_QA())

    QA = Slotcraft_QA(
        url=config["url"],
        username=config["username"],
        password=config["password"],
        database=config["database"],
        llm=config["llm"],
        whether_relevance_query=config["whether_relevance_query"],
        remember_historical_conversations=config["remember_historical_conversations"]
    )
    while True:
        question = input('请输入问题(输入quit退出对话)：')
        if question == "quit":
            break
        output = QA.questions_and_answers(question)
        print(output)

if __name__ == '__main__':
    main()