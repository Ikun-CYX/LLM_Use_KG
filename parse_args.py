import argparse

def parse_args_slotcraft_QA():
    parser = argparse.ArgumentParser()
    # Neo4j
    parser.add_argument('--url', default="bolt://localhost:7687", type=str, help='')
    parser.add_argument('--username', default="neo4j", type=str,  help='')
    parser.add_argument('--password', default="12345678", type=str, help='')
    parser.add_argument('--database', default="neo4jen", type=str, help='')
    parser.add_argument('--llm', default="ZhiPu", type=str, help='')

    parser.add_argument('--whether_relevance_query', action='store_false', help='')
    parser.add_argument('--remember_historical_conversations', action='store_false', help='')
    config = parser.parse_args()
    return config

def parse_args_build_knowledge_graph():
    parser = argparse.ArgumentParser()
    # Neo4j
    parser.add_argument('--url', default="bolt://localhost:7687", type=str, help='')
    parser.add_argument('--username', default="neo4j", type=str,  help='')
    parser.add_argument('--password', default="12345678", type=str, help='')
    parser.add_argument('--database', default="neo4j2", type=str, help='')

    parser.add_argument('--llm', default="local", type=str, help='')
    parser.add_argument('--file_path', default=r".\slotcraft_datasets\Original_Document\Slotcraft逻辑编辑文档V2.pdf", type=str, help='')
    parser.add_argument('--error_print_path', default=r".\error\error_log.txt.pdf", type=str, help='')
    parser.add_argument('--node_information_print_path', default=r".\node\node5.txt.pdf", type=str, help='')

    parser.add_argument('--whether_add_context', action='store_true', help='')
    config = parser.parse_args()
    return config