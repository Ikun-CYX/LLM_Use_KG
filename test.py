from langchain.memory import ConversationBufferMemory
# 构建会话内存
memory = ConversationBufferMemory()
# 添加用户对话内容
memory.chat_memory.add_user_message("你好!")
# 添加机器对话内容
memory.chat_memory.add_ai_message("你好，有什么需要帮助的吗?")

print(memory.load_memory_variables({}))