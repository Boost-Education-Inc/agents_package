from langchain.prompts import ChatPromptTemplate


CONTENT_AGENT_ANSW_QUESTION_CONTENT_TEMPLATE = ChatPromptTemplate.from_template("""
        You are an expert explaining about the following content.
        You received the following question.
        
        ----
        Question:
        {question}
        
        Thus, given the following passages (Chunks of data) from a content (Book, class, etc.) return a deep and extensive answer to the question about the content, remarking the most important concepts and ideas.
        At the beginning of the summary, show the content's metadata(title, author, etc.)
        ----
        Content Chunks:
        {content_chunks}
        ----
        """)