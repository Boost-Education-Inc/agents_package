from langchain.prompts import ChatPromptTemplate


##memoria procedimental

####################----------TUTOR AGENT PROMPTS-----------####################

TUTOR_CONTEXT_TEMPLATE = ChatPromptTemplate.from_template("""
        Answer in english.
        In each answer, call the the user by his/her name.
        Respond in a friendly way.
        Use markdown to format the text.
        Adapt your response base on the user's data and the chat history.
        Use the following passages to answer the user's question.
        If the answer doesn't require the context just answer base on your knowledge, but, if you don't know the answer, just say that you don't know, don't try to make up an answer.
        ----
        User data:
        {student_data}
        ----
        Chat History:
        {chat_history}
        ----
        Content (Book, class, etc.):
        {context}
        ----
        Question: {question}
        """)


# TUTOR_PLAN_TEMPLATE = ChatPromptTemplate.from_template("""
#         A learning goal is a specific, measurable objective that outlines what an individual aims to learn or achieve over a certain period. It's often used in educational and professional development contexts to guide learning activities and track progress.
#         Answer in english.
#         Respond in a friendly way.
#         Adapt your response base on the user's data and the chat history.
#         Use the following passages and data to create an extensive list of learning goals along with ways to measure the learning goal, and strategies to achieve it to accomplish each one.
#         Use markdown to format the text.
#         Just return the markdown table.
#         ----
#         User data:
#         {student_data}
#         ----
#         Chat History:
#         {chat_history}
#         ----
#         Content (Book, class, etc.):
#         {context}
#         """)


TUTOR_PLAN_TEMPLATE = ChatPromptTemplate.from_template("""
        A learning goal is a specific, measurable objective that outlines what an individual aims to learn or achieve over a certain period. It's often used in educational and professional development contexts to guide learning activities and track progress
        Answer in english
        Respond in a friendly way
        Adapt your response base on the user's data and the chat history
        Use the following passages and data to create a 6 nodes graph of learning goals as a learning path
        Do not add any text the graph edges
        The graph must have an start and an end node
        The graph must be create with mermaid
        Each node must have just one child node
        The graph must be a TD(Top down) graph
        Do not add any style to the graph and nodes, just make sure all nodes are circular. Hint: To make a node circular make sure to sorround the node like this: A((Hi))
        Just return the Html code that is inside the <pre class="mermaid">
        Do not return anything else besides the HTML code
        Your response must be in a single line of text that be interpreted as HTML code
        ----
        User data:
        {student_data}
        ----
        Chat History:
        {chat_history}
        ----
        Content (Book, class, etc.):
        {context}
        """)



# TUTOR_PRESENTATION_TEMPLATE = ChatPromptTemplate.from_template("""
#         Answer in english
#         Just return the HTML code that is inside the <div class="swiper" style="width: 40vw; height: 60vh; border-radius: 10px; margin-top:5px;">
#         Do not return anything else besides the HTML code
#         Avoid line jumps in the HTML code
#         Avoid adding an init label 'html' to your response
#         Your response must be in a single line of text that be interpreted as HTML code
        
#         Just use the following clases : swiper-wrapper, swiper-slide, .swiper-pagination, .swiper-button-next, .swiper-button-prev, .swiper-scrollbar
#         Use different font colors, background colors, aligments, images, tables, markdown and use the font: Arial.
#         Do not change the width of the slides.
#         Do not add padding or margin to the slides.
#         Create a beatiful, organized, visually attractive and clear presentation.
#         Adapt your response base on the user's data and the chat history.
#         Use the following passages and data to create a multiple slides presentation about the content using swiper.     
#         ----
#         User data:
#         {student_data}
#         ----
#         Chat History:
#         {chat_history}
#         ----
#         Content (Book, class, etc.):
#         {context}
#         """)

# TUTOR_PRESENTATION_TEMPLATE = ChatPromptTemplate.from_template("""
#         Take a deep breath and solve the following problem step by step:
        
#         Create a beatiful, organized, visually attractive, clear and 10 slides presentation about the content using swiper and base on the following indications:
#         ----
#         1) Answer in english
#         2) Just return the HTML code that is inside the <div class="swiper" style="width: 40vw; height: 50vh; border-radius: 10px;">
#         3) Do not return anything else besides the HTML code
#         4) Your response must be in a single line of text that be interpreted as HTML code
#         ----
#         5) Avoid line jumps in the HTML code
#         6) Avoid adding an init label 'html' to your response
#         ----
#         7) Use the following clases : swiper-wrapper, swiper-slide, .swiper-pagination, .swiper-button-next, .swiper-button-prev, .swiper-scrollbar
#         8) Use different font colors, background colors, aligments, background images, images divs, tables and markdown style.
#         9) Use the font: Arial in all the slides.
#         10) Do not change the width of the slides.
#         11) The .swiper-button-next and .swiper-button-prev must be black.
#         12) Do not add padding or margin to the slides.
#         ----        
#         13) Adapt your response base on the user's data and the chat history.
#         14) Use the following passages and data to enrich the presentation:
#         ----
#         User data:
#         {student_data}
#         ----
#         Chat History:
#         {chat_history}
#         ----
#         Content (Book, class, etc.):
#         {context}
#         """)


TUTOR_PRESENTATION_TEMPLATE = ChatPromptTemplate.from_template("""
        Create a beatiful, creative, visually attractive and clear 6 slides presentation about the content using swiper
        Answer in english
        Just return the HTML code that is inside the <div class="swiper" style="width: 40vw; height: 50vh; border-radius: 10px;">
        Use the following clases : swiper-wrapper, swiper-slide, .swiper-pagination, .swiper-button-next, .swiper-button-prev, .swiper-scrollbar
        Use different font colors, font styles, background colors, background images, aligments of elements and markdown style
        Use the font: Arial in all the slides
        Make sure the text has a different font color than the background
        In each slide add a title, a subtitle and and a paragraph
        Do not change the width of the slides
        The .swiper-button-next and .swiper-button-prev must be black
        Do not add padding or margin to the slides
        Do not return anything else besides the HTML code
        Your response must be in a single line of text that be interpreted as HTML code
        Explain the content(Book, class, etc) and also adapt it base on the thinks the user likes, the user's goals
        Do not include user's personal data inside the slides (name, email,age, etc)
        ----
        User data:
        {student_data}
        ----
        Chat History:
        {chat_history}
        ----
        Content (Book, class, etc.):
        {context}
        """)


TUTOR_PRESENTATION_SCRIPT_TEMPLATE = ChatPromptTemplate.from_template("""
        Given the following html code of a swiper presentation
        Return a short script for a person to read and explain what is inside the presentation
        Just describe the text inside each slide
        Do not describe the graphics, images, tables, colors, etc
        Start saying welcome to boost education.
        Do not mention swiper or any other library.
        ----
        Swiper presentation HTML code:
        {presentation_html}
        ----
        """)



####################----------CONTENT AGENT PROMPTS-----------####################

CONTENT_AGENT_SUMMARY_CONTENT_TEMPLATE = ChatPromptTemplate.from_template("""
        You are an expert explaining about the following content.
        Thus, given the following passages (Chunks of data) from a content (Book, class, etc.) return a deep and extensive summary of the content, remarking the most important concepts and ideas.
        At the beginning of the summary, show the content's metadata(title, author, etc.)
        ----
        Content Chunks:
        {content_chunks}
        ----
        """)



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