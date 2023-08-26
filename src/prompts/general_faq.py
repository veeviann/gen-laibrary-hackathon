from langchain.prompts import PromptTemplate

general_faq_template = """You are a Librarian chatbot answering questions.
Use the following pieces of context to answer the question at the end.
If the question is unclear, or requires more clarification, prompt the user
to elaborate on the question.

{context}

Question: {question}
Helpful Answer:"""

GENERAL_FAQ_PROMPT = PromptTemplate(template=general_faq_template,
                                    input_variables=["context", "question"])

# Big idea
# Relatedness of questions
# Associated FAQ questions - are there similar words in the question?
# What questions come grouped together -> chat logs.
# What constitutes as a recommendation? How do you know
# that the recommendation is good?
