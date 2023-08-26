from langchain.prompts import PromptTemplate

base_recommender_template = """
Your role is a librarian who is an expert in recommending learning resources for students.
If the student did not elaborate on what subjects the recommendation should be,
respond by asking the student to 
"""

BASE_RECOMMENDER_PROMPT = PromptTemplate(template=base_recommender_template,
                                         input_variables=["question"])
