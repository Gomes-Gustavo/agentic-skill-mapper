from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

# -- Pydantic Model for Structured Output --
# Defines the exact data structure we want the LLM to return.
# This ensures the output is consistent, predictable, and easy to work with.

class SkillSet(BaseModel):
    """Output model representing the extracted skills from a job description."""
    technical_skills: List[str] = Field(description="A comprehensive list of specific technical skills, e.g., 'Python', 'PyTorch', 'AWS S3', 'SQL', 'Git'.")
    soft_skills: List[str] = Field(description="A list of soft or behavioral skills, e.g., 'Teamwork', 'Agile Methodologies', 'Problem-solving', 'Communication'.")



def analyze_job_description(job_description_text: str) -> SkillSet:
    """
    Analyzes a job description using a local LLM to extract technical and soft skills.
    
    This function connects to a local Ollama server, sends the job description
    with a structured prompt, and parses the JSON output into a Pydantic model.
    
    Args:
        job_description_text: The full text of the job description to be analyzed.
        
    Returns:
        A SkillSet object containing the lists of extracted skills.
    """
    # Initialize the LLM connection.
    # Temperature=0 makes the output more deterministic and less "creative".
    llm = OllamaLLM(model="llama3:8b", temperature=0)

    # The parser will automatically convert the LLM's JSON string output
    # into our structured SkillSet Pydantic object.
    parser = JsonOutputParser(pydantic_object=SkillSet)

    # The prompt template is the core instruction for the LLM.
    # It defines its role, the task, rules, and the expected output format.
    prompt_template = """
    You are an expert AI assistant specialized in tech recruitment and HR analytics.
    Your primary task is to meticulously analyze the following job description and extract all relevant technical and soft skills.

    Follow these rules precisely:
    1.  **Extract, do not infer**: Only list skills that are explicitly mentioned or very strongly implied in the text.
    2.  **Be Specific**: Avoid overly generic terms. For example, prefer 'AWS S3' over 'Cloud Storage'.
    3.  **Normalize Skills**: Use the canonical name for technologies (e.g., "Python" not "python programming", "PyTorch" not "Pytorch framework").
    4.  **Format**: You MUST return your response *only* as a valid JSON object that adheres to the provided schema. Do not add any introductory text, explanations, or markdown formatting around the JSON.

    {format_instructions}

    JOB DESCRIPTION TEXT TO ANALYZE:
    ---
    {job_description}
    ---
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["job_description"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # The LCEL (LangChain Expression Language) chain pipes the components together.
    # It's a sequence of operations: format the prompt -> send to LLM -> parse the output.
    chain = prompt | llm | parser

    print("Market_Analyst Agent: Processing text...")
    try:
        # The invoke method runs the chain with the provided input.
        result = chain.invoke({"job_description": job_description_text})
        print("Market_Analyst Agent: Processing complete.")
        return SkillSet(**result)
    except Exception as e:
        print(f"Market_Analyst Agent: An error occurred during analysis: {e}")
        # Return an empty SkillSet in case of an error to prevent crashes.
        return SkillSet(technical_skills=[], soft_skills=[])