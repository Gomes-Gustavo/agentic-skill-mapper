from typing import List, Dict
from pydantic import BaseModel, Field, HttpUrl
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from tqdm.auto import tqdm



class LearningResource(BaseModel):
    """A model to represent a single, curated learning resource."""
    title: str = Field(description="The descriptive title of the resource.")
    url: HttpUrl = Field(description="The direct URL to the resource.")
    resource_type: str = Field(description="The type of the resource, e.g., 'Video', 'Article', 'Tutorial', 'Official Docs'.")

class LearningPlan(BaseModel):
    """A structured learning plan containing a list of resources for a single skill."""
    plan: List[LearningResource] = Field(description="A list of curated learning resources.")



def _find_resources_for_skill(skill_name: str) -> LearningPlan:
    """
    (Internal function) Finds a curated list of free learning resources for a
    single skill by querying the LLM.
    """
    parser = JsonOutputParser(pydantic_object=LearningPlan)
    
    prompt_template = """
    You are an expert curriculum developer and learning mentor for software engineers.
    Your task is to find the best free and stable online resources to learn a specific technical skill.

    For the skill "{skill_name}", find 3 to 4 high-quality, free-to-access learning resources.

    RULES:
    1.  **Prioritize Stable URLs**: Your knowledge has a cutoff date. To avoid 404 errors, prioritize links to high-level pages like the main documentation site (e.g., 'react.dev/learn'), a main tutorials page (e.g., 'kubernetes.io/docs/tutorials'), or a search query on a reputable platform (e.g., 'youtube.com/results?search_query=learn+terraform'). Avoid deep links to specific, obscure blog posts.
    2.  **Free Only**: All resources must be 100% free.
    3.  **Variety**: Provide a mix of resource types (e.g., 'Official Docs', 'Video Search', 'Tutorials Hub', 'Article').
    4.  **JSON Output**: You MUST return your response *only* as a valid JSON object. The main object must have a single key named "plan", which contains an array of resource objects.

    {format_instructions}
    """

    # We assume the Ollama server is running on the default local address
    llm = OllamaLLM(model="llama3:8b", base_url="http://127.0.0.1:11434", temperature=0.1)
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["skill_name"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | llm | parser

    try:
        result = chain.invoke({"skill_name": skill_name})
        return LearningPlan(**result)
    except Exception as e:
        
        print(f"Agent Error: Could not parse LLM output for skill '{skill_name}'. Details: {e}")
        return LearningPlan(plan=[])



def create_full_learning_plan(skills_to_learn: List[str]) -> Dict[str, List[Dict]]:
    """
    Generates a comprehensive learning plan for a list of skill gaps. This is the
    main entry point for the Learning Planner agent.

    Args:
        skills_to_learn: A list of skills the user needs to learn.

    Returns:
        A dictionary where each key is a skill and the value is a list of
        dictionaries, each representing a learning resource.
    """
    full_plan_dict = {}
    

    for skill in tqdm(skills_to_learn, desc="Generating Full Learning Plan"):
        resources_plan = _find_resources_for_skill(skill)
        
        if resources_plan and resources_plan.plan:
            
            full_plan_dict[skill] = [res.model_dump() for res in resources_plan.plan]
        else:
            full_plan_dict[skill] = []
            
    return full_plan_dict