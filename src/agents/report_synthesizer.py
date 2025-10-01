from typing import List, Dict
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import json

def generate_report(
    target_role: str,
    user_skills: List[str],
    skill_gaps: List[str],
    learning_plan: Dict[str, List[Dict]]
) -> str:
    """
    Generates a final, human-readable career development report in Markdown format.
    """
    user_skills_str = ", ".join(user_skills)
    skill_gaps_str = ", ".join(skill_gaps)
    learning_plan_str = json.dumps(learning_plan, indent=2)

    prompt_template = """
    You are an expert career coach and AI assistant. Your task is to generate a personalized, encouraging, and actionable career development report in Markdown format.

    Here is the data for the user:
    - **Target Role**: {target_role}
    - **User's Current Skills**: {user_skills}
    - **Identified Skill Gaps**: {skill_gaps}
    - **Curated Learning Plan (JSON format)**:
    ```json
    {learning_plan}
    ```

    **REPORT STRUCTURE AND INSTRUCTIONS:**
    Generate a complete report following this exact Markdown structure. Be encouraging and professional.

    1.  **Header**: Start with a main title: `# Your Personalized Career Plan`.
    2.  **Introduction**: Write a short, personalized introduction. Address the user's goal of becoming a `{target_role}` and explain that this report outlines their strengths and a clear path forward.
    3.  **Your Current Strengths**: Create a section `## Your Current Strengths`. List the `User's Current Skills` in a Markdown bulleted list. Add a brief, positive sentence acknowledging their solid foundation.
    4.  **Your Development Roadmap**: Create a section `## Your Development Roadmap`. List the `Identified Skill Gaps` in a Markdown bulleted list.
    5.  **Your Personalized Learning Plan**: Create a main section `## Your Personalized Learning Plan`. For **each skill** found in the `Identified Skill Gaps` data, create a sub-header formatted like: `### Learning: [Skill Name]`. If there are resources for the skill, list them in a bulleted list formatted exactly like this: `- **[Resource Title](Resource URL)** - *Resource Type*`. If there are no resources, write a fallback sentence.
    6.  **Conclusion**: Create a final section `## Next Steps` with a short, motivating paragraph.

    Do not add any text before the `# Your Personalized Career Plan` header or after the conclusion.
    """

    llm = OllamaLLM(model="llama3:8b", base_url="http://127.0.0.1:11434", temperature=0.2)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["target_role", "user_skills", "skill_gaps", "learning_plan"]
    )
    chain = prompt | llm

    try:
        report_markdown = chain.invoke({
            "target_role": target_role,
            "user_skills": user_skills_str,
            "skill_gaps": skill_gaps_str,
            "learning_plan": learning_plan_str
        })
        return report_markdown
    except Exception as e:
        return f"# Error\n\nAn error occurred while generating the report: {e}"