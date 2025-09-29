from typing import List, Set

def find_skill_gaps(market_skills: List[str], user_skills: List[str]) -> List[str]:
    """
    Compares a list of required market skills against a list of a user's
    current skills to identify any gaps.
    """
    market_set: Set[str] = {skill.lower().strip() for skill in market_skills}
    user_set: Set[str] = {skill.lower().strip() for skill in user_skills}
    
    gap_set: Set[str] = market_set.difference(user_set)
    
    return sorted(list(gap_set))