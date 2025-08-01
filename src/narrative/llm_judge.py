import openai
from typing import Dict, List
import json

class LLMJudge:
    def __init__(self, config: Dict):
        self.model_name = config['model_name']
        self.max_tokens = config['max_tokens']
        self.temperature = config['temperature']
    
    def generate_narrative(self, user_profile: Dict, recommended_items: List[Dict]) -> str:
        prompt = self._create_prompt(user_profile, recommended_items)
        
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        
        return response.choices[0].message.content
    
    def _create_prompt(self, user_profile: Dict, items: List[Dict]) -> str:
        return f"""
        Generate a personalized narrative for user recommendation:
        User Profile: {json.dumps(user_profile)}
        Recommended Items: {json.dumps(items)}
        
        Create an engaging story explaining why these items match the user's preferences.
        """
    
    def evaluate_narrative_quality(self, narrative: str) -> Dict[str, float]:
        prompt = f"""
        Evaluate this narrative on a scale of 1-10:
        Narrative: {narrative}
        
        Rate: coherence, engagement, personalization, clarity
        Return JSON format.
        """
        
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1
        )
        
        return json.loads(response.choices[0].message.content)