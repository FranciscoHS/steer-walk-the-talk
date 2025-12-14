# Class for Anthropic Claude model
import json
import os

import anthropic

from language_models.model import Model
from language_models.utils import add_retries, limiter

# set anthropic config
api_key = os.environ.get("ANTHROPIC_API_KEY")

class Claude(Model):
    def __init__(self, name="claude-sonnet-4-5-20250929", max_tokens=256, temperature=0.7):
        """
        Args:
            name: name of the model
            temperature: temperature parameter of model
        """
        super().__init__(name)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = anthropic.Client()
    
    @add_retries
    @limiter.ratelimit('identity', delay=True)
    def generate_response(self, prompt, n_completions=1):
        """
        Generates a response to a prompt.
        Args:
            prompt: prompt to generate a response to
            n_completions: number of completions to generate
        Returns:
            response: response to the prompt
        """
        completions = []
        for _ in range(n_completions):
            response = self.client.messages.create(
                model=self.name, 
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
            {"role": "user", "content": prompt}
            ])
            completions.append(response.content[0].text)
        return completions