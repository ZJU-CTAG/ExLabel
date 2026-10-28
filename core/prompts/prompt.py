from pathlib import Path

class PromptUtils:
   
    @staticmethod
    def read_prompt(prompt_name: str, category: str = None) -> str:
        """
        Read prompt template from markdown file.
        
        Args:
            prompt_name: Name of the prompt file (without .md extension)
            category: Optional subdirectory under prompts/
            
        Returns:
            String content of the prompt file
            
        Raises:
            FileNotFoundError: If prompt file does not exist
        """
        base_path = Path(__file__).parent

        prompt_path = base_path / category / f"{prompt_name}.md" if category else base_path / f"{prompt_name}.md"
            
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
            
        with open(prompt_path) as f:
            return f.read()
    
    @staticmethod
    def format_prompt(template: str, **kwargs) -> str:
        """
        Format a prompt template with given keyword arguments.
        
        Args:
            template: The template text with {placeholders}
            **kwargs: Values to replace placeholders with
            
        Returns:
            Formatted prompt text
            
        Raises:
            ValueError: If required placeholder is missing
        """
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required placeholder in prompt: {e}")
    