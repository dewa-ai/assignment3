from openai import OpenAI
import json
import os
from typing import Optional, List, Dict, Literal
from json.decoder import JSONDecodeError
import logging

class InstructionManualGenerator:
    def __init__(
        self,
        openai_api_key: str,
        task_goal: str,
        results: List[Dict],
        logger: logging.Logger,
        instruction_format: Literal["text_steps", "json_blocks"] = "text_steps",
        openai_org_id: Optional[str] = None,
        api_model: str = "gpt-4o"
    ):
        """
        Initialize the instruction manual generator for WebVoyager tasks.

        Args:
            openai_api_key (str): OpenAI API key.
            task_goal (str): The task goal string (e.g., identifying the professor of a course).
            results (List[Dict]): A list of dictionaries containing retrieved results.
            logger: Logging object
            instruction_format (Literal["text_steps", "json_blocks"]): The desired output format for the manual.
                - "text_steps": Generates a human-readable step-by-step manual.
                - "json_blocks": Outputs a structured JSON manual with descriptions and sources.
            openai_org_id (Optional[str]): OpenAI organization ID.
            api_model (str): The OpenAI model to use for generating the manual.
        """
        self.openai_client = OpenAI(
            api_key=openai_api_key,
            organization=openai_org_id
        )
        self.task_goal = task_goal
        self.results = results
        self.instruction_format = instruction_format
        self.logger = logger
        self.system_message = "You are a professional technical document assistant. Your task is to generate clear, concise, and well-structured instruction manuals based on the provided information. Focus on actionable steps and maintain a logical flow in your instructions."
        self.api_model = api_model

    def _generate_prompt(self):
        """
        Generates the prompt for OpenAI's GPT model based on task goal and results.
        :return: The formatted prompt string.
        """
        prompt = f"""
You are a professional technical document assistant for WebVoyager, a web browsing agent. Your task is to filter relevant information from the provided retrieval results based on the given task goal and compile it into a structured instruction manual with actionable, numbered steps to guide the agent in completing the task.

### Task Goal:
{self.task_goal}

### Retrieval Results Example:
Each result contains:
- section: (The title information)
- content: (The information retrieved)
- source: (The source of the information)

### Relevance Criteria:
- The goal is to compile an **instruction manual** that provides actionable steps to achieve the task.
- A result is **relevant** if it:
  - Contains keywords or terminology directly related to any possible approach for completing the task goal
  - Includes step-by-step instructions, procedures, or operations that could contribute to task completion
  - Describes key functions, tools, or settings that might be useful for the task
  - Contains configuration details, system behaviors, or technical information that could aid in achieving the goal
  - Provides partial but useful information, even if it only addresses one aspect of the task
  - Mentions alternative methods or approaches that could accomplish the same goal
- A result is **not relevant** if it:
  - Contains no keywords or terminology related to any approach for completing the task
  - Provides only general theoretical concepts without practical application
  - Is completely unrelated to the task goal or any of its components

### Filtering Process:
1. **Identify Relevant Information**  
   - Consider whether the retrieved content helps in accomplishing the task through ANY possible approach
   - Even if the information describes just one possible method or only a portion of a method, include it
   - If a section contains even one relevant keyword or concept related to task completion, consider it relevant

2. **Structured Output**  
   - Organize the relevant information into a step-by-step instruction manual
   - Each step must be actionable, clearly described, and numbered sequentially
   - Use action-oriented language (e.g., "Click the search button," "Type 'query' into the textbox") to ensure clarity
   - If multiple methods are available, present them as alternative approaches with clear labels (e.g., "Method 1: Step 1")
   - For irrelevant results, provide a clear explanation of why they do not contribute to the task goal

### Output Format:
Return a string containing the structured manual with numbered steps. Each step should be concise and actionable. Format as follows:
```
Task Goal: {self.task_goal}
Steps:
1. [Actionable step description]
2. [Actionable step description]
...

source: [The source of the information]
```

### Example:
For a task like "Search for the latest news on climate change":
```
Task Goal: Search for the latest news on climate change
Steps:
1. Open your web browser and navigate to www.google.com.
2. Type 'climate change latest news' into the search bar and press Enter.
3. Click on a news article from a reputable source like BBC or Reuters.
```

### Retrieval Results
{json.dumps(self.results, ensure_ascii=False, indent=2)}

Please reason step by step and ensure the manual is structured with clear, actionable steps tailored for a web browsing agent.
"""

        if self.instruction_format == "json_blocks":
            prompt = f"""
You are a professional technical document assistant. Your task is to filter the relevant information from the provided retrieval results based on the given task goal and compile it into an instruction manual.

### Task Goal:
{self.task_goal}

### Retrieval Results Example:
Each result contains:
- section: (The title information)
- content: (The information retrieved)
- source: (The source of the information)

### Relevance Criteria:
- The goal is to compile an **instruction manual** that provides actionable steps to achieve the task.
- A result is **relevant** if it:
  - Contains keywords or terminology directly related to any possible approach for completing the task goal
  - Includes step-by-step instructions, procedures, or operations that could contribute to task completion
  - Describes key functions, tools, or settings that might be useful for the task
  - Contains configuration details, system behaviors, or technical information that could aid in achieving the goal
  - Provides partial but useful information, even if it only addresses one aspect of the task
  - Mentions alternative methods or approaches that could accomplish the same goal
- A result is **not relevant** if it:
  - Contains no keywords or terminology related to any approach for completing the task
  - Provides only general theoretical concepts without practical application
  - Is completely unrelated to the task goal or any of its components

### Filtering Process:
1. **Identify Relevant Information**
   - Consider whether the retrieved content helps in accomplishing the task through ANY possible approach
   - Even if the information describes just one possible method or only a portion of a method, include it
   - If a section contains even one relevant keyword or concept related to task completion, consider it relevant

2. **Structured Output**
   - Format relevant results in JSON, including the title, description, and source
   - For irrelevant results, provide a clear explanation of why they do not contribute to the task goal


### Retrieval Results
{json.dumps(self.results, ensure_ascii=False, indent=2)}

### Output Format:
Please output the results in the following JSON format:
```json
{{
    "manual": [
        {{
            "title": "Relevant Title",
            "description": "Operation steps filtered and compiled based on the task goal from the retrieved content",
            "source": "Source of the information"
        }}
    ],
    "irrelevant_explanations": [
        {{
            "section": "Title of the irrelevant section",
            "reason": "Explanation of why this result is not relevant"
        }}
    ]
}}
```
"""
        return prompt

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API with the given prompt."""
        try:
            # Split results into smaller chunks if they exceed the maximum token limit
            max_tokens = 4000  # Reduced chunk size for gpt-3.5-turbo
            chunks = []
            current_chunk = []
            current_length = 0
            
            # Split the prompt into sentences for better chunking
            sentences = prompt.split('. ')
            
            for sentence in sentences:
                # Rough estimate of tokens (4 chars ≈ 1 token)
                sentence_tokens = len(sentence) // 4
                
                if current_length + sentence_tokens > max_tokens:
                    chunks.append('. '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_tokens
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_tokens
            
            if current_chunk:
                chunks.append('. '.join(current_chunk))
            
            # Process each chunk separately
            responses = []
            for chunk in chunks:
                response = self.openai_client.chat.completions.create(
                    model=self.api_model,
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": chunk}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                responses.append(response.choices[0].message.content)
            
            # Combine responses
            return "\n\n".join(responses)
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")
            raise

    def generate_instruction_manual(self) -> str:
        """
        Generates a structured instruction manual by filtering relevant information from the retrieval results
        based on the defined task goal.

        This method works by:
        1. Generating a prompt using the task goal and retrieved content.
        2. Sending the prompt to the OpenAI API via `_call_openai()` to obtain a response.
        3. Handling the response based on the selected `instruction_format` (default: "text_steps"):
           - If `instruction_format` is "text_steps" (default), the method returns a free-form,
             step-by-step instruction manual directly from the model response.
           - If `instruction_format` is "json_blocks", the method parses the JSON response and converts each entry
             (including title, description, and source) into a readable manual string.

        Returns:
            str: A formatted instruction manual string, either as:
                - A step-by-step plain-text guide (if `instruction_format` is "text_steps"), or
                - A structured set of entries parsed from JSON, including title, description, and source (if `instruction_format` is "json_blocks").
        """
        prompt = self._generate_prompt()
        response_text = self._call_openai(prompt)

        if self.instruction_format == "json_blocks":
            try:
                response_text = response_text.replace("```json", "").replace("```", "")
                response = json.loads(response_text)
                manual_obj = response["manual"]

                manual_str = "\n\n".join(
                    f"title: {entry['title']}\ndescription: {entry['description']}\nsource: {entry['source']}"
                    for entry in manual_obj
                )
                return manual_str

            except JSONDecodeError as e:
                self.logger.warning(f"[JSONDecodeError] Failed to parse response: {e}")
            except (KeyError, TypeError) as e:
                self.logger.warning(f"[FormatError] Missing expected fields in JSON response: {e}")

            return ""

        else:
            return response_text


# Example Usage
if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    org_id = os.getenv("OPENAI_ORG_ID")

    task_goal = "查詢資訊工程學系碩士班的課程中，AI代理系統之設計與開發這門課的授課教授是誰?"
    results = [
        {"section": "Course Information",
         "content": "The course 'AI Agent System Design and Development' is taught by Professor Zhang.",
         "source": "University Course Announcement"},
        {"section": "University News", "content": "The university is promoting intelligent course development...",
         "source": "University News Website"},
        {"section": "Student Forum", "content": "Does anyone know who teaches the AI agent system course?",
         "source": "Student Forum"}
    ]

    # Instantiate the class and generate the manual
    manual_generator = InstructionManualGenerator(
        openai_api_key=api_key,
        openai_org_id=org_id,
        task_goal=task_goal,
        results=results,
        logger=logger
    )
    manual = manual_generator.generate_instruction_manual()

    # Print the resulting manual
    print(manual)
