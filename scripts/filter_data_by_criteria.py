import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor
import json
from google import genai

from google.genai import types
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

class Gemini:
    def __init__(self, model_name, api_key=None):
        self.model_name = model_name
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        print(f"Using model: {model_name}")
        
    def call_api(self, user_prompt: str, system_prompt: str = None, **kwargs) -> str:
        """
        Makes call to Gemini model, returns just the string response.

        Args:
            user_prompt (str): Contents of user prompt  

        Returns:
            str: The response text
        """
        config_data = {"system_instruction": system_prompt} if system_prompt else {}
        config_data.update(kwargs)
        
        response = self.client.models.generate_content(
            model=self.model_name,
            config=types.GenerateContentConfig(
                **config_data
            ),
            contents=[user_prompt]
        )
        
        return response.text
    
    @retry(wait=wait_fixed(5), stop=stop_after_attempt(3))
    def __call__(self, user_prompt: str, system_prompt:str = None, max_output_tokens=1000) -> str:
        return self.call_api(user_prompt=user_prompt, system_prompt=system_prompt, max_output_tokens=max_output_tokens)
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data file.")
    parser.add_argument("--filter_data", action='store_true', help="Enable filtering of questions with LLMs.")
    parser.add_argument("--model_name", type=str, default="gemini-2.0-flash", help="Name of the Gemini model to use.")
    parser.add_argument("--api_key", type=str, required=False, help="Google AI API key. If not provided, looks for GEMINI_API_KEY environment variable.")
    parser.add_argument("--num_process", type=int, default=10, help="Number of parallel processes.")
    parser.add_argument("--limit_num", type=int, help="Limit the number of processed items.")
    return parser.parse_args()

def extract_bracket_content(text:str) -> str:
    " Extracts content between brackets from LLM response."
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else None

def parse_gemini_response(response: str) -> tuple[bool, str]:
    " Parses the response from the LLM."
    try:
        if not response.startswith('{'):    
            response = extract_bracket_content(response)
        parsed_data = json.loads(response.replace('\n', ''))
        
        assert len(parsed_data) == 4, "Response JSON must contain exactly 4 keys."
        assert all(key in parsed_data for key in ["facts", "legal_issues", "conclusion", "reasons"]), "Response JSON must contain keys: facts, legal_issues, conclusion, reasons."
        assert isinstance(parsed_data["facts"], str), "Facts must be a string."
        assert isinstance(parsed_data["legal_issues"], list), "Legal issues must be a list."
        assert isinstance(parsed_data["conclusion"], str), "Conclusion must be a string."
        assert isinstance(parsed_data["reasons"], list), "Reasons must be a list."
        
        return True, parsed_data
    
    except Exception as e:
        print(f"Error parsing response: {e}")
        return False, None
    
    
def process_case_item(item, gemini_instance, save_directory, filter_prompt, reformat_prompt, filter_enabled):
    """
    Processes a single case item from the input data.
    """
    _MAX_RETRIES = 2
    try:
        save_path = os.path.join(save_directory, f"{item['id']}.json")
        
        # Get the case text
        case_text = '\n\n'.join([
            f"Name: {item['name']}",
            f"Decision date: {item['decision_date']}",
            f"Court: {item['court']}",
            f"{item['text']}"
        ])
        case_text = f"<case text>{case_text}</case text>"
        
        # Filter questions if filtering is enabled
        if filter_enabled:
            filter_response = gemini_instance(user_prompt=case_text, system_prompt=filter_prompt)
            item['filter_response'] = filter_response
            
            if 'pass' not in filter_response.lower():
                with open(save_path, 'w', encoding='utf-8') as file:
                    json.dump(item, file, ensure_ascii=False, indent=2)
                return 1
            
        # Reformat the case text
        for _ in range(_MAX_RETRIES):
            reformat_response = gemini_instance(user_prompt=case_text, system_prompt=reformat_prompt)
            item['reformat_response'] = reformat_response
            valid, parsed_data = parse_gemini_response(reformat_response)
            
            if valid:
                item["facts"] = parsed_data["facts"]
                item["legal_issues"] = parsed_data["legal_issues"]
                item["conclusion"] = parsed_data["conclusion"]
                item["reasons"] = parsed_data["reasons"]
                break
            
        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(item, file, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Error processing item {item['id']}: {e}")
    
    return 1
        
def merge_saved_files(directory):
    "Merges all saved JSON files in the directory."
    _, _, filenames = next(os.walk(directory))
    json_files = [f for f in filenames if f.endswith('.json')]
    merged_data = []
    
    for file in json_files:
        try:
            with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                assert 'conclusion' in data or 'filter_response' in data, f"Missing conclusion or filter_response in {file}"
                merged_data.append(data)
        except Exception as e:
            print(f"Error merging file {file}: {e}")
    return merged_data

def deduplicate_data(data, processed_data):
    "Deduplicates the data based on the processed data."
    processed_ids = {item['id'] for item in processed_data}
    return [item for item in data if item['id'] not in processed_ids]

filter_system_prompt = """You are an expert in filtering and evaluating legal case text and legal reasoning. Your job is to evaluate a given case and determine whether it meets the following criteria:
1. **Depth of Reasoning:** The case should include detailed reasoning. If the case apepars too simple, mark it as "Too Simple".
2. **Unambiguous Decision:** The case should have a clear and unambiguous conclusion, decision, or holding. If the conclusion is ambiguous, mark it as "Ambiguous Decision".
3. **Facts Presented:** Case should present facts that are important to the ultimate decision of the court. The factual background should be sufficient for a reader to learn about the case or controversy between the parties and why the facts are legally relevant. If the case does not contain enough facts or the facts are not relevant to the legal reasoning, mark it as "Insufficient Facts".
4. **Legal Issues:** The case should explicitly delinate the legal issues involved. If the legal issues are not clear, mark it as "Unclear Legal Issues".

For each case, answer with one of the following labels:
- "pass" (if the case meets all criteria)
- "Too Simple"
- "Ambiguous Decision"
- "Insufficient Facts"
- "Unclear Legal Issues"
"""

reformat_system_prompt = """I will provide you with a legal case text. Your task is to rewrite it and separate it into three parts: the facts of the case, the legal issues, and the conclusion. The requirements are:

1. **Facts of the Case:** Provide the facts of the case, which refer to the events that led to the legal dispute. The facts are objective and should not include any reasoning or conclusions. They refer to the parties and the case or controversy between them. The facts should be fully encompassing and detailed, giving enough context to fully understand the case background.
2. **Legal Issues:** Identify the legal issue(s) involved in the case. Legal issues are the questions of law that are presented for determination by the court. They are the legal questions that the court must answer in order to resolve the dispute between the parties. The legal issues each must be no more than one sentence. They should be in an array format.
3. **Conclusion:** Provide the conclusion of the case, which is the final decision or holding of the court. The conclusion must be clear, unambiguous, and no more than one sentence.
4. **Reasons:** Provide the court's rationale for its decision as an array of single sentences. Each sentence must be comprehensive and detailed, explaining a distinct reason. Avoid summarizing multiple reasons within a single sentence.

Please output the result in the following JSON format:
```json
{
  "facts": "...",
  "legal_issues": "[...]",
  "conclusion": "",
  "reasons": "[...]"
}
```"""

# Detailed prompt for reasons with examples, if needed
# List the court's reasons for its decision in an array. Each element in the array must be a single, comprehensive, and detailed sentence explaining one distinct reason. Avoid summarizing multiple reasons within a single sentence. For example: ["The defendant failed to provide sufficient evidence.", "The plaintiff's testimony was deemed credible.", "The applicable law supports the plaintiff's claim."]

def main():
    args = parse_arguments()
    
    # Load input data
    with open(args.data_path, 'r', encoding='utf-8') as file:
        data = []
        for line in file:
            data.append(json.loads(line))
        
    
    # Get dataset name and save directory
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    save_directory = os.path.join('output_data', dataset_name)
    os.makedirs(save_directory, exist_ok=True)
    
    # Initialize Gemini instance
    gemini_instance = Gemini(model_name=args.model_name, api_key=args.api_key)
    
    # Merge previously proceassed data
    processed_data = merge_saved_files(save_directory)
    print(f"Previously processed {len(processed_data)} items.")
    
    input_data = deduplicate_data(data, processed_data)
    print(f"Processing {len(input_data)} items.")
    
    # Apply limit after deduplication
    if args.limit_num:
        input_data = input_data[:args.limit_num]
    print(f"Limited to {len(input_data)} items.")
    
    # Process data using a thread pool
    with ThreadPoolExecutor(max_workers=args.num_process) as executor:
        list(tqdm(executor.map(lambda item: process_case_item(item, 
                                                              gemini_instance, 
                                                              save_directory, 
                                                              filter_system_prompt, 
                                                              reformat_system_prompt,
                                                              args.filter_data), 
                               input_data), 
                  total=len(input_data), 
                  desc="Processing Items", 
                  unit="item"))
        
    # Merge and save final output
    final_data = merge_saved_files(save_directory)
    os.makedirs('final_output', exist_ok=True)
    output_path = f"final_output/{dataset_name}_final_{len(final_data)}.json"
    print(f"Processed {len(final_data)} items. Saving to {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as file:
        for item in final_data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')

if __name__ =='__main__':
    main()