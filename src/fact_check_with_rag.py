import re
import json
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.schema import Document
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
import timeit
import threading
import argparse
import gc


def analyze_fact_check(fact_check_content, model):
    if fact_check_content is None:
        return "There are no fact check content."

    info = []
    time_start = timeit.default_timer()

    for i in fact_check_content[1]:
        # Initialize LLM with lower temperature for consistent output and limit token generation
        llm = OllamaLLM(model=model, temperature=0.1, num_predict=4096)

        # Define PromptTemplate
        template = """
        You are an evidence extraction and summarization assistant.
        Given the following documents, your task is to:

        1. Identify and extract only the key factual statements, data points, and named entities.
        2. Preserve the exact meaning and facts as stated in the source; do not add interpretations or assumptions.
        3. Remove irrelevant or repetitive information.
        4. Organize the extracted facts into a concise, coherent summary.
        5. Maintain a neutral and objective tone.

        Documents:
        {context}
        """
        prompt = PromptTemplate.from_template(template)

        # Format prompt
        formatted_prompt = prompt.format(context=i)

        # Execute LLM
        result = llm.invoke(formatted_prompt)

        # Combine result with URL
        info.append(f"{result}, url: {i['metadata']}")

        del llm
        gc.collect()

    time_end = timeit.default_timer()

    # Convert results to Document format
    documents = [Document(page_content=item) for item in info]

    return documents, time_end - time_start

class FactCheckOutput(BaseModel):
    label: Literal["Supported", "Refuted", "Not Enough Information"] = Field(
        description="The fact-check verdict. Must be one of: Supported, Refuted, or Not Enough Information"
    )
    language: str = Field(description="Language of the claim and context")
    date: str = Field(description="Date in YYYY-MM-DD format")
    country: str = Field(description="ISO 3166-1 alpha-2 country code")
    url: list[str] = Field(description="List of source URLs used for evaluation")
    reasoning: str = Field(description="Complete reasoning process explaining the verdict")
    
def fact_check(query, documents, model, analyzer_time=None):
    if not documents:
        return

    # Initialize LLM with lower temperature for consistent output and limit token generation
    llm = OllamaLLM(model=model, temperature=0.1, num_predict=4096)

    # Use PydanticOutputParser
    parser = PydanticOutputParser(pydantic_object=FactCheckOutput)

    # Adjust the prompt to include explicit reasoning process
    template = """
    You are a professional fact-checker tasked with evaluating the following claim.
    Let's break down the evidence and reasoning step by step.
    First, analyze the provided context {context} and identify key information relevant to the claim {claim}.
    Then, evaluate the evidence step by step to determine if the claim is true or false.
    Ensure your response includes the URL(s) of the source(s) you used for evaluation.

    You must respond with a JSON object matching this structure:
    {{
        "label": "Supported" or "Refuted" or "Not Enough Information",
        "language": "language code",
        "date": "YYYY-MM-DD",
        "country": "ISO 3166-1 alpha-2 country code",
        "url": ["URL1", "URL2", ...],
        "reasoning": "your complete reasoning here"
    }}

    Label Definitions:
    - Supported: The claim is accurate and supported by evidence. Use this when the claim is True or Mostly True (largely accurate with only minor inaccuracies that don't change the fundamental meaning).
    - Refuted: The claim is inaccurate and contradicted by evidence. Use this when the claim is False or Mostly False (contains some truth but the overall statement is misleading or incorrect).
    - Not Enough Information: Insufficient evidence to make a determination, or the claim is Half True (partially accurate but missing critical context that makes it neither clearly supported nor refuted).

    Guidelines:
    - If the claim's core assertion is correct despite minor details being wrong, choose "Supported"
    - If the claim's core assertion is incorrect despite some minor details being right, choose "Refuted"
    - Only use "Not Enough Information" when evidence is truly insufficient or the claim is equally true and false

    IMPORTANT: Return ONLY the JSON data, not the schema definition.
    Do not include any text before or after the JSON object.
    """
    prompt = PromptTemplate.from_template(template)
    formatted_prompt = prompt.format(context=documents, claim=query)

    time_start = timeit.default_timer()
    result = llm.invoke(formatted_prompt)
    time_end = timeit.default_timer()
    del llm
    gc.collect()
    try:
        # Clean up possible schema definitions
        if '"$defs"' in result or '"properties"' in result:
            lines = result.split('\n')
            result = '\n'.join([line for line in lines if '"$defs"' not in line and '"properties"' not in line and '"required"' not in line])
        
        start_index = result.index('{')
        end_index = result.rindex('}')
        result_json = result[start_index:end_index + 1]
        
        # Try to parse JSON directly
        parsed_json = json.loads(result_json)
        
        # Check if it's a schema definition rather than actual data
        if "$defs" in parsed_json or ("properties" in parsed_json and "label" not in parsed_json):
            print("Warning: LLM returned schema instead of data")
            return None, None, 0
        
        parsed_result = parser.parse(result_json)
        if analyzer_time is None:
            return result, parsed_result.model_dump(), time_end - time_start
        else:
            return result, parsed_result.model_dump(), time_end - time_start + analyzer_time
    except Exception as e:
        return None, None, 0
    
def run_with_timeout(func, args=(), kwargs={}, timeout=60):
    result = [None]  # To store the function result
    exception = [None]  # To capture any exception

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)  # Wait for execution to complete

    if thread.is_alive():  # Check if the function timed out
        print("Function execution timed out.")
        return None
    if exception[0]:  # Check for any exceptions
        print(f"Error during execution: {exception[0]}")
        return None
    return result[0]

def test_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and process fact-checking claims.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON file.")
    parser.add_argument("--model", type=str, required=True, help="Path to input JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to input JSON file.")
    
    args = parser.parse_args()

    input_path = args.input_path
    model = args.model
    output_path = args.output_path
    content = test_data(input_path)

    keys = list(content.keys())
    values = list(content.values())
    save_result = []

    for key, value in zip(keys, values):
        try:
            if value == []:
                print(f"Skipping claim (no data): {key}")
                continue
            print(f"Processing claim: {key}")
            print("-----------------------------------------------------------------")
            
            # Analyze evidence
            try:
                documents, analyzer_time = analyze_fact_check([key, value], model)
            except Exception as e:
                print(f"Error analyzing fact check for claim {key}: {e}")
                continue
            
            fact_check_result, parsing_result, final_time = run_with_timeout(fact_check,args=(key, documents, model, analyzer_time))
            retry_count = 0

            # Retry mechanism
            while fact_check_result == None:
                retry_count += 1
                if retry_count > 20:
                    print("Failed to get a valid result after 20 attempts.")
                    fact_check_result = None
                    final_time = None
                    break

                fact_check_result, parsing_result, final_time = run_with_timeout(fact_check,args=(key, documents, model, analyzer_time))

            if fact_check_result == None:
                print(f"Skipping claim: {key}")
                continue

            tmp = {
                "claim": key,
                "label": parsing_result['label'],
                "reasoning": parsing_result['reasoning'],
                "date" : parsing_result['date'],
                "country" : parsing_result['country'],
                "urls" : parsing_result['url'],
                "time_taken": final_time
            }

            print("Result:")
            print(json.dumps(parsing_result, indent=4, ensure_ascii=False))
            print("-----------------------------------------------------------------")
            save_result.append(tmp)
        except Exception as e:
            print(f"Error processing claim {key}: {e}")
            import traceback
            traceback.print_exc()
            continue
    # Save results
    try:
        with open(output_path, 'w') as f:
            json.dump(save_result, f, indent=4, ensure_ascii=False)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")