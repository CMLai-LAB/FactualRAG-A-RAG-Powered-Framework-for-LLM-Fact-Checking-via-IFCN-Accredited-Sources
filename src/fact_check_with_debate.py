import re
import json
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.schema import Document
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal, Optional
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
        # Initialize the LLM with low randomness to improve output consistency and limit token generation.
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

class FinalAnswerOutput(BaseModel):
    label: Literal["Supported", "Refuted", "Not Enough Information"] = Field(
        description="The fact-check verdict. Must be one of: Supported, Refuted, or Not Enough Information"
    )
    reasoning: str = Field(description="Complete reasoning process explaining the verdict")
    date: str = Field(description="Date in YYYY-MM-DD format")
    language: Optional[str] = Field(default="", description="Language of the claim and context")
    country: str = Field(description="ISO 3166-1 alpha-2 country code")
    urls: list[str] = Field(description="List of source URLs used for evaluation")

class DebateOutput(BaseModel):
    agent_id: str = Field(description="ID of the agent (e.g., Agent1, Agent2, Agent3)")
    final_answer: FinalAnswerOutput = Field(
        description="The agent's final answer after debate with structured fields"
    )

def fact_check(query, documents, model, analyzer_time=None):
    if not documents:
        return

    # Initialize LLM with low randomness to improve output consistency and limit token generation
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
        "claim": "The textual claim subject to verification",
        "label": "Supported" or "Refuted" or "Not Enough Information",
        "language": "language code",
        "date": "YYYY-MM-DD",
        "country": "ISO 3166-1 alpha-2 country code",
        "url": ["URL1", "URL2", ...],
        "reasoning": "your complete reasoning here"
    }}

    Remember, output only the LEGAL JSON string mentioned above.
    """
    prompt = PromptTemplate.from_template(template)
    formatted_prompt = prompt.format(
        context=documents, 
        claim=query
    )

    time_start = timeit.default_timer()
    result = llm.invoke(formatted_prompt)
    time_end = timeit.default_timer()
    del llm
    gc.collect()
    try:
        # Remove any schema definitions from the result
        if '"$defs"' in result or '"properties"' in result:
            lines = result.split('\n')
            result = '\n'.join([line for line in lines if '"$defs"' not in line and '"properties"' not in line and '"required"' not in line])
        
        start_index = result.index('{')
        end_index = result.rindex('}')
        result_json = result[start_index:end_index + 1]

        # Try to parse JSON directly
        parsed_json = json.loads(result_json)

        # Check if it's a schema definition instead of actual data
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

    if thread.is_alive():  # If it exceeds the time limit, return None
        print("Function execution timed out.")
        return None
    if exception[0]:  # If an exception occurred
        print(f"Error during execution: {exception[0]}")
        return None
    return result[0]

def test_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def process_claim(claim, documents, model, analyzer_time):
    fact_check_result, parsing_result, final_time = run_with_timeout(
        fact_check, args=(claim, documents, model, analyzer_time)
    )
    retry_count = 0

    # Retry mechanism: up to 20 attempts if no valid result is obtained
    while fact_check_result is None:
        retry_count += 1
        if retry_count > 20:
            print("Failed to get a valid result after 20 attempts.")
            return None, 0
        fact_check_result, parsing_result, final_time = run_with_timeout(
            fact_check, args=(claim, documents, model, analyzer_time)
        )

    if fact_check_result is None:
        print(f"Skipping claim: {claim}")
        return None, 0

    # Combine results
    result_dict = {
        "claim": claim,
        "label": parsing_result['label'],
        "reasoning": parsing_result['reasoning'],
        "date": parsing_result['date'],
        "country": parsing_result['country'],
        "urls": parsing_result['url'],
    }
    return result_dict, final_time

# -------------------------------
# Implementation of the debate mechanism (including runtime tracking).
# -------------------------------

def agent_debate(agent_id, query, original_answer, other_answers, model, max_retries=3):
    parser = PydanticOutputParser(pydantic_object=DebateOutput)
    
    prompt = f"""
        You are a professional fact-checker participating in a debate task.
        Below is the claim you need to analyze and debate: {query}
        Below is your initial answer:
        {json.dumps(original_answer, ensure_ascii=False, indent=4)}
        Below are the answers provided by the other two agents:
        {json.dumps(other_answers, ensure_ascii=False, indent=4)}
        Please analyze and compare the strengths and weaknesses of each 
        answer, and decide whether you need to modify your own answer.
        If you find that parts of the other answers are more reasonable, 
        you may adjust your answer accordingly; otherwise, you may keep 
        your original answer unchanged.

        You must respond with a JSON object matching this structure:
        {{
            "agent_id": "{agent_id}",
            "final_answer": {{
                "claim": "The textual claim subject to verification",
                "label": "Supported" or "Refuted" or "Not Enough Information",
                "reasoning": "your reasoning here",
                "date": "YYYY-MM-DD",
                "language": "language code (e.g., en, zh)",
                "country": "ISO 3166-1 alpha-2 country code",
                "urls": ["URL1", "URL2", ...]
            }}
        }}

        Respond with only the above JSON format without adding any extra
        content.
    """

    # Retry loop
    for attempt in range(max_retries):
        try:
            start_time = timeit.default_timer()
            # Initialize the LLM with low randomness to improve output consistency and limit token generation
            llm = OllamaLLM(model=model, temperature=0.1, num_predict=4096)
            result = llm.invoke(prompt)
            del llm
            gc.collect()
            end_time = timeit.default_timer()
            debate_time = end_time - start_time
            
            # Attempt to parse the results
            try:
                # Remove any schema definitions from the result
                if '"$defs"' in result or '"properties"' in result:
                    lines = result.split('\n')
                    result = '\n'.join([
                        line for line in lines
                        if '"$defs"' not in line
                        and '"properties"' not in line
                        and '"required"' not in line
                    ])
                
                start_index = result.index('{')
                end_index = result.rindex('}')
                result_json = result[start_index:end_index + 1]

                # Attempt to parse JSON directly
                parsed_json = json.loads(result_json)

                # Check whether the output is a schema definition instead of actual data
                if "$defs" in parsed_json or ("properties" in parsed_json and "agent_id" not in parsed_json):
                    raise ValueError(f"{agent_id} returned a schema instead of data")

                # Automatically supplement missing language field
                if "final_answer" in parsed_json and "language" not in parsed_json["final_answer"]:
                    parsed_json["final_answer"]["language"] = ""
                    result_json = json.dumps(parsed_json)
                    print(f"  ⚠️  {agent_id} is missing the 'language' field; an empty string has been added automatically")
                
                # Use the parser for validation
                parsed_result = parser.parse(result_json)
                final_data = parsed_result.model_dump()
                final_data["debate_time"] = debate_time
                
                if attempt > 0:
                    print(f"  ✅ {agent_id} succeeded on attempt {attempt + 1}")
                
                return final_data
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  ⚠️  {agent_id} parsing failed on attempt {attempt + 1}: {e}")
                    print(f"  🔄 Retrying... ({max_retries - attempt - 1} attempts remaining)")
                    continue
                else:
                    print(f"  ❌ {agent_id} failed after all {max_retries} attempts")
                    print(f"     Last error: {e}")
                    return None
                    
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  ⚠️  {agent_id} execution failed on attempt {attempt + 1}: {e}")
                print(f"  🔄 Retrying... ({max_retries - attempt - 1} attempts remaining)")
                continue
            else:
                print(f"  ❌ {agent_id} failed after all {max_retries} attempts: {e}")
                return None
    
    return None


def debate_among_agents(query, results, model, num_rounds=5):
    current_results = results
    all_debate_history = []
    
    for round_num in range(1, num_rounds + 1):
        print(f"--- Debate Round {round_num} ---")
        debate_results = []
        
        for i, res in enumerate(current_results):
            try:
                other_answers = [current_results[j] for j in range(len(current_results)) if j != i]
                agent_id = f"Agent{i+1}"
                final = agent_debate(agent_id, query, res, other_answers, model)
                if final:
                    debate_results.append(final)
            except Exception as e:
                print(f"Error in debate round {round_num} for Agent{i+1}: {e}")
        
        all_debate_history.append(debate_results)
        
        # prepare for next round input (use this round's output)
        try:
            current_results = [d["final_answer"] for d in debate_results if d and "final_answer" in d]
        except Exception as e:
            print(f"Error extracting final answers from debate results: {e}")
            current_results = []
        
        if len(current_results) < 3:
            print(f"Warning: Only {len(current_results)} agents completed round {round_num}")
            break

    return debate_results  # return debate_results

def majority_vote(debate_results):
    votes = {}
    for debate in debate_results:
        if debate and "final_answer" in debate and "label" in debate["final_answer"]:
            label = debate["final_answer"]["label"]
            votes[label] = votes.get(label, 0) + 1

    if not votes:
        return None

    majority_label = max(votes, key=votes.get)

    # Select the agent that first proposed the answer as the final decision.
    for debate in debate_results:
        if debate and debate["final_answer"]["label"] == majority_label:
            return debate["final_answer"]
    return None

# -------------------------------
# Main Program
# -------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and process fact-checking claims.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON file.")
    parser.add_argument("--model", type=str, required=True, help="Model name to use.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSON file.")
    
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
            
            # Obtain initial answers from three agents
            result1, result1_time = process_claim(key, documents, model, analyzer_time)
            result2, result2_time = process_claim(key, documents, model, analyzer_time)
            result3, result3_time = process_claim(key, documents, model, analyzer_time)

            # Check whether at least two valid results are available
            # (lowering the requirement to improve success rate)
            valid_results = [r for r in [result1, result2, result3] if r is not None]
            if len(valid_results) < 2:
                print(f"Skipping claim due to insufficient valid results (only {len(valid_results)}/3): {key}")
                continue

            initial_results = valid_results

            # Conduct agent debate
            try:
                debate_start_time = timeit.default_timer()
                debate_results = debate_among_agents(key, initial_results, model)
                debate_end_time = timeit.default_timer()
            except Exception as e:
                print(f"Error during debate for claim {key}: {e}")
                continue

            # Determine the final answer using majority vote
            try:
                final_answer = majority_vote(debate_results)
            except Exception as e:
                print(f"Error in majority vote for claim {key}: {e}")
                final_answer = None
            
            total_time = (
                result1_time
                + result2_time
                + result3_time
                + (debate_end_time - debate_start_time)
            )

            if final_answer:
                print("Final Answer (Majority Vote):")
                print(json.dumps(final_answer, indent=4, ensure_ascii=False))
                print(f"Time taken: {total_time}")
                tmp = {
                    "claim": key,
                    "final_answer": final_answer,
                    "time_taken": total_time
                }
                save_result.append(tmp)
            else:
                print("The debate did not yield a majority consensus.")
            print("-----------------------------------------------------------------")
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
