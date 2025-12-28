import requests
import json
import os
import re
import gc
from fake_useragent import UserAgent

# Initialize UserAgent object
ua = UserAgent()
os.environ['USER_AGENT'] = ua.random
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
import requests
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
import timeit
from fp.fp import FreeProxy
from datetime import datetime
import urllib3
import signal
from time import sleep
import threading

# Disable InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")




def is_vector_db_exist(persist_directory, embeddings):
    try:
        if os.path.exists(persist_directory):
            vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            return vectordb
        else:
            return None
    except Exception as e:
        print(f"Error loading database at {persist_directory}: {e}")
        return None

def create_vector_db(persist_directory, embeddings):
    try:
        vectordb = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
        return vectordb
    except Exception as e:
        print(f"Error creating database at {persist_directory}: {e}")
        return None
    
def store_to_vectordb(query,result, questionDB, answerDB, extracted_data=None):
    if not result:
        return
    try:
        # Store answer in answerDB with token as metadata
        answer_doc = Document(page_content=result,metadata={
            "Language": str(extracted_data.get("Language")),
            "Date": str(extracted_data.get("Date")),
            "Country": str(extracted_data.get("Country")),
            "URL": str(extracted_data.get("URL"))
        })
        token = answerDB.add_documents([answer_doc])[0]
        # Store question in questionDB with token as metadata
        question_doc = Document(page_content=query, metadata={
            "id": token,
            "Language": str(extracted_data.get("Language")),
            "Date": str(extracted_data.get("Date")),
            "Country": str(extracted_data.get("Country")),
            "URL": str(extracted_data.get("URL"))
        })
        questionDB.add_documents([question_doc])
        
    except Exception as e:
        print(f"Error storing result to database: {e}")


def query_vectordb(question, questionDB, answerDB):
    try:
        # Get documents from the database
        db_documents = questionDB.get()['documents']
        db_size = len(db_documents)
    except Exception as e:
        print(f"Error retrieving vector database documents: {str(e)}")
        return "Error accessing the vector database."
    
    if db_size == 0:
        return "The vector database is empty."
    
    try:
        # Perform similarity search to find the closest k documents
        retriever = questionDB.similarity_search_with_score(
            query=question,
            k=min(db_size, 4),
        )
    except Exception as e:
        print(f"Error executing similarity search: {str(e)}")
        return
    
    if retriever[0][1] < 0.5:
        top_document_metadata = retriever[0][0].metadata['id']
    else:
        return "No relevant documents found."
    try:
        # Find the corresponding answer using metadata
        answer = answerDB.get(ids=[top_document_metadata])['documents'][0]
        return answer
    except Exception as e:
        print(f"Error retrieving answer from database: {str(e)}")
        return


def get_fact_check_url(query, api_key, search_engine_id, num_results=5):
    url = (
        f'https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={search_engine_id}&num={num_results}'
    )
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Ensure request is successful
        results = response.json()

        if 'items' in results:
            fact_check_urls = [item['link'] for item in results['items']]
            return fact_check_urls
        else:
            print("No items found in the search results.")
            return []
    except requests.RequestException as e:
        print(f"Error during API request: {e}")
        return []
    

def run_with_timeout(func, args=(), kwargs={}, timeout=10):
    result = [None]  # Store the function result
    exception = [None]  # Capture any exception

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

def web_loader(url, verify_ssl=False):
    try:
        proxy = FreeProxy(rand=True, timeout=3).get()  # Get a random proxy
        loader = WebBaseLoader(
            url,
            proxies={"http": proxy, "https": proxy},
            verify_ssl=verify_ssl,
        )
        docs = loader.load()  # Load the content
        return docs
    except Exception as e:
        print(f"Error loading content from {url}: {e}")
        return None

def get_fact_check_content(urls, max_retries=3):
    if not urls:
        print("No URLs provided.")
        return []

    print('-' * 50)
    print('Searching relevant info...')
    fact_check_content = []

    for i, url in enumerate(urls, start=1):
        for retries in range(max_retries):
            docs = run_with_timeout(web_loader, args=(url,))
            if docs:  # Successfully fetched content
                print(f"[{i}/{len(urls)}] Successfully fetched content from {url}")
                fact_check_content.append([docs, url])
                break
            else:  # Retry failed
                print(f"[{i}/{len(urls)}] Attempt {retries + 1} failed. Retrying...")
        else:  # Exceeded max retries
            print(f"[{i}/{len(urls)}] Failed to load content from {url} after {max_retries} attempts.")

    print('Searching completed.')
    print('-' * 50)
    return fact_check_content
    
def analyze_fact_check(fact_check_content, model):
    if fact_check_content is None:
        return "There are no fact check content."

    info = []
    time_start = timeit.default_timer()

    for i in fact_check_content:
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
        formatted_prompt = prompt.format(context=i[0])

        # Execute LLM
        result = llm.invoke(formatted_prompt)

        # Combine result with URL
        info.append(f"{result}, url: {i[1]}")

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
    
patterns = {
    "Claim Status": [r"Claim Status:\s*([^\n]+)", r"\*\*Claim Status\*\*:\s*([^\n]+)"],
    "Language": [r"Language:\s*([^\(\n]+)", r"\*\*Language\*\*:\s*([^\(\n]+)"],
    "Date": [r"Date:\s*([^\(\n]+)", r"\*\*Date\*\*:\s*([^\(\n]+)"],
    "Country": [r"Country:\s*([A-Z]+)", r"\*\*Country\*\*:\s*([A-Z]+)"],
    "URL": [r"URL:\s*(https?://[^\s]+)", r"https?://[^\s]+"]
}

def extract_data(response, patterns):
    result = {}

    # Check if response is a string
    if not isinstance(response, str):
        print(f"Skipping non-string response: {response} (type: {type(response)})")
        return None
    
    for key, regex_list in patterns.items():
        if key == "URL":
            matches = []
            for regex in regex_list:
                matches.extend(re.findall(regex, response))
            # Use set to remove duplicates, then convert back to list
            result[key] = list(set(matches)) if matches else None
        else:
            for regex in regex_list:
                match = re.search(regex, response)
                if match:
                    result[key] = match.group(1).strip()
                    break
            if key not in result:
                result[key] = None
    return result

def main(query, model, search_api_key, search_engine_id):
    # urls = get_fact_check_url(query, search_api_key, search_engine_id)
    urls = query[1]
    if urls == []:
        return "No relevant fact-checking articles found."
    content = get_fact_check_content(urls)
    if content == None:
        print("Failed to get fact check content.")
        return None
    documents, analyzer_time = analyze_fact_check(content, model)
    fact_check_result, parsing_result, final_time = fact_check(query[0], documents, model, analyzer_time)
    retry_count = 0
    while fact_check_result == None:
        retry_count += 1
        if retry_count > 20:
            print("Failed to get a valid result after 20 attempts.")
            return None
        fact_check_result, parsing_result, final_time = fact_check(query[0], documents, model, analyzer_time)
    
    return fact_check_result, parsing_result, final_time

def load_data(path):
    """Load claims data from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fact-checking with vector database cache.")
    
    # Create mutually exclusive group for input methods
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_path", type=str, help="Path to input JSON file.")
    input_group.add_argument("--claim", type=str, help="Single claim to fact-check.")
    
    parser.add_argument("--urls", type=str, nargs='+', help="URLs for evidence (required with --claim).")
    parser.add_argument("--model", type=str, required=True, help="LLM model to use (e.g., llama3, gemma2).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSON file.")
    parser.add_argument("--api_key_file", type=str, default="APIKey.json", help="Path to API key configuration file.")
    
    args = parser.parse_args()

    # Validate that if --claim is used, --urls must also be provided
    if args.claim and not args.urls:
        parser.error("--urls is required when using --claim")

    model = args.model
    output_path = args.output_path
    api_key_file = args.api_key_file

    # Load API keys
    with open(api_key_file, 'r') as f:
        config = json.load(f)

    search_engine_id = config.get('search_engine_id')
    search_api_key = config.get('search_api_key')

    # Prepare query data based on input method
    today = datetime.today().strftime('%Y-%m-%d')
    query = []
    
    if args.input_path:
        # Load from JSON file
        print(f"Loading claims from file: {args.input_path}")
        urls_data = load_data(args.input_path)
        for i in urls_data:
            if i['urls'] != []:
                query.append([i['claim'], i['urls']])
    else:
        # Single claim from command line
        print(f"Processing single claim from command line")
        query.append([args.claim, args.urls])

    save_result = []

    # Initialize vector databases
    questionDB = is_vector_db_exist(f"{today}-question-db", embeddings)
    answerDB = is_vector_db_exist(f"{today}-answer-db", embeddings)

    if not questionDB:
        questionDB = create_vector_db(f"{today}-question-db", embeddings)
    if not answerDB:
        answerDB = create_vector_db(f"{today}-answer-db", embeddings)

    # Process each claim
    for i in query:
        print(f"Processing claim: {i[0]}")
        query_result = query_vectordb(i[0], questionDB, answerDB)
        if query_result != "No relevant documents found." and query_result != "The vector database is empty.":
            print("Result found in database:")
            print(query_result)
        else:
            print("No relevant documents found in the database. Searching online...")
            result = main(i, model, search_api_key, search_engine_id)
            if result == None:
                print("No relevant fact-checking articles found.")
                continue
            
            fact_check_result, parsing_result, final_time = result
            
            tmp = {
                "claim": i[0],
                "result": fact_check_result,
                "label": parsing_result.get("label"),
                "language": parsing_result.get("language"),
                "date": parsing_result.get("date"),
                "country": parsing_result.get("country"),
                "url": parsing_result.get("url"),
                "reasoning": parsing_result.get("reasoning"),
                "time_taken": final_time
            }
            print("Result:")
            print(json.dumps(parsing_result, indent=4, ensure_ascii=False))
            store_to_vectordb(i[0], fact_check_result, questionDB, answerDB, extracted_data=parsing_result)
            save_result.append(tmp)

    # Save results
    with open(output_path, 'w') as f:
        json.dump(save_result, f, indent=4, ensure_ascii=False)
    
    print(f"\nResults saved to {output_path}")
