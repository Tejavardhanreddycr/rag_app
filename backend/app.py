from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from flask_socketio import SocketIO, Namespace
from utils.tools import google_web_search
from utils.utils import format_docs, format_web_docs, get_vectorstore, rerank_docs, grading
from utils.history import collection, append_chat_entry, get_chat_history_by_date
from dotenv import load_dotenv
import os, logging
from langchain_openai import ChatOpenAI
import time
import json
from bson import json_util
import warnings
import pandas as pd
import logging
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import pytz
from langsmith import Client
from flask import Flask
import re


load_dotenv()
warnings.filterwarnings('ignore', message="Valid config keys have changed in V2:")

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# # Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "default"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"

LANGCHAIN_TRACING_V2 = "true"  
LANGCHAIN_API_KEY = "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"
LANGCHAIN_PROJECT = "default"
TARGET_TIMEZONE = pytz.timezone('Asia/Kolkata')

"""---------------------------------------------dashboard------------------------------------------"""
"""Fetching the 24 hrs data using the below format"""

# Configure the logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IST = pytz.timezone('Asia/Kolkata')
logger.info("Start time: %s, End time: %s", datetime.now(IST) - timedelta(days=1), datetime.now(IST))





# Initialize empty DataFrame for storing data
df2 = pd.DataFrame(columns=[ 'Question','Name', 'Time', 'Total Tokens', 'Total Cost($)','Total Cost(rs)'])

@app.route("/data")
def data():
    names = []
    run_types = []
    rows = []
    COHERE_TOKEN_PRICE_input = 0.000003
    COHERE_TOKEN_PRICE_output = 0.000015
    CHATOPENAI_TOKEN_PRICE_input = 0.000005
    CHATOPENAI_TOKEN_PRICE_output = 0.000015
    # Set time range for data retrieval (last 24 hours)
    end_time_ist = datetime.now(TARGET_TIMEZONE)
    start_time_ist = end_time_ist - timedelta(days=1)
    end_time_utc = end_time_ist.astimezone(pytz.UTC)
    start_time_utc = start_time_ist.astimezone(pytz.UTC)

    logger.info("Fetching Data from LangSmith")
    st_time = time.time()

    # Initialize defaultdict to hold data, with each question storing up to 7 recent runs
    question_aggregates = defaultdict(lambda: {
        "Question": None,
        "Name": "ChatOpenAI",
        "Time": None,
        "Total Tokens": 0,
        "Total Cost($)": 0.0,
        "Total Cost(rs)": 0.0,
        "Runs": []  # Stores up to 7 runs per question
    })
    client = Client(api_key=LANGCHAIN_API_KEY)

    try:
        # Fetch runs from LangSmith with the specified time range
        runs = client.list_runs(
            project_name=LANGCHAIN_PROJECT,
            start_time=start_time_utc,
            end_time=end_time_utc
        )

        for run in runs:
            # Extract input and identify the question using regex
            input_text = run.inputs['messages'][0][0]['kwargs']['content']
            question_match = re.search(r"Question: (.*?)(?:\n|$)", input_text)
            question = question_match.group(1).strip() if question_match else "Question not found"

            # Convert start time to IST
            if run.start_time:
                utc_time = run.start_time.replace(tzinfo=timezone.utc)
                time_ist = utc_time.astimezone(TARGET_TIMEZONE).strftime("%d, %b, %Y, %H:%M:%S")
            else:
                time_ist = None

            # Get the entry for the current question
            entry = question_aggregates[question]
            entry["Question"] = question
            entry["Time"] = entry["Time"] or time_ist

            # Calculate cost for the run
            run_cost = (
                (run.prompt_tokens * (COHERE_TOKEN_PRICE_input if run.name == 'ChatCohere' else CHATOPENAI_TOKEN_PRICE_input)) +
                (run.completion_tokens * (COHERE_TOKEN_PRICE_output if run.name == 'ChatCohere' else CHATOPENAI_TOKEN_PRICE_output))
            )

            # Append the run if fewer than 7 exist for this question
            if len(entry["Runs"]) < 7:
                entry["Runs"].append({
                    "Time": time_ist,
                    "Tokens": run.total_tokens,
                    "Cost($)": round(run_cost, 6)
                })
            else:
                # Skip additional runs after 7 for each question
                continue

            # Update aggregated tokens and costs for the latest 7 runs
            entry["Total Tokens"] = sum(r['Tokens'] for r in entry["Runs"])
            entry["Total Cost($)"] = round(sum(r['Cost($)'] for r in entry["Runs"]), 6)
            entry["Total Cost(rs)"] = round(entry["Total Cost($)"] * 84, 3)

            # Collect data for other visualizations
            names.append(run.name)
            run_types.append(run.run_type)

            run_data = {}
            if run.error is None or run.end_time is None:
                run_data['Status'] = '✅'
            else:
                run_data['Status'] = '❗'
            rows.append(run_data)
        # Convert aggregated data to a DataFrame and reset index, excluding 'Runs'
        output_data = pd.DataFrame.from_records([
            {k: v for k, v in question_data.items() if k != 'Runs'}
            for question_data in question_aggregates.values()
        ])
        # Convert aggregated data to a DataFrame and reset index
        # output_data = pd.DataFrame.from_records(list(question_aggregates.values()))

        # Convert 'Time' column to datetime and set it as index
        output_data['Time'] = pd.to_datetime(output_data['Time'], errors='coerce')
        output_data.dropna(subset=['Time'], inplace=True)
        output_data.set_index('Time', inplace=True)

        # Generate bar chart data: resample by hour and count the number of requests
        bar_chart_data = output_data.resample('h').size().reset_index(name='Number of Requests')
        bar_chart_data.rename(columns={"Time": "name", "Number of Requests": "value"}, inplace=True)

        # Generate pie chart data for Names and Run Types
        names_count = pd.Series(names).value_counts().to_dict()
        keys_to_keep = {'ChatOpenAI', 'ChatCohere'}
        pie_chart_data_names_count = [{"name": name, "value": count} for name, count in names_count.items() if name in keys_to_keep]

        run_types_count = pd.Series(run_types).value_counts().to_dict()
        pie_chart_data_run_types_count = [{"name": run_type, "value": count} for run_type, count in run_types_count.items()]

        df3 = pd.DataFrame(rows)
        status_counts = df3['Status'].value_counts().to_dict()
        pie_chart_data_status = [{"name": status, "value": count} for status, count in status_counts.items()]

        # Convert output_data DataFrame to a dictionary for table data
        table_data = output_data.reset_index().to_dict(orient='records')

        end_time = time.time()
        print(f"Response from '/data' API took {end_time - st_time} seconds")

        return jsonify({
            "pieChartDataStatus": pie_chart_data_status,
            "pieChartDataNamesCount": pie_chart_data_names_count,
            "pieChartDataRunTypesCount": pie_chart_data_run_types_count,
            "tableData": table_data,
        }), 200

    except Exception as e:
        logging.error(f"Error while updating DataFrame: {e}")
        abort(500, description="Internal Server Error")
 
# Initialize global DataFrame outside of the route function
df = pd.DataFrame(columns=['TotalTokens', 'TotalCost'])
@app.route("/analytics")
def analytics():
    global df
    st_time = time.time()
    total_cost = 0
    # Define token prices
    COHERE_TOKEN_PRICE_input = 0.000003
    COHERE_TOKEN_PRICE_output = 0.000015
    CHATOPENAI_TOKEN_PRICE_input = 0.000005  
    CHATOPENAI_TOKEN_PRICE_output = 0.000015
    RERANKING_COST = 0.002
    
    # Initialize the LangSmith client
    client = Client(api_key=LANGCHAIN_API_KEY)
    end_time_ist = datetime.now(IST)
    start_time_ist = end_time_ist - timedelta(days=2)

    # Convert time to UTC
    end_time_utc = end_time_ist.astimezone(pytz.UTC)
    start_time_utc = start_time_ist.astimezone(pytz.UTC)


    runs = client.list_runs(
        project_name=LANGCHAIN_PROJECT,
        start_time=start_time_utc,
        end_time = end_time_utc
    )
 
    # Convert a UTC timestamp to the target timezone
    def convert_to_timezone(utc_timestamp):
        if utc_timestamp:
            try:
                # Try parsing with microseconds
                utc_time = datetime.strptime(utc_timestamp, "%Y-%m-%dT%H:%M:%S.%f")
            except ValueError:
                # Fallback if microseconds are missing
                utc_time = datetime.strptime(utc_timestamp, "%Y-%m-%dT%H:%M:%S")
 
            # Localize to UTC and then convert to the target timezone
            utc_zone = pytz.timezone('UTC')
            localized_time = utc_zone.localize(utc_time).astimezone(TARGET_TIMEZONE)
            return localized_time.isoformat()
   
        return None
 
    try:
        print("Analyzing data")
        
        # Create a dictionary to group by timestamp
        grouped_runs = defaultdict(lambda: {
            'cohere_input_tokens': 0,
            'cohere_output_tokens': 0,
            'cohere_cost': 0,
            'chatopenai_input_tokens': 0,
            'chatopenai_output_tokens': 0,
            'chatopenai_cost': 0
        })
        for run in runs:
            timestamp = convert_to_timezone(run.start_time.isoformat()) if run.start_time else None
 
            if run.name == 'ChatCohere':
                input_tokens = run.prompt_tokens if run.prompt_tokens else 0
                output_tokens = run.completion_tokens if run.completion_tokens else 0
                token_cost = (input_tokens * COHERE_TOKEN_PRICE_input) + (output_tokens * COHERE_TOKEN_PRICE_output)
 
                grouped_runs[timestamp]['cohere_input_tokens'] += input_tokens
                grouped_runs[timestamp]['cohere_output_tokens'] += output_tokens
                grouped_runs[timestamp]['cohere_cost'] += token_cost
 
            elif run.name == 'ChatOpenAI':
                input_tokens = run.prompt_tokens if run.prompt_tokens else 0
                output_tokens = run.completion_tokens if run.completion_tokens else 0
                token_cost = (input_tokens * CHATOPENAI_TOKEN_PRICE_input) + (output_tokens * CHATOPENAI_TOKEN_PRICE_output)
 
                grouped_runs[timestamp]['chatopenai_input_tokens'] += input_tokens
                grouped_runs[timestamp]['chatopenai_output_tokens'] += output_tokens
                grouped_runs[timestamp]['chatopenai_cost'] += token_cost
 
        # Group runs into sets of 7 and analyze the data
        # batched_runs = []
        batch_size = 7
        grouped_data = list(grouped_runs.items())  # List of timestamps and corresponding data
        count = 0
        for i in range(0, len(grouped_data), batch_size):
            count += 1
            batch = grouped_data[i:i+batch_size]  # Get 7 grouped runs at a time
            total_input_tokens = 0
            total_output_tokens = 0
            TC = 0
 
            for _, data in batch:
                total_input_tokens += data['cohere_input_tokens'] + data['chatopenai_input_tokens']
                total_output_tokens += data['cohere_output_tokens'] + data['chatopenai_output_tokens']
                total_cost += data['cohere_cost'] + data['chatopenai_cost'] + RERANKING_COST
                TC += data['cohere_cost'] + data['chatopenai_cost'] + RERANKING_COST

 
            # Add the batched result to a new row in the DataFrame
            new_row = pd.DataFrame([{
                'TotalTokens': total_input_tokens,
                'TotalCost': TC
            }])
            
            # Concatenate the new row to the existing DataFrame
            df = pd.concat([df, new_row], ignore_index=True)
        
        print("**********",count)
        # Calculate analytics
        average_tokens_per_question = df['TotalTokens'].mean()
        average_cost_per_question = df['TotalCost'].mean()
        average_latency_per_question = 5.05
        print(f"Response time form '/analytics' {time.time() - st_time}")
        return jsonify({
            "averageTokensPerQuestion": average_tokens_per_question,
            "averageCostPerQuestion": round(average_cost_per_question,4),
            "averageLatencyPerQuestion": average_latency_per_question,
            "totalCost": round(total_cost,4),
            "totalQuestions":count
        })
    except Exception as e:
        logging.error(f"Error while calculating analytics: {e}")
        abort(500, description="Internal Server Error")
 

"""# ---------------------------------------------------Chat API---------------------------------------------------------"""
class QueryNamespace(Namespace):
    def on_connect(self):
        print("Client connected to /query namespace")

    def on_disconnect(self):
        print("Client disconnected from /query namespace")

socketio.on_namespace(QueryNamespace("/openai"))

def rag_qa_stream(prompt):
    global global_time_to_first_response, global_total_process_time 
    try:
        source_origin = ""
        full_response = ""
        first_chunk_logged = False
        sources = []  # Initialize sources
        
        process_start_time = time.time()
        logger.info(f"Received prompt: {prompt}")

        # Retrieve documents from vector store
        try:
            vectorstore = get_vectorstore()
            retrieval_start_time = time.time()
            docs = vectorstore.similarity_search_with_score(prompt, k=10)
            fetching_time = time.time() - retrieval_start_time
            logger.info(f"Data fetched in {fetching_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error during document retrieval: {str(e)}")
            yield b"Error retrieving documents", 500

        # Rerank documents
        try:
            rerank_start_time = time.time()
            retrieved_docs = [doc for doc, _ in docs]
            reranked_docs = rerank_docs(prompt, retrieved_docs)
            rerank_time = time.time() - rerank_start_time
            logger.info(f"Rerank done in {rerank_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error during document reranking: {str(e)}")
            yield b"Error reranking documents", 500

        # Grading process
        try:
            grading_start_time = time.time()
            approved_docs = grading(reranked_docs, prompt)
            # print(approved_docs)
            print(f"approved {len(approved_docs)} docs")
            grading_time = time.time() - grading_start_time
            logger.info(f"Grading done in {grading_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error during document grading: {str(e)}")
            yield b"Error grading documents", 500

        # Check if additional Google search is needed
        try:
            if len(approved_docs) < 3:
                if len(approved_docs) == 2 or len(approved_docs) == 1:
                    source_origin = "Web & amd.VectorDB"
                else:
                    source_origin = "web"
                search_start_time = time.time()
                links, content = google_web_search(prompt, 5 - len(approved_docs))
                logger.info(f"Google search done in {time.time() - search_start_time:.2f} seconds")
                formatted_docs, sources = format_docs(approved_docs)
                approved_docs = formatted_docs + format_web_docs(content)
                sources.extend(links)
                
                
            else:
                approved_docs, sources = format_docs(approved_docs)
                source_origin = "amd.VectorDB"
            
            # Remove duplicate links
            sources = list(set(sources))  # Ensure unique links
            print(source_origin)
        except Exception as e:
            logger.error(f"Error during Google search: {str(e)}")
            yield b"Error fetching additional context", 500
        
        sources = sorted(sources[:3])  # Keep only the top 3 sources
        print(sources)
        # Generate response using OpenAI
        relevant_docs = approved_docs
        # print(relevant_docs)
        try:
            prompt_template = f"""
            You are a helpful and informative AI assistant. Your task is to answer the user's question based on the provided context.

            Context: {relevant_docs}

            Question: {prompt}

            Instructions:

            1. Answer the question truthfully and comprehensively, relying solely on the given context.
            2. Do not fabricate information or answer from your own knowledge base and don't talk about what is in the context.
            3. Cite the source documents (as clickable) that support your answer by providing their URLs mentioned here and don't give any placeholders for urls, sources = {sources}.

            [Provide your answer here]

            **Sources:**

            [List the URLs as urls of the source documents in bullet format used to answer the question, one clickable URL per line](sources = {sources})

            Print the below Origin as it mentioned below
            **Origin:** {source_origin} 
            
            """


            llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model_name="gpt-4o",temperature=0.6, streaming=True)
            # llm = get_cohere_llm()
            # llm = get_openai_llm()
            print(llm.model_name)
            response_start_time = time.time()
            for chunk in llm.stream(prompt_template):
                chunk_text = chunk.content
                full_response += chunk_text

                if not first_chunk_logged:
                    first_chunk_time = time.time()
                    global_time_to_first_response = fetching_time + rerank_time + grading_time + (first_chunk_time - response_start_time)
                    logger.info(f"Time to first response: {global_time_to_first_response:.2f} seconds")
                    first_chunk_logged = True

                yield chunk_text.encode('utf-8')
                socketio.emit('response', {'text': chunk_text}, namespace="/openai")
            # print(sources)
         # Only append if response is non-empty and sources exist
            append_chat_entry(prompt, full_response)
        

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            yield b"Error generating response", 500

    except Exception as e:
        logger.error(f"Error in overall process: {str(e)}")



@app.route('/rag_qa_api_stream', methods=['POST'])
def rag_qa_api_stream():
    data = request.json
    print(data)
    prompt = data.get('text')
    print(prompt)
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    return rag_qa_stream(prompt)


"""---------------------------------------------------------History-----------------------------------------------"""
# Error handling for history routes
@app.route('/list_files', methods=['POST', 'GET'])
def list_files():
    try:
        # Fetch distinct dates sorted in descending order
        files = collection.distinct('date')
        files.sort(reverse=True)  # Sort the dates in descending order
        return jsonify(files)
    except Exception as e:
        logger.error(f"Error retrieving file list: {str(e)}")
        return jsonify({"error": "Failed to retrieve file list"}), 500


@app.route('/one_file', methods=['POST', 'GET'])
def one_file():
    try:
        data = request.get_json()
        file_date = data.get('file')
        if not file_date:
            return jsonify({"error": "File parameter is required"}), 400

        chat_history = get_chat_history_by_date(file_date)
        if chat_history:
            return json.loads(json_util.dumps(chat_history))
        else:
            return jsonify([])

    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        return jsonify({"error": "Failed to retrieve data"}), 500

"""----------------------------------Related questions-----------------------------------------------"""
# Error handling for related questions
@app.route('/related_questions', methods=['POST'])
def related_questions():
    data = request.get_json()
    prompt = data.get('prompt')
    answer = data.get('answer')

    if not prompt or not answer:
        return jsonify({'error': 'Both prompt and answer are required'}), 400

    try:
        related_questions = generate_related_questions(prompt, answer)
        return jsonify({'related_questions': related_questions})
    except Exception as e:
        logger.error(f"Error generating related questions: {str(e)}")
        return jsonify({"error": "Failed to generate related questions"}), 500

# Example of generating related questions
def generate_related_questions(prompt, answer):
    try:
        llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), temperature=0.7, model_name="gpt-4o")
        prompt_template = f"""
        Question: {prompt}
        Answer: {answer}

        generate four related AMD questions based on the **Question** and **Answer**, one per line. Do not number the questions. 
        """

        response = llm.invoke(prompt_template)
        related_questions = [line.strip() for line in response.content.splitlines() if line.strip()]
        logger.info(f"Generated related questions: {related_questions}")
        return related_questions[:4]
    except Exception as e:
        logger.error(f"Error generating related questions: {str(e)}")
        return []

"""---------------------------------------------baseroute------------------------------------------"""
@app.route('/home',methods = ['GET'])
def basic_route():
    return "backend started successfully"


# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8088, debug=True)
    
