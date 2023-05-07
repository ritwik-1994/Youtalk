from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import requests
from tree_index_builder import build_index
from langchain.vectorstores import FAISS
import os
import shutil
import signal
import time
import re
import socket
from openai.error import AuthenticationError, RateLimitError
from decimal import Decimal
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.output_parsers import RegexParser
from langchain.prompts import PromptTemplate
import os
from youtube_transcript_api import YouTubeTranscriptApi


app = Flask(__name__)
CORS(app, origins="*")

@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

@cross_origin(origins="*")

@app.route('/transcribe_video', methods=['POST'])
def transcribe_video():
    req_data = request.get_json()
    print("Incoming request data:", req_data)
    video_url = req_data['url']

    global doc_index
    if 'url' not in req_data:
        return jsonify({'error': 'URL is required'}), 400
    
    # Extract the video ID from the video URL
    video_id = video_url.split("watch?v=")[-1]

    # Fetch the transcript from YouTube
    transcripts = fetch_youtube_transcript(video_id)

    try:
        doc_index = process_transcripts(transcripts)
    except Exception as e:
        print(f"An error occurred while processing the transcripts:", e)
        return jsonify({'error': 'An error occurred'}), 500
    
    print(transcripts, '\n\n\n\n\\n\n\n\n\n\\n')
    print(doc_index)

    return jsonify({
        'summary': 'Index created successfully',
    })
''' Commenting out the steps where video_process function is used
    try:
        transcripts, doc_index = process_video(video_url)
    except Exception as e:
        print(f"An error occurred while transcribing and summarizing the video:", e)
        return jsonify({'error': 'An error occurred'}), 500

    return jsonify({
        'summary': 'Index created successfully',
    })
'''

app.config['OPENAI_KEY'] = None

@app.route('/save_openai_key', methods=['POST'])
def save_openai_key():
    req_data = request.get_json()
    openai_key = req_data.get('openai_key')
    if not openai_key:
        return jsonify({'error': 'OpenAI API key is required'}), 400

    app.config['OPENAI_KEY'] = openai_key
    return jsonify({'message': 'OpenAI API key saved'}), 200

@app.route('/get_openai_key', methods=['GET'])
def get_openai_key():
    openai_key = None
    if 'openai_key' in request.args:
        openai_key = request.args.get('openai_key')
    return jsonify({'openai_key': openai_key})

def remove_pycache():
    pycache_path = os.path.join(os.path.dirname(__file__), '__pycache__')
    if os.path.exists(pycache_path):
        shutil.rmtree(pycache_path)


@app.route('/process_query', methods=['POST'])
def process_query():
    req_data = request.get_json()
    query = req_data['query']
    query = query + ', Can you please explain in detail.'

    if 'query' not in req_data:
        return jsonify({'answer': 'Query is required'}), 400

    if doc_index is None:
        return jsonify({'answer': 'Index not available'}), 500

    openai_key = app.config['OPENAI_KEY']

    if not openai_key:
        return jsonify({'answer': 'OpenAI API key is required'})
    '''output_parser = RegexParser(
    regex=r"'score': .*?,'input_documents': .*?, 'question': .*?, 'source': \d, 'output_text': ' (.*?)'",
    output_keys=["input_documents", "question","source","output_text", "score"],
)
    prompt_template = """Use the query provided in the question key to generate a detailed and specific response from the input provided in the doc_index. The output should be in the following format: Timestamp: [Timestamp] Question: [question here], Query: [query], Source: [source], score: [score]. Begin! Context: {context} / Question: {question}"""
    PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
    output_parser=output_parser,)'''

    chain = load_qa_with_sources_chain(OpenAI(temperature=0.7, openai_api_key=openai_key), metadata_keys=['source'], chain_type="map_rerank")
    try:
        answer = chain({"input_documents": doc_index, "question": query}, return_only_outputs=False)
    except ValueError as e:
        print(f"Error occurred while processing the query: {e}")
        best_answer = str({e})
        best_answer = "The context in the video regarding the question is vague. Predicting output: " + best_answer.split("Could not parse output:")[-1]
            #best_answer = best_answer.split("Error occurred while processing the query: Could not parse output:")[-1]
            #best_answer = "Vague Question: Approximating Answer" + best_answer

        return jsonify({'answer': best_answer, 'start_time': '00:00', 'end_time': '01:00'})

    except AuthenticationError as e:
        best_answer = "OpenAI Key is incorrect. Please enter the correct OpenAI key\n"
        return jsonify({'answer': best_answer, 'start_time': '00:00', 'end_time': '01:00'})
            
    except e:
        best_answer = "Either the video hasn't been loaded or your OpenAI key is throwing Rate Limit errors. Load the video again or while for a while will retry again!\n"
        return jsonify({'answer': best_answer, 'start_time': '00:00', 'end_time': '01:00'})

    source_to_find = answer['source']

    for document in answer['input_documents']:
        if document.metadata['source'] == source_to_find:
            timestamp = document.metadata['timestamp']
            break

    timestamp = timestamp.split('-')
    start_time = int(Decimal(timestamp[0]))
    end_time = int(Decimal(timestamp[1]))
    print(answer)
    response = {
        'answer': answer['output_text'],
        'start_time': start_time,
        'end_time': end_time
    }

    return jsonify(response)

def fetch_youtube_transcript(video_id: str) -> dict:

    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages = [
    'en',       # English,
    'zh-Hans',  # Chinese (Simplified)
    'es',       # Spanish
    'hi',       # Hindi
    'ar',       # Arabic
    'bn',       # Bangla
    'pt',       # Portuguese
    'ru',       # Russian
    'ja',       # Japanese
    'jv',       # Javanese
    'pa',       # Punjabi
    'de',       # German
    'jw',       # Javanese
    'ko',       # Korean
    'fr',       # French
    'te',       # Telugu
    'mr',       # Marathi
    'tr',       # Turkish
    'ta',       # Tamil
    'vi',       # Vietnamese
    'ur',       # Urdu
    'id',       # Indonesian
    'gu',       # Gujarati
    'pl',       # Polish
    'uk',       # Ukrainian
    'fa',       # Persian
    'ml',       # Malayalam
    'kn',       # Kannada
    'or',       # Odia
    'my',       # Burmese
    'th',       # Thai
    'ro',       # Romanian
    'nl',       # Dutch
    'hu',       # Hungarian
    'el',       # Greek
    'sv',       # Swedish
    'uz',       # Uzbek
    'az',       # Azerbaijani
    'kk',       # Kazakh
    'sr',       # Serbian
    'sq',       # Albanian
    'hy',       # Armenian
    'iu',       # Inuktitut
    'he',       # Hebrew
    'uz',       # Uzbek
    'lt',       # Lithuanian
    'et',       # Estonian
    'hr',       # Croatian
    'lv',       # Latvian
    'sl',       # Slovenian
    'bs',       # Bosnian
    'mk',       # Macedonian
    'sq',       # Albanian
    'mt',       # Maltese
    'ga',       # Irish
    'is',       # Icelandic
    'cy',       # Welsh
    'gd',       # Scottish Gaelic
    'fo',       # Faroese
    'kw',       # Cornish
    'gv',       # Manx
])
    return transcript

def process_transcripts(transcripts: list) -> FAISS:
    # Initialize variables
    formatted_transcripts = {}
    current_start = transcripts[0]['start']
    current_text = transcripts[0]['text']
    current_duration = transcripts[0]['duration']

    # Calculate the duration threshold
    last_element_duration = transcripts[-1]['duration']
    duration_threshold = max(180, round(last_element_duration / 20, -1))

    # Iterate through the transcripts
    for t in transcripts[1:]:
        # If the current duration is less than the duration threshold, append the text and update the duration
        if current_duration < duration_threshold:
            current_text += " " + t['text']
            current_duration += t['duration']
        else:
            # If the current duration is at least the duration threshold, add the entry to the formatted_transcripts dictionary
            formatted_transcripts[f"{current_start}-{current_start + current_duration}"] = current_text

            # Reset the current_start, current_text, and current_duration for the next entry
            current_start = t['start']
            current_text = t['text']
            current_duration = t['duration']

    # Add the last entry to the formatted_transcripts dictionary
    formatted_transcripts[f"{current_start}-{current_start + current_duration}"] = current_text

    # Build the index
    index = build_index(formatted_transcripts)

    return index
''' Commenting out the function which processes videos
def process_video(video_url: str) -> Tuple[Dict[float, Tuple[float, float, str]], FAISS]:
    video_file_path = download_video(video_url, "videos")
    audio_file_path = extract_audio(video_file_path, "audios")
    transcripts = transcribe_audio(audio_file_path)
    index = build_index(transcripts)

    return transcripts, index
'''
if __name__ == "__main__":
    remove_pycache()
    app.run(debug=True, port=65500)