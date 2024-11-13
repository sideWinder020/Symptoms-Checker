from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import openai
from cachetools import TTLCache
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import logging
import asyncio
import json
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    user_id: str


with open('config_key.json', 'r') as file:
    json_data = json.load(file)


# load_dotenv()
# openai.api_key = os.getenv("OPENAPIKEY")

openai.api_key = json_data["OPENAPIKEY"]

app = FastAPI()
# MAX_QUESTIONS = int(os.getenv("FREEPLAN", 5))
MAX_QUESTIONS = int(json_data["MAX_QUESTIONS_ALLOWED"])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

conversations_cache = TTLCache(maxsize=1000, ttl=600)

model_name = "dmis-lab/biobert-base-cased-v1.1"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    nlp_pipeline = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    logger.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logger.error("Error loading model or tokenizer: %s", str(e))
    nlp_pipeline = None 

SYSTEM_PROMPT = (
    "You are a concise medical assistant AI. The user will provide a symptom, "
    "and you should ask one short follow-up question at a time based on the symptom. "
    "If the user asks a question that is unrelated to their symptom or medical inquiry, "
    "politely redirect them back to discussing the symptom by reminding them to stay focused "
    "on information that can help assess their health concerns. "
    "Avoid any extra statements or empathy phrases, and provide only the specific question needed to gather information. "
    "Keep each response as brief and relevant as possible."
)

FINAL_DIAGNOSIS_PROMPT = (
    "Based on the previous conversation, please provide a specific and detailed diagnosis or recommendation. "
    "Avoid broad or general diagnoses, and instead include possible subtypes, severity levels, and related symptoms if relevant. "
    "If it helps clarify, specify exact conditions, related factors, or stages of the condition. "
    "Do not ask any further questions. Provide only a concise, targeted diagnosis or summary based on the information gathered."
)


def get_user_conversation(user_id: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Retrieve or initialize a user conversation from the cache.
    """
    if user_id in conversations_cache and conversations_cache[user_id]["final_diagnosis_given"]:
        del conversations_cache[user_id]

    if user_id not in conversations_cache:
        conversations_cache[user_id] = {
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
            "question_count": 0,
            "final_diagnosis_given": False
        }
    return conversations_cache[user_id]

async def generate_medical_insights(symptom: str) -> str:
    """
    Generate medically relevant follow-up questions based on the provided symptom using BioBERT.
    """
    if not nlp_pipeline:
        return "Medical insights unavailable due to model loading error."
    
    try:
        prompt = f"Patients with {symptom} may experience [MASK]."
        predictions = nlp_pipeline(prompt)
        conditions = [pred["token_str"] for pred in predictions[:3]]
        
        follow_up_questions = [
            "Is the symptom constant or does it come and go?",
            "Do any other symptoms accompany it?",
            "Does anything worsen or alleviate the symptom?"
        ]
        insight_text = (
            f"Related conditions: {', '.join(conditions)}. "
            f"Suggested questions: {', '.join(follow_up_questions)}"
        )
        return insight_text
    except Exception as e:
        logger.error("Error generating medical insights: %s", str(e))
        return "No specific conditions found."

@app.get("/", response_class=HTMLResponse)
async def get_chat_page():
    """
    Serve the chat homepage.
    """
    try:
        return FileResponse("static/symptom_chat.html")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Homepage not found.")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_message = request.message.strip()
    user_id = request.user_id.strip()

    if not user_message or not user_id:
        raise HTTPException(status_code=400, detail="Message and user ID are required.")

    conversation_data = get_user_conversation(user_id)
    conversation = conversation_data["messages"]
    question_count = conversation_data["question_count"]

    if conversation_data["final_diagnosis_given"]:
        return JSONResponse({
            "reply": "The diagnosis has been provided",
            "session_ended": True
        })

    # Generate medical insights asynchronously
    medical_insights = await generate_medical_insights(user_message)
    conversation.append({"role": "assistant", "content": medical_insights})
    logger.info(question_count)
    if question_count >= MAX_QUESTIONS:
        conversation.append({"role": "system", "content": FINAL_DIAGNOSIS_PROMPT})

        try:
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-4",
                messages=conversation,
                temperature=0.3,
                max_tokens=200
            )
            final_diagnosis = response['choices'][0]['message']['content']
            conversation_data["final_diagnosis_given"] = True

            return JSONResponse({
                "reply": final_diagnosis,
                "session_end_message": "The diagnosis has been provided",
                "session_ended": True
            })
        except openai.error.OpenAIError as e:
            logger.error("OpenAI API error: %s", str(e))
            raise HTTPException(status_code=500, detail="Error with OpenAI API.")

    conversation.append({"role": "user", "content": user_message})
    conversation_data["question_count"] += 1

    try:
        response = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model="gpt-4",
            messages=conversation,
            temperature=0.7,
            max_tokens=150
        )
        bot_reply = response['choices'][0]['message']['content']
        conversation.append({"role": "assistant", "content": bot_reply})
        
        return JSONResponse({"reply": bot_reply, "session_ended": False})
    except openai.error.OpenAIError as e:
        logger.error("OpenAI API error: %s", str(e))
        raise HTTPException(status_code=500, detail="Error with OpenAI API.")




# python3 -m  pipreqs.pipreqs .
#to save the packages used by the current python file
# pip freeze only saves the packages that are installed with pip install in your environment(BAD)