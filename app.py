from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from io import BytesIO
from dotenv import load_dotenv
import os
import openai
import pandas as pd
from typing import List
from pydantic import BaseModel
import json

app = FastAPI()

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class MedicalNoteExtraction(BaseModel):
    birads: str
    findings: str
    recommendations: str

@app.post("/process-medical-csv/")
async def process_medical_csv(
    file: UploadFile = File(...), 
    search_terms: str = Form(...)
):
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await file.read()
    file_binary = BytesIO(content)
    df = pd.read_csv(file_binary, sep=';', encoding='utf-8')

    structured_data = []

    for index, row in df.iterrows():
        notes = row['ESTUDIO']  
        
        response = openai.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert at extracting structured data from medical notes."},
                {"role": "user", "content": f"Extract the following information from the notes: {notes}. Look for terms like {search_terms}."}
            ],
            response_format=MedicalNoteExtraction,
        )

        extracted_info = response.choices[0].message.parsed
        

    return {"structured_data": extracted_info}
