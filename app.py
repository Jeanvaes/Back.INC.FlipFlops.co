from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from io import BytesIO
from dotenv import load_dotenv
import os
import openai
import pandas as pd
from typing import List
from pydantic import BaseModel
import oracledb

app = FastAPI()

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

os.environ["TNS_ADMIN"] = os.getenv("TNS_ADMIN")
connection = oracledb.connect(
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    dsn=os.getenv("DB_DSN"),
    config_dir=os.getenv("CONFIG_DIR"),
    wallet_location=os.getenv("WALLET_LOCATION"),
    wallet_password=os.getenv("WALLET_PASSWORD")
)

# Function to parse the extracted_info string
def parse_extracted_info(extracted_info):
    # Ensure extracted_info is a string
    info_str = str(extracted_info)
    parsed_info = {}
    for item in info_str.split(' '):
        if '=' in item:
            key, value = item.split('=')
            parsed_info[key.upper()] = value.strip("'")
    return parsed_info

class MedicalNoteExtraction(BaseModel):
    id_paciente: str
    prestacion: str
    nodulos: str  # 0(No), 1(Si)
    morfologia_nodulos: str  # 1(Ovalado), 2(Redondo), 3(Irregular)
    margenes_nodulos: str  # 1(Circunscritos), 2(Microlobulados), 3(Indistintos o mal definidos), 4(Obscurecidos), 5(Espiculados)
    densidad_nodulo: str  # 1(Densidad Grasa), 2(Baja Densidad (hipodenso)), 3(Igual Densidad (isodenso)), 4(Alta Densidad (hiperdenso))
    microcalcificaciones: str  # 0(No), 1(Si)
    calcificaciones_benignas: str  # 1(Cutaneas), 2(Vasculares), 3(Gruesas o Pop Corn), 4(Leño o Vara), 5(Redondas o puntiformes), 6(Anulares), 7(Distroficas), 8(Leche de Calcio), 9(Suturas)
    calcificaciones_sospechosas: str  # 1(Gruesas heterogeneas), 2(Amorfas), 3(Finas pleomorficas), 4(Lineas finas o lineales ramificadas)
    distribucion_calcificaciones: str  # 1(Difusas), 2(Regionales), 3(Agrupadas (cumulo)), 4(Segmentaria), 5(Lineal)
    presencia_asimetrias: str  # 0(No), 1(Si)
    tipo_asimetria: str  # 1(Asimetria), 2(Asimetria global), 3(Asimetria focal), 4(Asimetria focal evolutiva)
    hallazgos_asociados: str  # 1(Retracción de la piel), 2(Retracción del pezón), 3(Engrosamiento de la piel), 4(Engrosamiento trabecular), 5(Adenopatias axilares)
    lateralidad_hallazgo: str  # 1(DERECHO), 2(IZQUIERDO), 3(BILATERAL)
    birads: str  # 0, 1, 2, 3, 4A, 4B, 4C, 5, 6
    edad: str

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
    print("DF CARGADO")
    content= '''You are an expert at extracting structured data from medical notes. Follow the same parameters: -Nodulos: 0(No) 1(Si)
    -Morfologia de los nodulos:1(Ovalado) 2(Redondo) 3(Irregular) -Margenes de los nodulos: 1(Circunscritos), 2(Microlobulados), 3(Indistintos o mal definidos), 4(Obscurecidos), 5(Espiculados)
    -Densidad del nodulo: 1(Densidad Grasa), 2(Baja Densidad (hipodenso)), 3(Igual Densidad (isodenso)), 4(Alta Densidad (hiperdenso))
    -Microcalcificaciones: 0(No) 1(Si) -Calcificaciones tipicamente benignas: 1(Cutaneas), 2(Vasculares), 3(Gruesas o Pop Corn), 4(Leño o Vara), 5(Redondas o puntiformes), 6(Anulares), 7(Distroficas), 8(Leche de Calcio), 9(Suturas)
    -Calsificaciones morfologia sospechosa: 1(Gruesas heterogeneas), 2(Amorfas), 3(Finas pleomorficas), 4(Lineas finas o lineales ramificadas)
    -Distribicion de las calcificaciones: 1(Difusas), 2(Regionales), 3(Agrupadas (cumulo)), 4(Segmentaria), 5(Lineal)
    -Presencia de asimetrias: 0(No), 1(Si) -Tipo de asimetria: 1(Asimetria), 2(Asimetria global), 3(Asimetria focal), 4(Asimetria focal evolutiva)
    -Hallazgos asociados: 1(Retracción de la piel), 2(Retracción del pezón), 3(Engrosamiento de la piel), 4(Engrosamiento trabecular), 5(Adenopatias axilares)
    -LATERALIDAD HALLAZGO:1(DERECHO), 2(IZQUIERDO), 3(BILATERAL) -BIRADS: 0, 1, 2, 3, 4A, 4B, 4C, 5, 6.
    Solo devuelve el numero, busca en todo el texto, ten en cuenta que cada nota puede contener variaciones del lenguaje y que pueden haber palabras usadas como sinonimos,
    y no necesariamente toda la información está disponible, si ni está disponible escribe NULL, terminos como MICROCALCIFICACIONES y CALCIFICACIONES BENIGNAS
    son importantes para la extracción de información y si existen calcificaciones benignas o no benignas son microcalcificaciones de igual manera. ej: ( Indicaciones: Control anual por antecedente de neoplasia en mama izquierda  tratada con mastectomía.
    Se cuenta con estudios previos de diciembre de 2017 y ecografía de diciembre de 2018.
    Técnica: Se realiza mamografía digital unilateral derecha con compresión graduada en proyecciones CC y MLO con tomosíntesis e imagen sintética 2D.
    Hallazgos: El parénquima mamario es fibroglandular disperso. No hay nódulos, zonas de distorsión ni microcalcificaciones que sugieran malignidad.
    Calcificaciones típicamente benignas.  Nódulo isodenso, de centro hipodenso, de aspecto benigno, estable, de 7 x 6 mm.
    Estables desde 2014. La piel y el complejo areola - pezón son normales. La región axilar tiene ganglios con centro hipodenso.
    No hay cambios evolutivos significativos al comparar con estudios previos.   Conclusión: Hallazgos benignos BIRADS 2.
    Se sugiere continuar con control mamográfico anual.   Dra Camila Morán.
    Residente Radiología.MEDICO RADIOLOGO: Aidee Zoraya Baez Guzman   RM: 3622-92
    2018-12-14 07:06:32) output = nodulos=0, presencia_microcalcificaciones=1, lateralidad_hallazgo=1, birads=2, edad=80  . Revisa todo el texto antes de dar un output'''

    structured_data = []

    for index, row in df.head(2).iterrows():
        presentacion = row['PRESTACION']
        id_paciente = row['ID_DOCUMENTO']
        edad = row['EDAD_EN_FECHA_ESTUDIO']
        notes = row['ESTUDIO']

        response = openai.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": content },
                {"role": "user", "content": f"Add ID_PACIENTE: {id_paciente}, PRESTACION: {presentacion}, EDAD{edad} to each corresponding register. Extract the following information from the notes: {notes}. Look for terms like {search_terms}."}
            ],
            response_format=MedicalNoteExtraction,
        )

        extracted_info = response.choices[0].message.parsed
        print(extracted_info)

        parsed_info = parse_extracted_info(extracted_info)

        # Create the insert query
        insert_query = """
        INSERT INTO DATOS_EXTRAIDOS (ID_PACIENTE, PRESTACION, NODULOS, MORFOLOGIA_NODULOS, MARGENES_NODULOS, DENSIDAD_NODULO, MICROCALCIFICACIONES, CALCIFICACIONES_BENIGNAS, CALCIFICACIONES_SOSPECHOSAS, DISTRIBUCION_CALCIFICACIONES, PRESENCIA_ASIMETRIAS, TIPO_ASIMETRIA, HALLAZGOS_ASOCIADOS, LATERALIDAD_HALLAZGO, BIRADS, EDAD)
        VALUES (:ID_PACIENTE, :PRESTACION, :NODULOS, :MORFOLOGIA_NODULOS, :MARGENES_NODULOS, :DENSIDAD_NODULO, :MICROCALCIFICACIONES, :CALCIFICACIONES_BENIGNAS, :CALCIFICACIONES_SOSPECHOSAS, :DISTRIBUCION_CALCIFICACIONES, :PRESENCIA_ASIMETRIAS, :TIPO_ASIMETRIA, :HALLAZGOS_ASOCIADOS, :LATERALIDAD_HALLAZGO, :BIRADS, :EDAD)
        """
        # Execute the insert query
        with connection.cursor() as cursor:
            cursor.execute(insert_query, parsed_info)
            connection.commit()

        structured_data.append(extracted_info)

    return {"structured_data": structured_data}