from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse
import torch
import os
from pydantic import BaseModel
from moviepy.editor import VideoFileClip
from tempfile import NamedTemporaryFile
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import aiofiles
import asyncio

# FastAPI 인스턴스 생성
app = FastAPI()

# Task 변수 선언
current_task = None

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Whisper 모델 로드
print("Whisper 모델 로딩 중...")
whisper_model = whisper.load_model("small").to(device)
print("Whisper 모델 로딩 완료")

# GPT 모델과 토크나이저 초기화
model, tokenizer = None, None
model_id = "Qwen/Qwen2-7B-Instruct"

def load_gpt_model_and_tokenizer():
    global model, tokenizer
    if model is None or tokenizer is None:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 4bit 양자화 활성화
            llm_int8_enable_fp32_cpu_offload=False,  # CPU 오프로딩 비활성화 (GPU만 사용)
            bnb_4bit_compute_dtype=torch.bfloat16,  # 4080 GPU에서 bfloat16을 사용하여 계산 최적화
            bnb_4bit_quant_type="nf4",  # NF4 양자화 유형 사용 (FP4보다 높은 정확도와 안정성)
            llm_int8_has_fp16_weight=True  # LLM.int8()과 함께 16-bit 가중치 사용 (백워드 패스 최적화)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

model, tokenizer = load_gpt_model_and_tokenizer()

async def extract_audio_from_mp4(mp4_file_path, output_audio_path="extracted_audio.wav"):
    """MP4 파일에서 오디오를 추출하여 WAV 파일로 저장"""
    loop = asyncio.get_event_loop()
    video = await loop.run_in_executor(None, VideoFileClip, mp4_file_path)
    audio = video.audio
    await loop.run_in_executor(None, audio.write_audiofile, output_audio_path)  # codec 인자 제거
    return output_audio_path

async def transcribe_audio_file(audio_file_path):
    """WAV, M4A, MP3 등 파일을 텍스트로 변환"""
    loop = asyncio.get_event_loop()
    audio = await loop.run_in_executor(None, whisper.load_audio, audio_file_path)
    result = await loop.run_in_executor(None, whisper_model.transcribe, audio)
    return result['text']

@app.post("/transcribe_video")
async def transcribe_video(file: UploadFile = File(...)):
    global current_task
    try:
        current_task = asyncio.create_task(_transcribe_video_logic(file))
        transcription = await current_task
        return PlainTextResponse(content=transcription)
    except asyncio.CancelledError:
        raise HTTPException(status_code=400, detail="작업이 중지되었습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        current_task = None

async def _transcribe_video_logic(file: UploadFile):
    async with aiofiles.tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_mp4:
        content = await file.read()
        await temp_mp4.write(content)
        temp_mp4_path = temp_mp4.name

    temp_audio_path = "extracted_audio.wav"
    audio_file_path = await extract_audio_from_mp4(temp_mp4_path, temp_audio_path)
    transcription = await transcribe_audio_file(audio_file_path)

    os.remove(temp_mp4_path)
    os.remove(temp_audio_path)

    return transcription

@app.post("/transcribe_audio")
async def transcribe_audio(file: UploadFile = File(...)):
    global current_task
    try:
        current_task = asyncio.create_task(_transcribe_audio_logic(file))
        transcription = await current_task
        return PlainTextResponse(content=transcription)
    except asyncio.CancelledError:
        raise HTTPException(status_code=400, detail="작업이 중지되었습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        current_task = None

async def _transcribe_audio_logic(file: UploadFile):
    async with aiofiles.tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        content = await file.read()
        await temp_audio.write(content)
        temp_audio_path = temp_audio.name

    transcription = await transcribe_audio_file(temp_audio_path)
    os.remove(temp_audio_path)

    return transcription

class PromptRequest(BaseModel):
    prompt: str
    context: str

@app.post("/generate")
async def generate_response(request: PromptRequest):
    global current_task
    try:
        current_task = asyncio.create_task(_generate_response_logic(request))
        response = await current_task
        return {"response": response}
    except asyncio.CancelledError:
        raise HTTPException(status_code=400, detail="작업이 중지되었습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        current_task = None

async def _generate_response_logic(request: PromptRequest):
    messages = [
        {"role": "system", "content": request.prompt},
        {"role": "user", "content": request.context}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)

    # 작업 도중에 이벤트 루프에 양보
    await asyncio.sleep(0)
    
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=4096,
        top_k=50,
        top_p=0.9,
        temperature=0.2
    )
    
    # 작업 도중에 이벤트 루프에 양보
    await asyncio.sleep(0)
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

@app.post("/emergency_stop")
async def emergency_stop():
    global current_task
    if current_task is not None:
        current_task.cancel()
        current_task = None
        return PlainTextResponse(content="작업이 중지되었습니다.")
    else:
        return PlainTextResponse(content="중지할 작업이 없습니다.", status_code=404)



def send_request_to_model_server(context, query):
    try:
        response = requests.post(API_URL, json={"prompt": query, "context": context})
        response.raise_for_status()
        return response.json().get("response")
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while requesting model: {e}")
        return None


@app.post("/similarity")
async def check_answer_with_llm(request: PromptRequest):
    try:
        messages = [
            {"role": "system", "content": request.prompt},
            {"role": "user", "content": request.context}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)

        # 작업 도중에 이벤트 루프에 양보
        await asyncio.sleep(0)
        
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            top_k=50,
            top_p=0.9,
            temperature=0.1
        )
        
        # 작업 도중에 이벤트 루프에 양보
        await asyncio.sleep(0)
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return {"response": response_text}  # "response" 키로 응답 반환
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM API 호출 실패: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)