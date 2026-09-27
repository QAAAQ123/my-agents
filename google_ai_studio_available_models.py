from google import genai
import dotenv 
from pathlib import Path

dotenv.load_dotenv(Path(__file__).with_name(".env"))
# 클라이언트 초기화 (환경 변수 GEMINI_API_KEY 자동 인식)
client = genai.Client()

# 사용 가능한 모델 목록 출력
for model in client.models.list():
  print(model.name)