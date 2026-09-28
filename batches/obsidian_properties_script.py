"""
{
            "id": "openai/gpt-oss-120b",
            "context_length": 131072,
            "max_output_length": 65536,
            "pricing": {
                "prompt": "0.00000015",
                "completion": "0.0000006",
                "image": "0",
                "request": "0",
                "input_cache_read": "0.000000075"
            }
"""
import os
from time import sleep
import dotenv
import json
from pathlib import Path
from google import genai
from google.genai import types



ROOT_DIR = Path(__file__).resolve().parents[1]
dotenv.load_dotenv(ROOT_DIR / ".env")

MODEL="models/gemini-3.5-flash-lite"
ROOT_PATH = Path(os.environ["OBSIDIAN_ROOT_PATH"])

client = genai.Client()

def create_content_list(batch: list[Path]) -> str:
    contents = []

    for path in batch:
        content = path.read_text(encoding="utf-8")

        contents.append(
            f"""파일 경로: {str(path)}
```markdown
{content}
```"""
        )

    return "\n\n".join(contents)

def create_prompt(batch: list[Path], used_keywords: dict) -> str:

    content_list = create_content_list(batch)

    prompt = f"""
        Analyze Obsidian Markdown files and generate keywords for each file.

        ### Determine Keywords Based on Directory Path and File Name (TOP PRIORITY)
        * The directory path and file name are the PRIMARY basis for keyword decisions
        * Decide keywords from the directory path and file name FIRST
        * Use file content only as supplementary evidence when the path and name are insufficient
        * Example: "/Dev/Java/Spring.md" → ["Java","SpringBoot"]

        ### Existing Keywords
```json
        {used_keywords}
```

        ### Base Keywords
        Java, Investment, CodingTest, Python, ComputerScience, SQL, Pandas, DataBase, RDB, NoSQL, Finance, BackEnd, Infra, DevOps, DataEngineering, AI, Git, TestCode, Logging, Network, Deploy, BlockChain, Redis, Job, School, Project,FastAPI, SpringBoot, Docker

        ### Rules
        1. Prioritize base keywords and existing keywords; create a new keyword only if none fit
        2. Each keyword must be a single word without spaces (CamelCase or PascalCase)
        3. 1 to 5 keywords per file (recommended 3), never empty
        4. Use English only

        ### Output Format
        Output JSON only, with no other text or Markdown. Keys are absolute file paths. Include any new keywords in `used_keywords` (no duplicates).

```json
        {{
            "file_path1": ["keyword1","keyword2"],
            "file_path2": ["keyword1","keyword3"],
            "used_keywords": ["keyword1","keyword2","keyword3"]
        }}
```

        ### Files to Analyze

        {content_list}

    """

    return prompt

def request_to_gemini(prompt: str) -> str:
    try:
        chat = client.chats.create(
            model=MODEL,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        response = chat.send_message(prompt)
    except Exception as e:
        print(f"Gemini API 오류 발생: {e}")
        raise
    else:
        return response.text

def clean_result(qroq_result: str) -> tuple[dict, list[str]]:
    validated, properties, used_keywords = validate_keyword_result(qroq_result)

    if not validated:
        raise ValueError("GenAi가 잘못된 형식의 응답을 반환했습니다.")

    return properties, used_keywords

def write_properties(properties: dict) -> None:
    for path, values in properties.items():
        print(f"파일: {path}\n태그: {values}")

        file_path = Path(path)

        if not os.path.exists(file_path):
            print(f"파일 없음: {file_path}")
            continue 

        with open(file_path, "r+", encoding="utf-8") as md:
            content = md.read()

            md.seek(0)
            md.write(parse_properties(values) + content)
            md.truncate()

def parse_properties(tags: list) -> str:
    tags_str = "\n".join(f"  - {tag}" for tag in tags)
    return f"""---
tags:
{tags_str}
---
"""

def remove_frontmatter(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)

    if not lines or lines[0].strip() != "---":
        return

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            path.write_text(
                "".join(lines[index + 1:]),
                encoding="utf-8",
            )
            return

def validate_keyword_result(
    qroq_result: str,
) -> tuple[bool, dict, list[str]]:
    try:
        result_dict = json.loads(qroq_result)
    except json.JSONDecodeError:
        return False, {}, []

    if not isinstance(result_dict, dict):
        return False, {}, []

    used_keywords = result_dict.get("used_keywords")
    if not isinstance(used_keywords, list):
        return False, {}, []

    if not all(isinstance(keyword, str) for keyword in used_keywords):
        return False, {}, []

    if len(used_keywords) != len(set(used_keywords)):
        return False, {}, []

    file_results = {
        path: keywords
        for path, keywords in result_dict.items()
        if path != "used_keywords"
    }

    if not file_results:
        return False, {}, []

    for file_path, keywords in file_results.items():
        if not isinstance(file_path, str):
            return False, {}, []

        if not isinstance(keywords, list):
            return False, {}, []

        if len(keywords) > 5:
            return False, {}, []

        if not all(isinstance(keyword, str) for keyword in keywords):
            return False, {}, []

        if not set(keywords).issubset(used_keywords):
            return False, {}, []

    print(f"Groq 배치 완료 파일 개수: {len(file_results)}")
    return True, file_results, used_keywords

def main():
    """
    이전에 작성한 키워드들을 토대로 작성해야하기 때문에 배치의 의존성이 있음
    코루틴을 사용하면 정확성이 떨어짐
    """
    print(f"파일 루트 경로: {ROOT_PATH}")
    paths = list(ROOT_PATH.rglob("*.md"))
    #print(f"파일 개수: {len(paths)}")
    batches = [paths[i: i+10] for i in range(0, len(paths), 10)]
    print(f"배치 개수: {len(batches)}")

    all_used_keywords: list[str] = []

    for idx, batch in enumerate(batches, start=1):
        print(f"배치 {idx}/{len(batches)} 처리 중...")

        for file_idx, path in enumerate(batch, start=(idx - 1) * 10 + 1):
            print(f"파일 {file_idx}/{len(paths)}: {path}")
            remove_frontmatter(path)

        prompt = create_prompt(batch, all_used_keywords)
        groq_result = request_to_gemini(prompt)

        properties, used_keywords = clean_result(groq_result)

        all_used_keywords.extend(
            keyword for keyword in used_keywords
            if keyword not in all_used_keywords
        )

        write_properties(properties)

        sleep(5)

    print(
            f"배치 {idx}/{len(batches)} 완료 "
            f"(누적 키워드: {len(used_keywords)}개)"
        )

    print(f"전체 키워드 개수: {len(all_used_keywords)}개")

if __name__ == "__main__":
    main()
        



    
