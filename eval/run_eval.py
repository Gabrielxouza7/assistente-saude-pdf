import argparse
import json
import os
import sys
import time

# Import datetime utilities and HTTPX client for evaluation API requests
from datetime import datetime, timezone
import httpx

# Define API endpoint URL and file paths for evaluation datasets and failed cases
API_URL = "http://localhost:8000/chat"
DATASET_PATH = "eval/dataset.json"
FAILED_PATH = "eval/failed_cases.json"

# Persist newly failed evaluation cases to disk while avoiding duplicate questions
def save_failed_cases(failed_items: list):

    # Return early if there are no failed items to save
    if not failed_items:
        return

    # Initialize list and load existing failed cases from disk if the file exists
    existing = []
    if os.path.exists(FAILED_PATH):
        with open(FAILED_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)

    # Filter out failed items whose normalized questions are already recorded
    existing_questions = {item["question"].strip().lower() for item in existing}
    new_failures = [
        item for item in failed_items
        if item["question"].strip().lower() not in existing_questions
    ]

    # Append new failures to the existing list and write the updated JSON to disk
    if new_failures:
        existing.extend(new_failures)
        with open(FAILED_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"\n⚠️ {len(new_failures)} prompt(s) reprovado(s) salvo(s) em {FAILED_PATH}")

# Append a new regression test case to the evaluation dataset if not already present
def add_regression_case(question: str, keywords: list[str], check_disclaimer: bool = True):
    dataset = []

    # Load existing evaluation dataset records if the file exists
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)

    # Abort insertion if an identical question already exists in the dataset
    if any(item["question"].strip().lower() == question.strip().lower() for item in dataset):
        print("Este prompt já existe no eval/dataset.json.")
        return

    # Assign the next sequential ID and construct the new regression test object
    next_id = max((item.get("id", 0) for item in dataset), default=0) + 1
    new_case = {
        "id": next_id,
        "question": question,
        "expected_keywords": keywords,
        "check_disclaimer": check_disclaimer
    }

    # Append the new test case and save the updated dataset back to disk
    dataset.append(new_case)
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"✅ Caso de regressão #{next_id} adicionado ao {DATASET_PATH}")

# Execute all test cases in the evaluation dataset against the live chat API
def run_evaluation():

    # Exit with an error code if the evaluation dataset file is missing
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset de avaliação não encontrado em {DATASET_PATH}")
        sys.exit(1)

    # Load evaluation test cases and initialize tracking counters and result lists
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"=== Iniciando Avaliação (Eval Harness) com {len(dataset)} casos ===")
    results = []
    failed_to_record = []
    success_count = 0

    # Iterate through each evaluation test case and send its question to the API
    for item in dataset:
        question = item["question"]
        history = item.get("history", [])
        expected_keywords = item["expected_keywords"]
        check_disclaimer = item["check_disclaimer"]
        print(f"\n[Test Case {item['id']}] Pergunta: {question}")
        start_time = time.time()

        # Send the HTTP POST request to the chat endpoint and validate the response
        try:
            response = httpx.post(
                API_URL,
                json={"question": question, "history": history},
                timeout=180.0
            )
            latency = time.time() - start_time

            # Record a failure and skip further checks if the HTTP status is not 200
            if response.status_code != 200:
                print(f"❌ Erro HTTP {response.status_code}: {response.text}")
                results.append({"id": item["id"], "passed": False, "error": response.text})
                continue

            # Parse response fields and evaluate missing keywords and medical disclaimer
            data = response.json()
            raw_answer = data.get("answer", "")
            answer = raw_answer.lower()
            cached = data.get("cached", False)
            missing_keywords = [kw for kw in expected_keywords if kw.lower() not in answer]
            disclaimer_found = "consulte seu médico" in answer if check_disclaimer else True

            # Increment success counter when all expected keywords and disclaimers match
            passed = (len(missing_keywords) == 0) and disclaimer_found
            if passed:
                success_count += 1
                cache_tag = " [CACHE HIT]" if cached else ""
                print(f"✅ Passou (Latência: {latency:.2f}s){cache_tag}")

            # Log detailed failure metadata when keyword or disclaimer validation fails
            else:
                print(f"❌ Falhou nas validações. Resposta gerada:\n{raw_answer}")
                failed_to_record.append({
                    "original_id": item["id"],
                    "question": question,
                    "expected_keywords": expected_keywords,
                    "check_disclaimer": check_disclaimer,
                    "missing_keywords": missing_keywords,
                    "disclaimer_found": disclaimer_found,
                    "model_answer": raw_answer,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

            # Store the individual test case execution metrics in the results list
            results.append({
                "id": item["id"],
                "passed": passed,
                "cached": cached,
                "latency_seconds": latency,
                "answer": raw_answer
            })

        # Terminate evaluation immediately if the API server cannot be reached
        except httpx.RequestError as e:
            print(f"❌ Falha de conexão com a API em {API_URL}: {e}")
            print("Certifique-se de que o servidor FastAPI está rodando.")
            sys.exit(1)

    # Persist all failed cases and print the final evaluation summary statistics
    save_failed_cases(failed_to_record)
    print(f"\n=== Sumário da Avaliação ===")
    print(f"Total: {len(dataset)}")
    print(f"Sucessos: {success_count}")
    print(f"Falhas: {len(dataset) - success_count}")
    print(f"Taxa de Sucesso: {(success_count / len(dataset)) * 100:.1f}%")

# Parse command-line arguments to either add a regression case or run full evaluation
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval Harness do Assistente de Saúde RAG")
    parser.add_argument("--add-question", type=str, help="Adiciona um prompt que o modelo errou ao dataset.json")
    parser.add_argument("--keywords", nargs="+", default=[], help="Palavras-chave esperadas na resposta")
    parser.add_argument("--no-disclaimer", action="store_true", help="Não exigir o aviso médico neste caso")
    args = parser.parse_args()

    # Add a new regression test case when the --add-question flag is provided
    if args.add_question:
        add_regression_case(
            question=args.add_question,
            keywords=args.keywords,
            check_disclaimer=not args.no_disclaimer
        )

    # Run the full evaluation suite by default when no question addition is requested
    else:
        run_evaluation()