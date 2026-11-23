import json
import time
import httpx
import sys

API_URL = "http://localhost:8000/chat"
DATASET_PATH = "eval/dataset.json"

def run_evaluation():
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset de avaliação não encontrado em {DATASET_PATH}")
        sys.exit(1)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"=== Iniciando Avaliação (Eval Harness) com {len(dataset)} casos ===")
    
    results = []
    success_count = 0

    for item in dataset:
        question = item["question"]
        expected_keywords = item["expected_keywords"]
        check_disclaimer = item["check_disclaimer"]

        print(f"\n[Test Case {item['id']}] Pergunta: {question}")
        start_time = time.time()
        
        try:
            response = httpx.post(API_URL, json={"question": question}, timeout=60.0)
            latency = time.time() - start_time
            
            if response.status_code != 200:
                print(f"❌ Erro HTTP {response.status_code}: {response.text}")
                results.append({"id": item["id"], "passed": False, "error": response.text})
                continue
                
            data = response.json()
            answer = data.get("answer", "").lower()
            
            # Checks
            keywords_found = all(kw.lower() in answer for kw in expected_keywords)
            disclaimer_found = "consulte seu médico" in answer if check_disclaimer else True
            
            passed = keywords_found and disclaimer_found
            if passed:
                success_count += 1
                print(f"✅ Passou (Latência: {latency:.2f}s)")
            else:
                print(f"❌ Falhou nas validações. Resposta gerada: {answer}")
                
            results.append({
                "id": item["id"],
                "passed": passed,
                "latency_seconds": latency,
                "answer": answer
            })
            
        except httpx.RequestError as e:
            print(f"❌ Falha de conexão com a API em {API_URL}: {e}")
            print("Certifique-se de que o servidor FastAPI está rodando.")
            sys.exit(1)

    print(f"\n=== Sumário da Avaliação ===")
    print(f"Total: {len(dataset)}")
    print(f"Sucessos: {success_count}")
    print(f"Taxa de Sucesso: {(success_count / len(dataset)) * 100:.1f}%")

if __name__ == "__main__":
    import os
    run_evaluation()
