import requests
from math import ceil
from requests.adapters import HTTPAdapter, Retry

API_KEY = "cDZHYzlZa0JadVREZDJCendQbXY6SkJlTzNjLV9TRENyQk1RdnFKZGRQdw=="
HEADERS = {"Authorization": f"APIKey {API_KEY}", "Content-Type": "application/json"}

session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504], allowed_methods=["POST"])
session.mount("https://", HTTPAdapter(max_retries=retries))

def pega_total_trt(trt_num):
    index = f"api_publica_trt{trt_num}"
    url = f"https://api-publica.datajud.cnj.jus.br/{index}/_search"
    # size=0 retorna só os metadados (total) sem carregar os hits
    payload = {"size": 0, "query": {"match_all": {}}}
    resp = session.post(url, headers=HEADERS, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    total = data["hits"]["total"]["value"]
    return total

def main():
    tamanho_lote = 1000
    resultados = []
    for trt in range(1, 25):
        try:
            total = pega_total_trt(trt)
            n_lotes = ceil(total / tamanho_lote) if total > 0 else 0
            resultados.append((f"TRT{trt:02d}", total, n_lotes))
            print(f"TRT{trt:02d}: {total:,} processos → {n_lotes} lotes de {tamanho_lote}")
        except Exception as e:
            print(f"TRT{trt:02d}: ERRO ao consultar -> {e}")
    # se quiser, retorna a lista
    return resultados

if __name__ == "__main__":
    main()
