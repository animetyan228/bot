MODEL_PATH = "./model/Qwen2.5-7B-Instruct-Q4_K_M.gguf"  # путь к ллм

N_CTX = 8192 # токены в память при одном запросе
N_THREADS = 8 # потоки кпу
N_GPU_LAYERS = 0 # слои гпу