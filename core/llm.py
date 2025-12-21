from llama_cpp import Llama
from crewai import BaseLLM

from config import MODEL_PATH, N_CTX, N_THREADS, N_GPU_LAYERS  # настройки


# загружаем модель
llama_core = Llama(
    model_path=MODEL_PATH, # путь к gguf
    n_ctx=N_CTX, # размер контекста
    n_threads=N_THREADS, # потоки cpu
    n_gpu_layers=N_GPU_LAYERS, # слои на gpu
    use_mmap=True, # быстрее старт, меньше ram
    use_mlock=False, # не фиксируем память
    verbose=False, # меньше логов
)


class LocalQwenLLM(BaseLLM):# обертка чтобы crewai мог юзать ллм

    def __init__(self, llama_obj: Llama, temperature: float = 0.2):
        super().__init__(model="qwen2.5-local", temperature=temperature) # мета инфа для crewai
        self._llama = llama_obj # сохраняем объект модели

    def call(self, messages, tools=None, callbacks=None, **kwargs):  # crewai дергает это
        if isinstance(messages, str): # если пришла строка
            prompt = messages # это уже готовый промпт
        else: # если пришел список сообщений
            parts = [] # сюда собираем текст
            for m in messages:
                if isinstance(m, dict):
                    content = m.get("content", "") # берем текст
                else:
                    content = str(m) # иначе просто в строку
                if content:
                    parts.append(content) # складываем
            prompt = "\n".join(parts) # склеиваем в один промпт

        stop = self.stop if getattr(self, "stop", None) else ["</s>", "###"]  # стоп слова

        result = self._llama(
            prompt,
            max_tokens=250,  # хватит на 3-6 предложений
            temperature=self.temperature,
            top_p=0.9,
            repeat_penalty=1.20,  # главный анти-повтор
            stop=["</s>", "###", "\n\n\n", "\n\nEND"],  # режем длинные простыни
            echo=False,
        )

        text = result["choices"][0]["text"] # достаем текст ответа
        return text.strip()

    def get_context_window_size(self) -> int:
        return N_CTX # говорим crewai размер контекста


local_llm = LocalQwenLLM(llama_core) # готовый llm для агентов
