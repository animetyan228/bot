from crewai import Task, Crew, Process

from ai.agents import answer_agent, refine_agent
from ai.tools import pdf_reader_tool, build_vector_index_tool, rag_search_tool


def _to_raw_text(result) -> str:
    return getattr(result, "raw", str(result))

def _clean_answer(text: str) -> str:
    # выкидываем мусорные служебные строки
    bad = ["end of response", "i've ensured", "markdown", "```"]
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if any(b in low for b in bad):
            continue
        lines.append(s)

    text = " ".join(lines)

    # режем повторы по предложениям (оставляем максимум 4)
    parts = [p.strip() for p in text.replace("!", ".").replace("?", ".").split(".") if p.strip()]
    seen = set()
    out = []
    for p in parts:
        key = " ".join(p.lower().split())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= 4:
            break

    if not out:
        return text.strip()

    return ". ".join(out).strip() + "."



def _pick_evidence(context: str) -> str:
    if not context:
        return ""
    parts = [p.strip() for p in context.split("\n\n---\n\n") if p.strip()]
    for p in parts:
        if len(p) >= 80:
            return p[:900]
    return parts[0][:900] if parts else ""


def _expand_query(question: str) -> str:
    q = question.lower()
    extra = []
    if any(w in q for w in ["телефон", "мобиль", "смартфон", "съем", "съём", "видео", "аудио", "камера", "запись"]):
        extra.append("использование электронных устройств мобильные телефоны фото видео аудио съемка запись согласие запрещается разрешается")
    if any(w in q for w in ["перемена", "урок", "школа"]):
        extra.append("на уроке на перемене во внеурочное время на территории школы")
    return question + " " + " ".join(extra) if extra else question


def run_pipeline(pdf_path: str, question: str) -> tuple[str | None, str | None, str | None, str | None]:
    # returns: answer, context, evidence, error

    try:
        # читаем pdf
        doc_text = pdf_reader_tool.run(pdf_path=pdf_path)

        # защита от “пустого/обрезанного” pdf
        if len(doc_text) < 1500:
            return None, None, None, "pdf прочитался слишком коротко. проверь что pdf не скан и что текст реально извлекается."

        # 2) строим индекс НАПРЯМУЮ (без агента)
        index_id = build_vector_index_tool.run(text=doc_text)

        # 3) rag поиск НАПРЯМУЮ (без агента)
        query = _expand_query(question)
        context = rag_search_tool.run(index_id=index_id, query=query, top_k=3)

        evidence = _pick_evidence(context)

        # 4) генерация ответа агентом
        answer_task = Task(
            description=(
                "тебе дан контекст из документа и вопрос.\n"
                "ответь строго по контексту, внешние знания запрещены.\n"
                "обязательно вставь 1 короткую точную цитату из контекста (слово в слово), 1 раз.\n"
                "если в контексте нет ответа — напиши: "
                "\"в предоставленном документе нет информации, чтобы ответить на этот вопрос.\".\n"
                "нельзя: markdown, код-блоки, символ #, служебные фразы.\n"
                "каждое предложение должно добавлять новую информацию, без перефразирования одного и того же. \n"
                "ответ строго 4 предложений.\n"
                "вставь ровно 1 цитату слово в слово в кавычках, отдельным предложением. \n\n"
                f"контекст:\n{context}\n\n"
                f"вопрос: {question}\n"
            ),
            agent=answer_agent,
            expected_output="Ответ (3-6 предложений) с одной короткой точной цитатой.",
        )

        # 5) чистим ответ агентом
        refine_task = Task(
            description=(
                "отредактируй ответ.\n"
                "правила:\n"
                "1) убери воду, удали все повторяющиеся предложения. \n"
                "2) оставь ровно 1 цитату\n"
                "3) не добавляй новых фактов\n"
                "4) нельзя: markdown, код-блоки, символ #, служебные фразы (например: 'End of response')\n"
                "верни только чистый текст."
            ),
            agent=refine_agent,
            expected_output="Короткий финальный ответ без мусора и без служебных фраз.",
            context=[answer_task],
        )

        crew_answer = Crew(
            agents=[answer_agent, refine_agent],
            tasks=[answer_task, refine_task],
            process=Process.sequential,
            verbose=False,
        )

        result = crew_answer.kickoff()
        final_answer = _to_raw_text(result).strip()
        final_answer = _clean_answer(final_answer)

        return final_answer, context, evidence, None

    except Exception as e:
        return None, None, None, str(e)
