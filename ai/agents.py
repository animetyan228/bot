# агенты crew ai

from crewai import Agent
from core.llm import local_llm

from ai.tools import (
    pdf_reader_tool,
    build_vector_index_tool,
    rag_search_tool,
)

pdf_agent = Agent(
    role="PDF Reader",
    goal="Извлечь текст из PDF.",
    backstory="Специалист по извлечению текста из документов.",
    llm=local_llm,
    tools=[pdf_reader_tool],
    allow_delegation=False,
    verbose=False,
)

chunk_agent = Agent(
    role="Indexer",
    goal="Подготовить документ для поиска (векторизация и сохранение).",
    backstory="Специалист по подготовке данных для семантического поиска.",
    llm=local_llm,
    tools=[build_vector_index_tool],
    allow_delegation=False,
    verbose=False,
)

retriever_agent = Agent(
    role="Retriever",
    goal="Найти релевантные фрагменты документа по запросу пользователя.",
    backstory="Специалист по семантическому поиску фрагментов в документе.",
    llm=local_llm,
    tools=[rag_search_tool],
    allow_delegation=False,
    verbose=False,
)

answer_agent = Agent(
    role="Answer Generator",
    goal="Ответить строго по контексту документа и привести цитату.",
    backstory="Аналитик документов. Не использует внешние знания.",
    llm=local_llm,
    allow_delegation=False,
    verbose=False,
)

refine_agent = Agent(
    role="Editor",
    goal="Сделать ответ яснее и убрать воду, сохранив смысл и факты.",
    backstory="Редактор деловых текстов.",
    llm=local_llm,
    allow_delegation=False,
    verbose=False,
)
