"""
RAG (Retrieval-Augmented Generation) engine.

Combines FAISS vector retrieval with LLM generation to answer
questions about screening results, resumes, and job descriptions
with grounded, context-aware responses.
"""

import json
from typing import Optional
from app.config import settings, LLMProvider
from app.utils.helpers import setup_logger, generate_session_id, truncate

log = setup_logger(__name__)


class RAGEngine:
    def __init__(self):
        self._llm_client = None
        self._embedding_engine = None
        self._conversation_memory: dict[str, list[dict]] = {}

    @property
    def embedding_engine(self):
        if self._embedding_engine is None:
            from app.services.embeddings import embedding_engine
            self._embedding_engine = embedding_engine
        return self._embedding_engine

    # ── LLM client initialization ─────────────────────────────

    def _init_llm(self):
        if self._llm_client is not None:
            return

        provider = settings.llm_provider

        if provider == LLMProvider.GEMINI:
            self._init_gemini()
        elif provider == LLMProvider.OLLAMA:
            self._init_ollama()
        elif provider == LLMProvider.OPENAI:
            self._init_openai()
        else:
            log.warning(f"Unknown LLM provider: {provider}, using fallback")
            self._llm_client = "fallback"

    def _init_gemini(self):
        try:
            import google.generativeai as genai
            if not settings.gemini_api_key:
                log.warning("GEMINI_API_KEY not set, LLM features will use fallback")
                self._llm_client = "fallback"
                return
            genai.configure(api_key=settings.gemini_api_key)
            self._llm_client = genai.GenerativeModel(settings.gemini_model)
            log.info(f"Gemini initialized: {settings.gemini_model}")
        except Exception as e:
            log.error(f"Failed to init Gemini: {e}")
            self._llm_client = "fallback"

    def _init_ollama(self):
        try:
            import requests
            resp = requests.get(f"{settings.ollama_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                self._llm_client = "ollama"
                log.info(f"Ollama connected: {settings.ollama_model}")
            else:
                log.warning("Ollama not available, using fallback")
                self._llm_client = "fallback"
        except Exception:
            log.warning("Could not reach Ollama server, using fallback")
            self._llm_client = "fallback"

    def _init_openai(self):
        try:
            from openai import OpenAI
            if not settings.openai_api_key:
                log.warning("OPENAI_API_KEY not set, using fallback")
                self._llm_client = "fallback"
                return
            self._llm_client = OpenAI(api_key=settings.openai_api_key)
            log.info(f"OpenAI initialized: {settings.openai_model}")
        except Exception as e:
            log.error(f"Failed to init OpenAI: {e}")
            self._llm_client = "fallback"

    # ── Retrieval ─────────────────────────────────────────────

    def retrieve_context(self, query: str, top_k: int = 5) -> list[dict]:
        """Fetch relevant resume chunks from the FAISS index."""
        try:
            if self.embedding_engine.index is None:
                self.embedding_engine.load_index()

            if self.embedding_engine.index is None:
                return []

            results = self.embedding_engine.search(query, top_k=top_k)
            return [
                {"content": meta.get("text", ""), "score": float(score), **meta}
                for meta, score in results
            ]
        except Exception as e:
            log.error(f"Retrieval failed: {e}")
            return []

    # ── Generation ────────────────────────────────────────────

    def generate(
        self,
        query: str,
        context: list[dict] = None,
        screening_context: dict = None,
        session_id: str = None,
    ) -> tuple[str, str, list[str]]:
        """
        Generate a response using retrieved context + LLM.
        Returns (response_text, session_id, source_references).
        """
        self._init_llm()

        if session_id is None:
            session_id = generate_session_id()

        # Retrieve relevant context if not provided
        if context is None:
            context = self.retrieve_context(query)

        prompt = self._build_prompt(query, context, screening_context, session_id)
        sources = [c.get("name", f"Source {i+1}") for i, c in enumerate(context) if c]

        response = self._call_llm(prompt)

        # Update memory
        self._add_to_memory(session_id, "user", query)
        self._add_to_memory(session_id, "assistant", response)

        return response, session_id, sources

    def _build_prompt(
        self,
        query: str,
        context: list[dict],
        screening_context: Optional[dict],
        session_id: str,
    ) -> str:
        parts = [
            "You are an AI career advisor and resume screening assistant. "
            "Provide helpful, specific, and actionable advice based on the "
            "context provided. Be professional yet encouraging.\n"
        ]

        # Add screening context if available
        if screening_context:
            parts.append("## Current Screening Results")
            parts.append(json.dumps(screening_context, indent=2, default=str))
            parts.append("")

        # Add retrieved context
        if context:
            parts.append("## Relevant Resume Data")
            for i, ctx in enumerate(context[:5]):
                content = ctx.get("content", ctx.get("text", ""))
                parts.append(f"[Source {i+1}] {truncate(content, 300)}")
            parts.append("")

        # Add conversation history
        history = self._conversation_memory.get(session_id, [])
        if history:
            parts.append("## Previous Conversation")
            for msg in history[-6:]:  # Last 3 exchanges
                role = msg["role"].capitalize()
                parts.append(f"{role}: {truncate(msg['content'], 200)}")
            parts.append("")

        parts.append(f"## User Question\n{query}")
        parts.append(
            "\nRespond concisely and helpfully. If discussing scores, "
            "explain what they mean and how to improve. Reference specific "
            "skills and experience when possible."
        )

        return "\n".join(parts)

    def _call_llm(self, prompt: str) -> str:
        """Route to the active LLM backend."""
        provider = settings.llm_provider

        try:
            if self._llm_client == "fallback" or self._llm_client is None:
                return self._fallback_response(prompt)

            if provider == LLMProvider.GEMINI:
                return self._call_gemini(prompt)
            elif provider == LLMProvider.OLLAMA:
                return self._call_ollama(prompt)
            elif provider == LLMProvider.OPENAI:
                return self._call_openai(prompt)
        except Exception as e:
            log.error(f"LLM call failed: {e}")
            return self._fallback_response(prompt)

        return self._fallback_response(prompt)

    def _call_gemini(self, prompt: str) -> str:
        response = self._llm_client.generate_content(prompt)
        return response.text

    def _call_ollama(self, prompt: str) -> str:
        import requests
        resp = requests.post(
            f"{settings.ollama_url}/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=60,
        )
        return resp.json().get("response", "I couldn't generate a response.")

    def _call_openai(self, prompt: str) -> str:
        response = self._llm_client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800,
        )
        return response.choices[0].message.content

    def _fallback_response(self, prompt: str) -> str:
        """Generate a basic response without an LLM."""
        return (
            "I'm currently operating in offline mode without an LLM connection. "
            "I can still show you the screening results and score breakdowns. "
            "To enable AI-powered conversations, please configure your LLM "
            "provider in the .env file (GEMINI_API_KEY, or set up Ollama locally)."
        )

    # ── Memory management ────────────────────────────────────

    def _add_to_memory(self, session_id: str, role: str, content: str):
        if session_id not in self._conversation_memory:
            self._conversation_memory[session_id] = []
        self._conversation_memory[session_id].append({
            "role": role,
            "content": content,
        })
        # Keep memory bounded
        if len(self._conversation_memory[session_id]) > 20:
            self._conversation_memory[session_id] = \
                self._conversation_memory[session_id][-20:]

    def clear_session(self, session_id: str):
        self._conversation_memory.pop(session_id, None)

    def get_history(self, session_id: str) -> list[dict]:
        return self._conversation_memory.get(session_id, [])


# Singleton
rag_engine = RAGEngine()
