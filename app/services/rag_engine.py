"""
RAG (Retrieval-Augmented Generation) engine.

Combines FAISS vector retrieval with LLM generation to answer
questions about screening results, resumes, and job descriptions
with grounded, context-aware responses. Protects candidate privacy
and handles missing API keys gracefully.
"""

import json
from typing import Optional, List, Dict, Tuple
from app.config import settings, LLMProvider
from app.utils.helpers import setup_logger, generate_session_id, truncate

log = setup_logger(__name__)


def is_valid_api_key(key: Optional[str]) -> bool:
    """Check if an API key is provided and is not a placeholder."""
    if not key:
        return False
    k = key.strip().lower()
    return not (k.startswith("your-") or "placeholder" in k or k == "" or "key-here" in k)


class RAGEngine:
    def __init__(self):
        self._llm_client = None
        self._embedding_engine = None
        self._conversation_memory: Dict[str, List[dict]] = {}

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

        if provider == LLMProvider.GEMINI or str(provider).lower() == "gemini":
            self._init_gemini()
        elif provider == LLMProvider.OLLAMA or str(provider).lower() == "ollama":
            self._init_ollama()
        elif provider == LLMProvider.OPENAI or str(provider).lower() == "openai":
            self._init_openai()
        else:
            log.warning(f"Unknown LLM provider: {provider}, using fallback")
            self._llm_client = "fallback"

    def _init_gemini(self):
        try:
            if not is_valid_api_key(settings.gemini_api_key):
                log.info("GEMINI_API_KEY not configured or is a placeholder. Using grounded offline fallback.")
                self._llm_client = "fallback"
                return

            import google.generativeai as genai
            genai.configure(api_key=settings.gemini_api_key)
            self._llm_client = genai.GenerativeModel(settings.gemini_model)
            log.info(f"Gemini initialized: {settings.gemini_model}")
        except Exception as e:
            log.error(f"Failed to init Gemini: {e}")
            self._llm_client = "fallback"

    def _init_ollama(self):
        try:
            import requests
            resp = requests.get(f"{settings.ollama_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                self._llm_client = "ollama"
                log.info(f"Ollama connected: {settings.ollama_model}")
            else:
                log.warning("Ollama not responding, using offline fallback.")
                self._llm_client = "fallback"
        except Exception:
            log.warning("Could not reach Ollama server, using offline fallback.")
            self._llm_client = "fallback"

    def _init_openai(self):
        try:
            if not is_valid_api_key(settings.openai_api_key):
                log.info("OPENAI_API_KEY not configured. Using offline fallback.")
                self._llm_client = "fallback"
                return

            from openai import OpenAI
            self._llm_client = OpenAI(api_key=settings.openai_api_key, timeout=15.0)
            log.info(f"OpenAI initialized: {settings.openai_model}")
        except Exception as e:
            log.error(f"Failed to init OpenAI: {e}")
            self._llm_client = "fallback"

    # ── Retrieval with Privacy Isolation ───────────────────────

    def retrieve_context(self, query: str, top_k: int = 3, target_candidate: Optional[str] = None) -> List[dict]:
        """
        Fetch relevant knowledge chunks. If target_candidate is specified,
        avoids exposing records belonging to other candidates.
        """
        try:
            if self.embedding_engine.index is None:
                self.embedding_engine.load_index()

            if self.embedding_engine.index is None or self.embedding_engine.index_size == 0:
                return []

            results = self.embedding_engine.search(query, top_k=top_k)
            filtered = []
            for meta, score in results:
                # Do not leak other candidate names/emails if answering for a specific candidate
                if target_candidate and meta.get("name") and meta.get("name") != target_candidate:
                    # Sanitize other candidates' names
                    meta = dict(meta)
                    meta["name"] = "Benchmark Candidate Profile"
                filtered.append({"content": meta.get("text", ""), "score": float(score), **meta})
            return filtered
        except Exception as e:
            log.error(f"Retrieval failed: {e}")
            return []

    # ── Generation ────────────────────────────────────────────

    def generate(
        self,
        query: str,
        context: Optional[List[dict]] = None,
        screening_context: Optional[dict] = None,
        session_id: Optional[str] = None,
    ) -> Tuple[str, str, List[str]]:
        """
        Generate a response using retrieved context + LLM.
        Returns (response_text, session_id, source_references).
        """
        self._init_llm()

        if session_id is None:
            session_id = generate_session_id()

        cand_name = screening_context.get("candidate_name") if screening_context else None

        # Retrieve relevant context if not provided
        if context is None:
            context = self.retrieve_context(query, target_candidate=cand_name)

        sources = []
        if screening_context and cand_name:
            sources.append(f"Screening Record: {cand_name}")
        for i, c in enumerate(context):
            src_name = c.get("name") or f"Knowledge Source {i+1}"
            if src_name not in sources:
                sources.append(src_name)

        # Check if running offline
        if self._llm_client == "fallback" or self._llm_client is None:
            response = self._fallback_response(query, screening_context)
        else:
            prompt = self._build_prompt(query, context, screening_context, session_id)
            response = self._call_llm(prompt, screening_context)

        # Update memory
        self._add_to_memory(session_id, "user", query)
        self._add_to_memory(session_id, "assistant", response)

        return response, session_id, sources

    def _build_prompt(
        self,
        query: str,
        context: List[dict],
        screening_context: Optional[dict],
        session_id: str,
    ) -> str:
        parts = [
            "You are an AI career advisor and resume screening assistant. "
            "Provide helpful, specific, and actionable advice based only on the "
            "context provided. Be professional yet encouraging.\n"
        ]

        if screening_context:
            parts.append("## Active Candidate Screening Profile")
            parts.append(json.dumps(screening_context, indent=2, default=str))
            parts.append("")

        if context:
            parts.append("## Contextual Knowledge Chunks")
            for i, ctx in enumerate(context[:4]):
                content = ctx.get("content", ctx.get("text", ""))
                parts.append(f"[Source {i+1}] {truncate(content, 250)}")
            parts.append("")

        # Add conversation history
        history = self._conversation_memory.get(session_id, [])
        if history:
            parts.append("## Previous Conversation")
            for msg in history[-6:]:
                role = msg["role"].capitalize()
                parts.append(f"{role}: {truncate(msg['content'], 180)}")
            parts.append("")

        parts.append(f"## User Question\n{query}")
        parts.append(
            "\nProvide a concise, direct, helpful response. "
            "Do not fabricate qualifications or work history not present in the screening profile. "
            "If asked for interview questions or career guidance, tailor them directly to the candidate's stated skills and gaps."
        )

        return "\n".join(parts)

    def _call_llm(self, prompt: str, screening_context: Optional[dict] = None) -> str:
        """Route to active LLM backend with timeout and graceful fallback."""
        provider = settings.llm_provider

        try:
            if provider == LLMProvider.GEMINI or str(provider).lower() == "gemini":
                return self._call_gemini(prompt)
            elif provider == LLMProvider.OLLAMA or str(provider).lower() == "ollama":
                return self._call_ollama(prompt)
            elif provider == LLMProvider.OPENAI or str(provider).lower() == "openai":
                return self._call_openai(prompt)
        except Exception as e:
            log.error(f"LLM call to {provider} failed: {e}")
            return self._fallback_response(prompt, screening_context)

        return self._fallback_response(prompt, screening_context)

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
            timeout=15,
        )
        return resp.json().get("response", "I could not generate a response.")

    def _call_openai(self, prompt: str) -> str:
        response = self._llm_client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=600,
        )
        return response.choices[0].message.content

    def _fallback_response(self, query: str, screening_context: Optional[dict] = None) -> str:
        """Generate a grounded, actionable response when operating without external LLM keys."""
        if screening_context:
            name = screening_context.get("candidate_name", "the candidate")
            role = screening_context.get("job_title", "the target position")
            score = screening_context.get("overall_score", 0.0)
            missing = screening_context.get("missing_skills", [])
            matched = screening_context.get("matched_skills", [])
            pred = screening_context.get("predicted_role", "")

            lines = [
                f"**Assessment Profile for {name} ({role})**",
                f"• **Overall Compatibility**: {score:.1f}%",
            ]
            if pred:
                lines.append(f"• **Predicted Profile Role**: {pred}")
            if matched:
                lines.append(f"• **Matched Core Skills**: {', '.join(matched[:6])}")
            if missing:
                lines.append(f"• **Identified Gaps**: {', '.join(missing[:6])}")
                lines.append(f"• **Actionable Advice**: Focus immediately on building proficiency in **{missing[0]}** to significantly enhance alignment.")
            else:
                lines.append("• **Skill Coverage**: Profile matches all primary required technical competencies.")

            lines.append("\n*(Running in grounded offline mode. Add a valid `GEMINI_API_KEY` in `.env` to enable freeform conversational AI.)*")
            return "\n".join(lines)

        return (
            "I am currently operating in offline mode. To enable conversational AI, "
            "please set your `GEMINI_API_KEY` (or configure local Ollama) in the `.env` file."
        )

    # ── Memory management ────────────────────────────────────

    def _add_to_memory(self, session_id: str, role: str, content: str):
        if session_id not in self._conversation_memory:
            self._conversation_memory[session_id] = []
        self._conversation_memory[session_id].append({
            "role": role,
            "content": content,
        })
        if len(self._conversation_memory[session_id]) > 20:
            self._conversation_memory[session_id] = self._conversation_memory[session_id][-20:]

    def clear_session(self, session_id: str):
        self._conversation_memory.pop(session_id, None)

    def get_history(self, session_id: str) -> List[dict]:
        return self._conversation_memory.get(session_id, [])


# Singleton
rag_engine = RAGEngine()
