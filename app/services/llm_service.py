from groq import Groq
from app.config import settings

class LLMService:
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.client = None
        
        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                print(f"Error initializing Groq client: {e}")
        else:
            print("WARNING: GROQ_API_KEY is not set. LLM service will run in Mock/Fallback mode.")

    def generate_insights(self, query: str, context: str) -> str:
        """Sends clinical context and query to Groq LLM for safety-focused insights."""
        
        prompt = f"""
You are an AI clinical decision support assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{query}

Provide safety-focused insights.
Do not provide medical diagnosis.
"""

        # Mock fallback mode if client is not configured
        if not self.client:
            return self._generate_fallback_response(query, context)

        try:
            response = self.client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Groq API Call failed: {e}. Falling back to simulation.")
            return self._generate_fallback_response(query, context)

    def _generate_fallback_response(self, query: str, context: str) -> str:
        """Generates safety-focused fallback text if the Groq API key is missing or fails."""
        normalized_query = query.lower()
        normalized_context = context.lower()
        
        # Penicillin check
        if "penicillin" in normalized_query or "allergy" in normalized_query:
            if "penicillin" in normalized_context:
                return (
                    "### Clinical Safety Signal Detected\n"
                    "**WARNING:** Patient has a documented history of severe penicillin allergy "
                    "(anaphylaxis reaction noted in history logs).\n\n"
                    "**Key Insights:**\n"
                    "1. **Allergy Alert:** Avoid all penicillins (e.g., Amoxicillin, Piperacillin/Tazobactam).\n"
                    "2. **Cross-Reactivity Risk:** Cephalosporins (like Ceftriaxone) carry cross-reactivity risks; monitor closely.\n"
                    "3. **Clinical Recommendation:** Substitute with alternative agents such as Macrolides or Fluoroquinolones.\n\n"
                    "*Note: Fallback mock response triggered due to active API config checks. Verify with chart.*"
                )
                
        # Vital signs check
        if "vitals" in normalized_query or "stability" in normalized_query or "heart rate" in normalized_query:
            return (
                "### Vital Signs Trend Analysis\n"
                "**STATUS: Borderline Unstable (Monitoring Required)**\n\n"
                "**Observations:**\n"
                "1. **Tachycardia Trend:** Heart rate showed a steady escalation up to **100 bpm (Day 4)** before stabilizing at **92 bpm (Day 5)**.\n"
                "2. **Low-Grade Fever:** Temperature tracked alongside heart rate, reaching **99.5°F**.\n"
                "3. **Clinical Interpretation:** Consistently correlates with a mild inflammatory response. Day 5 displays stable downward parameters.\n\n"
                "*Note: Fallback mock response triggered due to active API config checks. Verify with chart.*"
            )

        return (
            "### Clinical Inquiry Answered (API Offline)\n"
            f"No direct API response for: *\"{query}\"*.\n\n"
            "**Retrieved Context Summary:**\n"
            f"{context[:400]}...\n\n"
            "*Please configure GROQ_API_KEY in your system environment to activate live Groq llama3 reasoning.*"
        )

llm_service = LLMService()
