import json
import logging
import os

import openai
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

MAX_LIST_ITEMS = 12

class SummaryGenerator:

    def __init__(self, api_key=None, llm_model=None):
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY").strip())
        self.model = llm_model or os.getenv("LLM_MODEL").strip()
        self.safety_identifier = os.getenv("SAFETY_IDENTIFIER", "").strip()

    SYSTEM_PROMPT = """Si prijazen podatkovni analitik v trgovskem podjetju.
Uporabnik je zastavil vprašanje o prodajnih podatkih, sistem pa je zanj že izračunal rezultate (JSON spodaj).

Tvoja naloga: napiši kratek, razumljiv povzetek rezultatov v slovenščini za ne-tehničnega uporabnika.

PRAVILA:
1. Uporabljaj IZKLJUČNO številke, ki so v podanem JSON-u. Ničesar ne izračunavaj sam in si ne izmišljuj vrednosti.
2. Zneske zaokroži smiselno in zapiši berljivo (npr. "6,9 milijona €" namesto "6916634.21").
3. Najprej odgovori na vprašanje v 1-2 stavkih, nato po potrebi dodaj 2-4 ključne ugotovitve kot alineje.
4. Poudari trende ("prodaja narašča/pada/je stabilna"), velika odstopanja in anomalije, če so v rezultatih.
5. Če je v rezultatih polje "note" ali je rezultatov 0, povej uporabniku, da za njegovo vprašanje ni podatkov, in predlagaj, kaj naj vpraša drugače.
6. Brez tehničnega žargona (ne omenjaj JSON-a, z-score opiši kot "močno odstopanje od običajnih vrednosti").
7. Če rezultati vsebujejo imena (izdelkov, trgovin, kategorij, znamk), jih v odgovoru IZRECNO poimenuj - "najbolje se je prodajal iPhone 15", ne "nek izdelek". Če rezultat imena nima, ne trdi, da gre za posamezen izdelek/trgovino.
8. Odgovor naj ima manj kot 150 besed."""

    RETRY_PROMPT = """Prejšnji poskus je vseboval števila, ki jih ni bilo mogoče najti v podanih rezultatih: {numbers}.
Povzetek napiši znova in uporabi izključno števila iz podanega JSON-a. Zaokroževanje in berljiv zapis (pravilo 2) ostaneta v veljavi, ne uvajaj pa vrednosti, ki jih v rezultatih ni. Če kakšne številke v rezultatih ni, je v povzetku ne navajaj."""

    def generate(self, user_question: str, params: dict, aggregated: dict,
                 unverified: list = None) -> str:
        payload = {
            "vprasanje_uporabnika": user_question,
            "tip_analize": params["metadata"]["query_type"],
            "rezultati": shrink_for_prompt(aggregated["analysis"]),
        }
        if aggregated.get("supporting"):
            payload["dodatni_rezultati"] = shrink_for_prompt(aggregated["supporting"])

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ]
        # Retry: name the numbers the validator could not ground, instead of just
        # asking again and hoping temperature 0.2 lands somewhere better.
        if unverified:
            messages.append({"role": "user", "content": self.RETRY_PROMPT.format(
                numbers=", ".join(unverified))})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=500,
                safety_identifier=self.safety_identifier or None,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("Summary generation failed: %s", e)
            return f"(Povzetka ni bilo mogoče ustvariti: {e})"


def shrink_for_prompt(obj, max_items: int = MAX_LIST_ITEMS):
    if isinstance(obj, dict):
        return {k: shrink_for_prompt(v, max_items) for k, v in obj.items()}
    if isinstance(obj, list) and len(obj) > max_items:
        half = max_items // 2
        head = [shrink_for_prompt(v, max_items) for v in obj[:half]]
        tail = [shrink_for_prompt(v, max_items) for v in obj[-half:]]
        omitted = len(obj) - 2 * half
        return head + [f"... ({omitted} vmesnih zapisov izpuščenih) ..."] + tail
    if isinstance(obj, list):
        return [shrink_for_prompt(v, max_items) for v in obj]
    return obj