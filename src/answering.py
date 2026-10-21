from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

SYSTEM_PROMPT = """You are NABH Assistant, a careful hospital-accreditation reference assistant.
Answer using ONLY the supplied evidence passages. Do not invent clauses, objective-element IDs, scores, deadlines, legal requirements, or official interpretations.
If the evidence is insufficient, clearly say so and recommend checking the applicable official NABH standard.
Distinguish what the cited document explicitly states from practical implementation suggestions and legal requirements needing separate verification.
Use concise headings and practical hospital examples when helpful.
Cite claims inline using [PDF page N] based on the source labels provided.
Do not claim this chatbot is official or guarantees accreditation. Requirements may vary by edition, programme and scope.
Never provide patient-specific diagnosis or treatment. For clinical emergencies, direct the user to local emergency services or qualified clinicians.
"""

def answer_question(question: str, evidence: list[dict], api_key: str, model: str = "gpt-4o-mini"):
    context_parts = [
        f"[SOURCE {i} | {item.get('label', 'source')} | chunk {item.get('chunk_id', 'n/a')}]\n{item['text']}"
        for i, item in enumerate(evidence, 1)
    ]
    context = "\n\n---\n\n".join(context_parts)
    llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"QUESTION:\n{question}\n\nEVIDENCE PASSAGES:\n{context}\n\nAnswer with page citations."),
    ])
    answer = response.content if isinstance(response.content, str) else str(response.content)
    return {"answer": answer, "sources": evidence}
