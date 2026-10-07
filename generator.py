from retriever import retrieve

def generate_answer(query: str, k: int = 3):
    fragments = retrieve(query, k=k)
    
    answer_parts = []
    for frag in fragments:
        answer_parts.append(f"[{frag['id']}] {frag['content'][:250]}...")
        
    answer = (
        "По вашему запросу найдены следующие регламенты для роя БПЛА:\n\n"
        + "\n\n".join(answer_parts)
        + "\n\nРекомендую изучить указанные статьи ITIL полностью."
    )
    
    sources = [
        {"id": f["id"], "title": f["title"], "score": round(f["score"], 3)}
        for f in fragments
    ]
    
    return {"answer": answer, "sources": sources}
