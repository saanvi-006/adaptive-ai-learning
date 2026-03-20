from app.services.document.parser import extract_text
from app.services.document.chunker import chunk_text
from app.services.document.cleaner import clean_text
from app.services.embeddings.embedder import embed_text, embed_query
from app.services.embeddings.vector_store import store_embeddings, retrieve_chunks
from app.services.quiz.generator import generate_quiz
from app.services.flashcards.generator import generate_flashcards
from app.services.quiz.variation import generate_variations

text = extract_text("data/uploads/sample.pdf")
text = clean_text(text)
chunks = chunk_text(text)
embeddings = embed_text(chunks)
store_embeddings(chunks, embeddings)

q_emb = embed_query("What is garbage collection?")
results = retrieve_chunks(q_emb)
print("Generated Quiz:")
print(generate_quiz(results))
print("Generated Flashcards:")
print(generate_flashcards(results))
print("Generated Variations:")
print(generate_variations(generate_quiz(results)))