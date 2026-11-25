import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Organized data set
data = [
    {
        "question": "What does a data scientist do?",
        "context": "A data scientist analyzes data to extract insights using statistics and machine learning.",
        "answer": "A data scientist analyzes data to extract insights.",
    },
    {
        "question": "What is machine learning?",
        "context": "Machine learning is a subset of AI that enables systems to learn from data.",
        "answer": "Machine learning allows systems to learn from data.",
    },
]

df = pd.DataFrame(data)
# Each row includes the input question, the ground truth context and the expected answer.

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = df["context"].tolist()
doc_embeddings = model.encode(documents)

index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))


def retrieve(query, k=1):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]


generator = pipeline("text-generation", model="distilgpt2")


def generate_answer(query, context):
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    # return_full_text=False so the prediction excludes the prompt (which contains the context)
    output = generator(
        prompt, max_new_tokens=50, num_return_sequences=1, return_full_text=False
    )
    # distilgpt2 rambles on; keep only the first line as the answer
    return output[0]["generated_text"].strip().split("\n")[0]


def rag_pipeline(question):
    retrieved_context = retrieve(question)[0]
    answer = generate_answer(question, retrieved_context)
    return retrieved_context, answer


results = []

for _, row in df.iterrows():
    context, prediction = rag_pipeline(row["question"])

    results.append(
        {
            "question": row["question"],
            "ground_truth": row["answer"],
            "prediction": prediction,
            "retrieved_context": context,
        }
    )

results_df = pd.DataFrame(results)


# LLM Metrics
def similarity_score(a, b):
    emb1 = model.encode([a])
    emb2 = model.encode([b])
    return cosine_similarity(emb1, emb2)[0][0]


# first check cosine similarity
results_df["similarity"] = results_df.apply(
    lambda row: similarity_score(row["ground_truth"], row["prediction"]), axis=1
)

# next check the context relevance
results_df["context_score"] = results_df.apply(
    lambda row: similarity_score(row["retrieved_context"], row["ground_truth"]), axis=1
)


# check groundedness
def groundedness(answer, context):
    return int(any(word in context for word in answer.split()))


results_df["groundedness"] = results_df.apply(
    lambda row: groundedness(row["prediction"], row["retrieved_context"]), axis=1
)

print("Average Similarity:", results_df["similarity"].mean())
print("Average Context Score:", results_df["context_score"].mean())
print("Groundedness Rate:", results_df["groundedness"].mean())
