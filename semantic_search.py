from sentence_transformers import SentenceTransformer
import numpy as np

jobs = [
    "Python backend engineer building REST APIs and SQL databases",
    "Machine learning engineer working on recommendation systems",
    "Frontend developer using React and TypeScript",
    "Embedded systems engineer programming microcontrollers in C",
    "Data analyst creating dashboards with SQL and Excel",
    "LLM engineer building AI agents with tool calling",
    "DevOps engineer managing Docker and AWS deployments",
    "VLSI design engineer working on Verilog and FPGA",
    "Wireless communication engineer working on RF and antennas",
    "Technical account manager supporting enterprise customers",
    "Cybersecurity analyst handling network security",
    "Mobile app developer building Android apps with Kotlin",
]

model = SentenceTransformer("all-MiniLM-L6-v2")
job_vecs = model.encode(jobs, normalize_embeddings=True)

def search(query, k=3):
    q = model.encode([query], normalize_embeddings=True)[0]
    scores = job_vecs @ q
    for i in np.argsort(-scores)[:k]:
        print(f"{scores[i]:.2f}  {jobs[i]}")

while True:
    query = input("\nSearch jobs (or q to quit): ")
    if query.lower() == "q":
        break
    search(query)