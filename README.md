# DocSearchAPI

The project provides a RESTful API and web interface for answering questions using the retrieval-augmented generation approach.  
Three search methods are supported: BM25, BERT, and Hybrid (BM25 + BERT).

The API is built using FastAPI and packaged in Docker for fast and easy deployment.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/FedorKaltiugin/DocSearchAPI.git
cd DocSearchAPI
```

Build the Docker image:

```bash
docker build -t docsearch-api .
```

Run the container:

```bash
docker run -d --name docsearch -p 8000:8000 docsearch-api
```

After that, the API will be available at:

```bash
http://0.0.0.0:8000
```

## Experiment Summary
An experiment was conducted in the DocSearch to select the optimal parameters for obtaining answers.
With a small number of relevant documents, BERT finds documents faster than all others in the search,
but with a large number it is inferior to BM25. With an increase in documents for comparing the cosine distance,
the complexity of the calculation also increases, thereby slowing down BERT. Hybrid shows better metrics on large top_n,
but takes a bit more time. With a small number of top_n, the model is unsure of its answers, but answers correctly,
with a large top_n, the model is confident, but its answers are incorrect. The range of normal values is from 3 to 10,
with this range, the time spent on execution is small, and the result is better.
