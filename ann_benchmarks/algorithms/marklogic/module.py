import numpy as np
from time import sleep
from marklogic import Client
from marklogic.documents import Document, DefaultMetadata
import subprocess

from ..base.module import BaseANN

class MarkLogicBF(BaseANN):
    def __init__(self, metric):
        self._metric = {"angular": "cosineSimilarity", "euclidean": "euclideanDistance"}[metric]
        self._order = {"angular": "desc", "euclidean": "asc"}[metric]
        self.start_marklogic()
        num_tries = 10
        for _ in range(num_tries):
            sleep(5)
            try:
                healthcheck = Client("http://localhost:7997", digest=("admin", "admin"))
                response = healthcheck.get("/LATEST/healthcheck")
                if response.status_code == 200:
                    break
            except Exception as e:
                print(f"MarkLogic not ready yet: {e}")
        sleep(10)
        if(response.status_code != 200):
            print("MarkLogic not ready after 10 tries. Exiting.")
            exit()
        self._client = Client("http://localhost:8000", digest=("admin", "admin"))

    def start_marklogic(self):
        try:
            subprocess.run(["docker", "compose", "down"])
            subprocess.run(["docker", "compose", "up", "-d", "--build"])
            print("[MarkLogic] docker compose up successful!")
        except Exception as e:
            print(f"[MarkLogic] docker compose up failed: {e}!")

    def stop_marklogic(self):
        try:
            subprocess.run(["docker", "compose", "down"])
            print("[MarkLogic] docker compose down successful!")
        except Exception as e:
            print(f"[MarkLogic] docker compose down failed: {e}!")

    def fit(self, X):
        print("Fitting MarkLogicBF")
        tde_view = {
            "template": {
                "context": "/array-node('embedding')",
                "rows": [{
                    "schemaName": "ann",
                    "viewName": "items",
                    "columns": [
                        {"name": "id", "scalarType": "int", "val": "../id"},
                        {"name": "embedding", "scalarType": "vector", "val": "vec:vector(.)", "dimension": "%d" % X.shape[1]}
                    ]
                }]
            }
        }
        # insert TDE view
        self._client.documents.write(
            Document(
                "/tde/embeddings.json", tde_view,
                permissions={"admin": ["read", "update"]},
                collections=["http://marklogic.com/xdmp/tde"]
            ),
            params={"database": "Schemas"}
        )
        # insert data
        batch_size = 1000
        for i in range(0, len(X), batch_size):
            batch_data = X[i: min(i + batch_size, len(X))]
            documents = [
                Document(
                    "embedding_%d.json" % (i + j), 
                    {"id": (i + j), "embedding": embedding.tolist()},
                    permissions={"admin": ["read", "update"]}
                )
                for j, embedding in enumerate(batch_data)
            ]
            self._client.documents.write(
                documents,
                params={"database": "Documents"}
            )
            if(i+batch_size) % 10000 == 0:
                print("Inserted %dth document" % (i + batch_size))
        print("done!")


    def query(self, v, k):
        query = """
        const op = require('/MarkLogic/optic');
    
        const qv = vec.vector(({queryEmbedding}))
        const rows =
            op.fromView('ann', 'items')
            .bind(op.as('queryvector',qv))
            .bind(op.as('sim', op.vec.{metric}(op.col('embedding'),op.col('queryvector'))))
            .orderBy(op.{order}(op.col('sim')))
            .select(['id'])
            .limit({k})
            .result();
        rows;
        """.format(queryEmbedding=np.array2string(v, separator=',', formatter={'float_kind':lambda x: "%.8f" % x}), metric=self._metric, order=self._order, k=k)
        ids = [item['ann.items.id'] for item in self._client.eval(query)]
        return ids
    
    def done(self):
        self.stop_marklogic()

    def __str__(self):
        return "MarkLogicBF(metric=%s)" % (self._metric)

