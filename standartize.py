import chromadb
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

class TripletStandardizer:
    def __init__(self):
        self.client = chromadb.Client()

    def init_database(self, collection_name):
        self.collection_name = collection_name
        self.collection = self.client.create_collection(name=collection_name)
        self.relations = set()

    def get_number_of_saved_relations(self):
        return self.collection.count()

    def standartize_triplets(self, triplets, descriptions, threshold=0.15):
        if triplets is None or len(triplets) == 0 or descriptions is None:
            return None, []
        
        substituitons = []
        for triplet in triplets:
            subject, relation, obj = triplet
            relation_desc = descriptions.get(relation, None)

            triplet[1] = (relation, relation_desc)
            
            if relation_desc:
                # Search for similar relations in the database
                results = self.collection.query(
                    query_texts=[relation_desc],
                    n_results=2
                )

                replaced = False
                if results and len(results['distances'][0]) > 0:
                    idx = 0
                    if results['metadatas'][0][0]['relation'] == relation and len(results['distances'][0]) > 1:
                        idx = 1
                    elif results['metadatas'][0][0]['relation'] == relation:
                        continue
                    if results['distances'][0][idx] < threshold:
                    # Replace relation with the most similar one
                        most_similar_relation = results['metadatas'][0][idx]['relation']
                        most_similar_relation_desc = results['documents'][0][idx]
                        triplet[1] = (most_similar_relation, most_similar_relation_desc)
                        replaced = True
                        substituitons.append((subject, (relation, most_similar_relation, results['distances'][0][0]), obj))
                        print(f"Relation '{relation}' replaced with '{most_similar_relation}' (similarity: {results['distances'][0][0]}) [{subject} -> {relation}/{most_similar_relation} -> {obj}]")
                if not replaced:
                    # Store the new relation and its description
                    if relation not in self.relations:
                        self.collection.add(
                            documents=[relation_desc],
                            metadatas=[{"relation": relation}],
                            ids=[relation]
                        )
                        self.relations.add(relation)
            else:
                pass
        
        return triplets, substituitons