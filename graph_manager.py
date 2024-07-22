from neo4j import GraphDatabase

class Neo4jTripletManager:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        with self.driver.session() as session:
            session.execute_write(self._clear_all_data)

    @staticmethod
    def _clear_all_data(tx):
        tx.run("MATCH (n) DETACH DELETE n")

    def add_triplets(self, triplets):
        with self.driver.session() as session:
            for triplet in triplets:
                session.execute_write(self._create_relationship, triplet)

    @staticmethod
    def _create_relationship(tx, triplet):
        subject, relation, obj = triplet
        tx.run(
            f"""
            MERGE (a:Entity {{name: $subject}})
            MERGE (b:Entity {{name: $obj}})
            MERGE (a)-[r:{relation}]->(b)
            """, subject=subject, obj=obj
        )

if __name__ == '__main__':
    neo4j_manager = Neo4jTripletManager(uri="bolt://localhost:7687", user="neo4j", password="admin")

    # Initialize a new database
    neo4j_manager.clear_database()

    # Add triplets to the database
    triplets = [
        ["Mount_Everest", "locatedIn", "Himalayas"],
        ["Mount_Everest", "height", "8848_meters"]
    ]
    neo4j_manager.add_triplets(triplets)

    neo4j_manager.close()