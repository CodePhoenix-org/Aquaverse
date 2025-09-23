import pandas as pd
from neo4j import GraphDatabase

# --- Step 1: Data Preprocessing ---
# Read the CSV file
df = pd.read_csv('argo_labeled.csv')

# Define bins for continuous features
bins = {
    'temperature': pd.qcut(df['temperature'], 4, labels=['Low', 'Medium', 'High', 'Very High']),
    'oxygen': pd.qcut(df['oxygen'], 4, labels=['Low', 'Medium', 'High', 'Very High']),
    'chlorophyll': pd.qcut(df['chlorophyll'], 4, labels=['Low', 'Medium', 'High', 'Very High']),
    'salinity': pd.qcut(df['salinity'], 4, labels=['Low', 'Medium', 'High', 'Very High']),
    'depth': pd.qcut(df['depth'], 4, labels=['Shallow', 'Medium', 'Deep', 'Very Deep'])
}

# Add the binned categories to the dataframe
for feature, binned_values in bins.items():
    df[f'{feature}_binned'] = binned_values

# --- Step 2: Connect to Neo4j and Build the Graph ---
uri = "bolt://localhost:7687"
username = "neo4j"
password = "neo4jsumukhi"

def create_structured_graph(tx, row_data):
    """
    Creates a more structured knowledge graph for anomaly prediction.
    """
    # Create the Measurement node with all its properties
    tx.run("""
        CREATE (m:Measurement {
            latitude: $latitude,
            longitude: $longitude,
            depth: $depth,
            temperature: $temperature,
            oxygen: $oxygen,
            chlorophyll: $chlorophyll,
            salinity: $salinity,
            anomaly: $anomaly
        })
    """, **row_data)

    # Connect the Measurement node to its binned feature nodes
    for feature in bins.keys():
        query = (
            f"MERGE (f:{feature.capitalize()} {{category: $category}})"
            " WITH f"
            " MATCH (m:Measurement {latitude: $latitude, longitude: $longitude, depth: $depth})"
            f" MERGE (m)-[:HAS_{feature.upper()}]->(f)"
        )
        tx.run(query,
               category=row_data[f'{feature}_binned'],
               latitude=row_data['latitude'],
               longitude=row_data['longitude'],
               depth=row_data['depth'])

    # Connect the Measurement node to the Anomaly status node
    anomaly_status = 'YES' if row_data['anomaly'] == 1 else 'NO'
    query = (
        "MERGE (a:Anomaly {status: $status})"
        " WITH a"
        " MATCH (m:Measurement {latitude: $latitude, longitude: $longitude, depth: $depth})"
        " MERGE (m)-[:RESULTED_IN]->(a)"
    )
    tx.run(query,
           status=anomaly_status,
           latitude=row_data['latitude'],
           longitude=row_data['longitude'],
           depth=row_data['depth'])

# Main execution block
with GraphDatabase.driver(uri, auth=(username, password)) as driver:
    with driver.session() as session:
        # Clear the database before creating the new graph
        session.run("MATCH (n) DETACH DELETE n")

        for index, row in df.iterrows():
            row_data = row.to_dict()
            # Use the newer execute_write function
            session.execute_write(create_structured_graph, row_data)

print("Structured knowledge graph created successfully!")