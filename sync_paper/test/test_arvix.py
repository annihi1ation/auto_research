import json

# Read the arxiv metadata file
with open('arxiv-metadata-oai-snapshot.json', 'r') as f:
    # Read first line to demonstrate content
    first_paper = json.loads(f.readline())

    print("Sample paper metadata:")
    print(f"Title: {first_paper['title']}")
    print(f"Authors: {first_paper['authors']}")
    print(f"Categories: {first_paper['categories']}")
    print(f"Abstract: {first_paper['abstract'][:200]}...")

    # Count total papers
    count = 1
    for line in f:
        count += 1

    print(f"\nTotal number of papers in dataset: {count}")
