from src.upload_paper import upload_graph, SetupDirectories

# Generate workflow diagram and save to .mmd file
print("Generating workflow diagram...")
mermaid_code = upload_graph.mermaid_code(start_node=SetupDirectories)
print(mermaid_code)

# Save diagram code to .mmd file
with open('outputs/upload_paper_diagram.mmd', 'w') as f:
    f.write(mermaid_code)
