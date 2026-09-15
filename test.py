from agent.graph import build_graph, run_agent
graph = build_graph()
print(run_agent(graph, "explique-moi le théorème de Pythagore", "math", "test-1"))
print(run_agent(graph, "pose-moi une question", "chimie", "test-2"))