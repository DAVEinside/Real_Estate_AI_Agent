try:
      import langchain
      import langgraph
      import crewai
      import langfuse
      import openai
      import pandas
      import numpy
      print("✅ All core dependencies installed successfully!")

      from areip.agents.coordinator import AgentCoordinator
      print("✅ AREIP agents import successfully!")

      print("🎉 Ready to run the full demo!")

except ImportError as e:
      print(f"❌ Missing dependency: {e}")