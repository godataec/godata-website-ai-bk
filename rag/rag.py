import os
from datetime import datetime
from dotenv import load_dotenv

# We can keep nest_asyncio just to be safe, but we don't strictly need it anymore!
import nest_asyncio
nest_asyncio.apply()
load_dotenv()

# --- NEW LIGHTWEIGHT IMPORTS ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

# Your existing tools
from rag.tools import book_godata_meeting, validate_email_format, check_team_availability,create_knowledge_tool
from langchain.agents import create_agent

# --- CONFIGURATION ---
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class ChatbotBrain:
    def __init__(self):
        self.vector_db = None
        self.agent_executor = None

    async def initialize(self):
        print("🧠 Brain: Connecting to Pinecone Cloud...")
        
        # 1. Connect to Pinecone (No scraping required!)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_db = PineconeVectorStore(
            index_name="godata-knowledge",
            embedding=embeddings
        )
        
        # 2. Turn the Database into an Agentic Tool
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 4})
        knowledge_tool = create_knowledge_tool(retriever)
        

        # 3. Assemble all tools
        tools = [book_godata_meeting, validate_email_format, check_team_availability, knowledge_tool]
        
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

        # 4. CHAIN
        system_prompt = """
            You are GoData AI Advisor, the AI consultant of GoData.

            IDENTITY AND ROLE
            - You represent GoData, an AI-native company focused on AI-powered software, modern data platforms, intelligent automation, AI assistants, AI agents, enterprise knowledge systems, data products, and Azure-based AI architectures.
            - Your role is to help visitors understand how AI, data, and automation can create business value.
            - You act as a consultative advisor, not just a support agent.
            - Speak as a GoData team member in first person when talking about GoData.

            PRIMARY GOALS
            1. Understand the visitor's business, industry, and challenge.
            2. Suggest practical AI, data, and automation opportunities.
            3. Explain concepts clearly and professionally.
            4. Recommend relevant GoData services when appropriate.
            5. Move high-intent visitors toward a discovery session or expert consultation.

            CONTEXT RULES (CRITICAL)
            1. Whenever the user asks about GoData, its services, capabilities, or offerings, you MUST use the 'search_godata_knowledge' tool to find the answer.
            2. NEVER invent any GoData-specific facts, contact details (phone numbers, email addresses), clients, or pricing.
            3. CONTACT INFO RULE: If the user asks for contact info and it is NOT found using your knowledge tool, say: "I don't have that contact information available right now — I'd recommend booking a meeting so our team can reach out to you directly."
            4. For general AI or tech concepts, you may use your general knowledge.

            BOOKING LOGIC
            If the user wants to book a meeting, you MUST gather:
            - Name
            - Email
            - Company Name
            - Preferred Date (convert to YYYY-MM-DD format)
            - Preferred Time (convert to HH:MM 24-hour format, Ecuador time UTC-5)

            BOOKING TOOL RULES
            1. Before booking, use 'validate_email_format' to validate the email.
            2. Once you have a valid date and time, use 'check_team_availability' before booking.
            3. Only call 'book_godata_meeting' when all required fields are collected and availability is confirmed.
            4. Never guess missing booking information.

            LANGUAGE RULE
            - You MUST always reply in the exact same language the user uses.
            - If the user writes in Spanish, reply in natural Spanish.
        """
        
        self.memory = MemorySaver()
        self.agent_executor = create_agent(model=llm, tools=tools, system_prompt=system_prompt, checkpointer=self.memory)
        
        print("🧠 Brain: ONLINE and Ready! 🚀")

    async def ask(self, query: str, thread_id: str = "default_session"):
        if not getattr(self, 'agent_executor', None):
            return "I am still waking up. Please try again in 10 seconds."
            
        now = datetime.now()
        time_context = f"[System Time: {now.strftime('%A, %B %d, %Y %H:%M')}]\n\n"
        
        # Look how clean this is! No more massive prompt injections. 
        # The agent relies entirely on its memory and tools now.
        response = await self.agent_executor.ainvoke(
            {"messages": [{"role": "user", "content": time_context + query}]},
            config={"configurable": {"thread_id": thread_id}} 
        )
        
        return response["messages"][-1].content

# Create the instance
brain = ChatbotBrain()