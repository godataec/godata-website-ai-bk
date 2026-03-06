import os
import nest_asyncio
from dotenv import load_dotenv
from urllib.parse import urljoin, urlparse
from datetime import datetime

# Patch asyncio just in case, though we are fixing the root cause
nest_asyncio.apply()
load_dotenv()

# IMPORTS
from playwright.async_api import async_playwright
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from rag.tools import book_godata_meeting,validate_email_format,check_team_availability
from langgraph.checkpoint.memory import MemorySaver

# NEW IMPORTS FOR AGENTS
from langchain.agents import create_agent


# --- CONFIGURATION ---
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
TARGET_URL = os.getenv("TARGET_URL") 

class ChatbotBrain:
    def __init__(self):
        self.vector_db = None
        self.chain = None
        # We DO NOT initialize here anymore. 
        # We will call await brain.initialize() from main.py

    async def crawl_website(self, start_url):
        """
        Custom Async Crawler using Playwright.
        Scrapes the homepage and all internal links found.
        """
        print(f"🕷️  Async Crawler: Starting at {start_url}...")
        visited_urls = set()
        documents = []
        
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            # 1. SCRAPE HOMEPAGE & FIND LINKS
            try:
                await page.goto(start_url, wait_until="networkidle")
                
                # Get page content
                content = await page.content()
                documents.append(Document(page_content=content, metadata={"source": start_url}))
                visited_urls.add(start_url)
                
                # Find all links
                anchors = await page.query_selector_all("a")
                for a in anchors:
                    href = await a.get_attribute("href")
                    if href:
                        full_url = urljoin(start_url, href)
                        # Filter for internal links only
                        if urlparse(full_url).netloc == urlparse(start_url).netloc:
                            if "#" not in full_url and full_url not in visited_urls:
                                visited_urls.add(full_url)
            except Exception as e:
                print(f"⚠️ Error scraping {start_url}: {e}")

            # 2. SCRAPE DISCOVERED LINKS
            print(f"🕷️  Found {len(visited_urls)} pages. Scraping them now...")
            
            for url in visited_urls:
                if url == start_url: continue # Skip homepage (already done)
                try:
                    print(f"   scrapping: {url}")
                    await page.goto(url, wait_until="networkidle", timeout=10000)
                    content = await page.inner_text("body") # Better for RAG than raw HTML
                    if content:
                        documents.append(Document(page_content=content, metadata={"source": url}))
                except Exception as e:
                    print(f"   Failed to scrape {url}: {e}")

            await browser.close()
            
        return documents

    async def initialize(self):
        print("🧠 Brain: Warming up...")
        tools=[book_godata_meeting,validate_email_format,check_team_availability] # Add more tools here as you create them
        
        # 1. CRAWL (Async)
        docs = await self.crawl_website(TARGET_URL)
        
        if not docs:
            print("⚠️ Brain: No content found. Chatbot will be empty.")
            return

        print(f"🧠 Brain: Processing {len(docs)} pages into knowledge...")

        # 2. SPLIT
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        
        # 3. EMBED
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self.vector_db = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings,
            collection_name="godata_knowledge"
        )
        
        # 4. LLM
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0
        )
        self.retriever=self.vector_db.as_retriever()

        # 5. CHAIN
        system_prompt = """You are a helpful support specialist for GoData.
        
        INSTRUCTIONS:
        1. **GoData Questions:** If the user asks about GoData, products, or features, you MUST use the "Context" below. Do not make up facts about the company.
        2. **General Software Questions:** If the user asks about general tech concepts (e.g., "What is an API?", "Explain React", "What is RAG?"), you may use your own general knowledge to answer, even if it's not in the Context.
        3. **Refusal:** If the question is completely unrelated to software or business (e.g., "What is the capital of France?", "How to cook pasta?"), politely refuse.
        4. **Tone:** Always maintain a professional, concise, natural and helpful tone. Talk as a GoData member in first person. When answering general tech questions, try to relate them back to GoData if possible (e.g., "APIs are how different software talks to each other. GoData uses APIs to..."). Try not to introduce yourself always.
        5.**Booking Meetings:** If the user wants to book or to schedule a meeting or consultation,or contact with specialist you MUST ask for their:
           - Before booking, use the 'validate_email_format' tool to ensure the user's email is legitimate. If it fails, ask the user for a corrected email.
           - Name
           - Email
           - Company Name
           - Preferred Date (Convert to YYYY-MM-DD format)
           - Preferred Time (Convert to HH:MM 24-hour format)
        6. **Collision Protocol:** - Once you have a Date and Time, you MUST call 'check_team_availability' BEFORE booking.
   -        If there is a conflict, inform the user politely and ask for a different time.
   -        Only call 'book_godata_meeting' if the availability check returns "Success".
        6. **Strict Gatekeeper:** Do NOT call the booking tool until you have gathered ALL 5 pieces of information from the user. Ask clarifying questions if they miss anything.
        7. **Timezone:** Assume all requested times are in Ecuador Time (UTC-5).
        8. **Using the Tool:** Once you have all 5 details, execute the 'book_godata_meeting' tool.
        
        TONE:
        - Professional, concise, and helpful.
        - When answering general tech questions, relate them back to GoData if possible (e.g., "APIs are how different software talks to each other. GoData uses APIs to...").
        """
        self.memory=MemorySaver()
        self.agent_executor = create_agent(model=llm, tools=tools, system_prompt=system_prompt, checkpointer=self.memory)
        
        print("🧠 Brain: ONLINE and Ready! 🚀")

    async def ask(self, query: str, thread_id: str = "default_session"):
        if not getattr(self, 'agent_executor', None):
            return "I am still waking up. Please try again in 10 seconds."
        now=datetime.now()
        current_time_context=f"Current date and time: {now.strftime('%A, %B %d, %Y %H:%M')}"
        # 1. Fetch RAG context manually
        docs = await self.retriever.ainvoke(query)
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        # 2. Inject context
        augmented_query = f"Context from GoData docs:\n{context_text}\n\nUser Question: {query}\n\nCurrent date and time: {current_time_context}"
        
        # 3. Execute with thread_id for memory tracking
        response = await self.agent_executor.ainvoke(
            {"messages": [{"role": "user", "content": augmented_query}]},
            config={"configurable": {"thread_id": thread_id}} # <--- Tells the AI which conversation this is
        )
        
        return response["messages"][-1].content

# Create the instance, but DO NOT initialize it yet
brain = ChatbotBrain()