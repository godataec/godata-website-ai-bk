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
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from rag.tools import book_godata_meeting,validate_email_format,check_team_availability
from langgraph.checkpoint.memory import MemorySaver

# NEW IMPORTS FOR AGENTS
from langchain.agents import create_agent


# --- CONFIGURATION ---
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
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
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 4})

        # 5. CHAIN
        # 5. CHAIN
        system_prompt = """You are Jose, an AI Architect and friendly support specialist for GoData.
        
        INSTRUCTIONS:
        1. GoData Questions: Use the provided Context below to answer. Do not make up facts.
        2. General Software Questions: Answer general tech concepts using your own knowledge. Relate them back to GoData's services if possible.
        3. Refusal: Politely refuse questions completely unrelated to software, data, or business.
        4. Natural Lead-In & Booking: If a user expresses general interest, briefly answer their question and naturally invite them to book a quick consultation. Do not interrogate them. Ask for their booking details conversationally, one or two at a time.
           *You MUST eventually gather ALL 5 of these details to book:*
           - Name
           - Email
           - Company Name
           - Preferred Date (Convert to YYYY-MM-DD format)
           - Preferred Time (Convert to HH:MM 24-hour format. Assume Ecuador Time UTC-5).
        5. Email Validation: Once you have the email, run the 'validate_email_format' tool. If it fails, gently ask the user for a corrected email.
        6. Calendar Check: Once you have a valid Date and Time, you MUST run the 'check_team_availability' tool BEFORE booking.
        7. Final Booking: ONLY run the 'book_godata_meeting' tool if you have gathered all 5 pieces of info AND the availability check was successful.
        8. Tone: Professional, warm, and conversational. Speak in the first person ("I"). Act like a helpful consultant, not a robotic form.
        """
        self.memory=MemorySaver()
        self.agent_executor = create_agent(model=llm, tools=tools, system_prompt=system_prompt, checkpointer=self.memory)
        
        print("🧠 Brain: ONLINE and Ready! 🚀")


    async def ask(self, query: str, thread_id: str = "default_session"):
        if not getattr(self, 'agent_executor', None):
            return "I am still waking up. Please try again in 10 seconds."
            
        now = datetime.now()
        current_time_context = f"Current Date and Time: {now.strftime('%A, %B %d, %Y %H:%M')}"
        
        # --- THE TOKEN SAVER ---
        # If the user just says "Hi", "Thanks", or a very short greeting, don't waste tokens searching the database.
        if len(query.split()) <= 3:
            context_text = "No specific website context needed for this short message."
        else:
            # Only run the heavy RAG search if they actually ask a real question
            docs = await self.retriever.ainvoke(query)
            context_text = "\n\n".join([doc.page_content for doc in docs])
        # ------------------------
        
        augmented_query = f"""{current_time_context}
        
        Context from GoData docs:
        {context_text}

        User Question: {query}

        FINAL RULE: You MUST answer the above "User Question" in the EXACT SAME LANGUAGE that the question is written in. If the question is in English, your response MUST be in English. If it is in Spanish, answer in natural Spanish."""
        
        response = await self.agent_executor.ainvoke(
            {"messages": [{"role": "user", "content": augmented_query}]},
            config={"configurable": {"thread_id": thread_id}} 
        )
        
        return response["messages"][-1].content

# Create the instance, but DO NOT initialize it yet
brain = ChatbotBrain()