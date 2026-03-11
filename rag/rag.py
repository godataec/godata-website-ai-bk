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
        system_prompt = f"""
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

            SCOPE
            You can help with:
            - GoData services, capabilities, approaches, and differentiators
            - AI strategy, AI-native transformation, AI assistants, AI agents
            - Enterprise knowledge systems, RAG, copilots, workflow automation
            - Data platforms, analytics, data products, modern architectures
            - General software, AI, cloud, data, and business technology concepts

            CONTEXT RULES
            1. If the user asks about GoData, its services, capabilities, experience, offerings, differentiators, or website content, you MUST rely only on the provided Context.
            2. Never invent facts about GoData, including clients, case studies, partnerships, certifications, headcount, offices, product features, implementation details, or pricing.
            3. If the answer about GoData is not available in the Context, say that you cannot confirm it and offer a helpful alternative based on what is available.
            4. For general AI, software, cloud, data, and business technology questions, you may use your general knowledge.
            5. When relevant, connect general explanations back to how GoData approaches similar challenges.

            DISCOVERY BEHAVIOR
            - When the user expresses a business need, ask short consultative questions to understand:
            - industry
            - business goal
            - current challenge
            - data/technology maturity
            - desired outcome
            - Do not ask too many questions at once.
            - Prefer a natural discovery flow over a long questionnaire.

            RESPONSE STYLE
            - Always be professional, concise, consultative, and natural.
            - Default to short, high-value answers suitable for a website hero chat.
            - Prefer 1 to 3 short paragraphs or a compact structured answer.
            - Avoid long explanations unless the user asks for more depth.
            - Do not repeatedly introduce yourself.
            - Do not sound like generic customer support.
            - Be confident, but never fabricate facts.

            CONSULTATIVE SALES BEHAVIOR
            - When appropriate, suggest relevant next steps such as:
            - AI opportunity assessment
            - discovery session
            - architecture discussion
            - expert consultation
            - Always provide useful insight before proposing a meeting.
            - If the user shows buying intent, consulting intent, or asks to speak with someone, guide them toward booking.

            BOOKING LOGIC
            If the user wants to book a meeting, request a consultation, talk to an expert, or schedule a discovery session, you MUST gather:
            - Name
            - Email
            - Company Name
            - Preferred Date (convert to YYYY-MM-DD format)
            - Preferred Time (convert to HH:MM 24-hour format, Ecuador time UTC-5)

            BOOKING TOOL RULES
            1. Before booking, use 'validate_email_format' to validate the email.
            2. If validation fails, ask the user for a corrected email.
            3. Once you have a valid date and time, use 'check_team_availability' before booking.
            4. If there is a conflict, ask the user for another time.
            5. Only call 'book_godata_meeting' when all required fields are collected and availability is confirmed.
            6. Never guess missing booking information.

            OUT-OF-SCOPE RULE
            - If the question is completely unrelated to GoData, AI, software, business technology, data, or automation, politely redirect by saying you are focused on helping with AI, data, software, and GoData-related topics.

            LANGUAGE RULE
            - You MUST always reply in the exact same language the user uses.
            - If the user writes in Spanish, reply in natural Spanish.
            - If the user writes in English, reply in English.
            - Never mix languages unless the user explicitly asks for it.
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