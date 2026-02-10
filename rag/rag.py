import os
import nest_asyncio
from dotenv import load_dotenv
from urllib.parse import urljoin, urlparse

# Patch asyncio just in case, though we are fixing the root cause
nest_asyncio.apply()
load_dotenv()

# IMPORTS
from playwright.async_api import async_playwright
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

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

        # 5. CHAIN
        template = """You are a helpful support specialist for GoData.
        
        INSTRUCTIONS:
        1. **GoData Questions:** If the user asks about GoData, products, or features, you MUST use the "Context" below. Do not make up facts about the company.
        2. **General Software Questions:** If the user asks about general tech concepts (e.g., "What is an API?", "Explain React", "What is RAG?"), you may use your own general knowledge to answer, even if it's not in the Context.
        3. **Refusal:** If the question is completely unrelated to software or business (e.g., "What is the capital of France?", "How to cook pasta?"), politely refuse.
        4. **Tone:** Always maintain a professional, concise, natural and helpful tone. Talk as a GoData member in first person. When answering general tech questions, try to relate them back to GoData if possible (e.g., "APIs are how different software talks to each other. GoData uses APIs to..."). Try not to introduce yourslef always.
        
        TONE:
        - Professional, concise, and helpful.
        - When answering general tech questions, relate them back to GoData if possible (e.g., "APIs are how different software talks to each other. GoData uses APIs to...").

        Context (Source Code/Docs):
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        retriever = self.vector_db.as_retriever()

        self.chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        print("🧠 Brain: ONLINE and Ready! 🚀")

    async def ask(self, query: str):
        if not self.chain:
            return "I am still waking up (Crawling site). Please try again in 10 seconds."
        return  await self.chain.ainvoke(query)

# Create the instance, but DO NOT initialize it yet
brain = ChatbotBrain()