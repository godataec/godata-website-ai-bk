import os
import nest_asyncio
from dotenv import load_dotenv
from urllib.parse import urljoin, urlparse

# Apply async patch for Playwright
nest_asyncio.apply()
load_dotenv()

# IMPORTS
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from playwright.sync_api import sync_playwright

# --- CONFIGURATION ---
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
TARGET_URL = os.getenv("TARGET_URL")

class ChatbotBrain:
    def __init__(self):
        self.vector_db = None
        self.chain = None
        self.initialize()

    def find_internal_links(self, start_url):
        """
        Uses Playwright to visit the homepage and find all sub-pages 
        that belong to the same website.
        """
        print(f"🕷️  Crawler: Visiting {start_url} to find links...")
        found_urls = {start_url} # Use a set to avoid duplicates
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(start_url, wait_until="networkidle") # Wait for React to load
                
                # Get all anchor tags
                anchors = page.query_selector_all("a")
                
                for a in anchors:
                    href = a.get_attribute("href")
                    if href:
                        # Convert relative paths (/about) to full URLs (https://.../about)
                        full_url = urljoin(start_url, href)
                        
                        # Only keep links that stay on your domain (don't crawl Twitter/LinkedIn)
                        if urlparse(full_url).netloc == urlparse(start_url).netloc:
                            # Filter out junk like #section or mailto:
                            if "#" not in full_url and "mailto:" not in full_url:
                                found_urls.add(full_url)
                
                browser.close()
        except Exception as e:
            print(f"⚠️ Link discovery failed: {e}")
            
        print(f"🕷️  Crawler: Found {len(found_urls)} unique pages: {list(found_urls)}")
        return list(found_urls)

    def initialize(self):
        # 1. DISCOVER LINKS
        # Instead of just loading TARGET_URL, we first find all its sub-pages
        all_pages = self.find_internal_links(TARGET_URL)

        print(f"🧠 Brain: Scraping {len(all_pages)} pages...")
        
        # 2. LOAD CONTENT (Using Playwright for all pages)
        loader = PlaywrightURLLoader(
            urls=all_pages,
            remove_selectors=["header", "footer", "nav"], 
            continue_on_failure=True
        )
        
        try:
            docs = loader.load()
            print(f"   Successfully scraped content from {len(docs)} pages.")
        except Exception as e:
            print(f"⚠️ Error during scraping: {e}")
            return

        # 3. SPLIT
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        
        # 4. EMBED
        print("   Loading embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self.vector_db = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings,
            collection_name="godata_knowledge"
        )
        
        # 5. LLM
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0
        )

        # 6. PROMPT
        template = """You are a helpful support specialist for GoData.
        
        INSTRUCTIONS:
        1. **GoData Questions:** If the user asks about GoData, products, or features, you MUST use the "Context" below. Do not make up facts about the company.
        2. **General Software Questions:** If the user asks about general tech concepts (e.g., "What is an API?", "Explain React", "What is RAG?"), you may use your own general knowledge to answer, even if it's not in the Context.
        3. **Refusal:** If the question is completely unrelated to software or business (e.g., "What is the capital of France?", "How to cook pasta?"), politely refuse.
        4. **Tone:** Always maintain a professional, concise, and helpful tone. Talk as a GoData member in first person. When answering general tech questions, try to relate them back to GoData if possible (e.g., "APIs are how different software talks to each other. GoData uses APIs to...").
        
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
        
        print("🧠 Brain: Ready and loaded with full site knowledge!")

    def ask(self, query: str):
        if not self.chain:
            return "System is initializing..."
        return self.chain.invoke(query)

brain = ChatbotBrain()

if __name__ == "__main__":
    # Simple test
    question = "What is GoData?"
    print(f"❓ Question: {question}")
    answer = brain.ask(question)
    print(f"💡 Answer: {answer}")