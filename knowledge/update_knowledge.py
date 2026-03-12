import os
import asyncio
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv

# Fix User Agent warning
os.environ["USER_AGENT"] = "GoData-AI-Scraper/1.0"

from playwright.async_api import async_playwright
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load API keys (OpenAI and Pinecone) from .env
load_dotenv()

async def crawl_website(start_url):
    """
    Custom Async Crawler using Playwright.
    Scrapes the homepage and all internal links found.
    """
    print(f"🕷️ Async Crawler: Starting at {start_url}...")
    visited_urls = set()
    documents = []
    
    async with async_playwright() as p:
        # Launch headless browser
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        # 1. SCRAPE HOMEPAGE & FIND LINKS
        try:
            await page.goto(start_url, wait_until="networkidle")
            
            # Get page content as clean text
            content = await page.inner_text("body")
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
        print(f"🕷️ Found {len(visited_urls)} pages. Scraping them now...")
        
        for url in visited_urls:
            if url == start_url: continue # Skip homepage (already done)
            try:
                print(f"   scrapping: {url}")
                await page.goto(url, wait_until="networkidle", timeout=10000)
                content = await page.inner_text("body")
                if content:
                    documents.append(Document(page_content=content, metadata={"source": url}))
            except Exception as e:
                print(f"   Failed to scrape {url}: {e}")

        await browser.close()
        
    return documents

async def inject_data():
    base_url = "https://godata-website-ai.lovable.app"
    
    # 1. Run the Playwright crawler
    docs = await crawl_website(base_url)

    # 2. Sanity Check
    print("-" * 40)
    print(f"🔍 SANITY CHECK - Snippet from homepage:")
    if docs:
        print(docs[0].page_content[:300].strip())
    print("-" * 40)

    # 3. Process and Upload
    print("✂️ Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    print("☁️ Pushing data to Pinecone Cloud...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    PineconeVectorStore.from_documents(
        splits,
        embeddings,
        index_name="godata-knowledge"
    )
    
    print("✅ Success! Your Pinecone database is now fully loaded with REAL text.")

if __name__ == "__main__":
    # Because Playwright is async, we must run the script using asyncio
    asyncio.run(inject_data())