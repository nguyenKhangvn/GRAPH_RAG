"""
Setup Script - Khởi tạo môi trường cho GraphRAG
"""
from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable
import sys
import subprocess
import os
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# ------------------

def check_python_version():
    """Kiểm tra Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(" Python 3.9+ recommended")
        return False
    print(f" Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Cài đặt dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Cập nhật danh sách thư viện mới nhất
    requirements = [
        "neo4j>=5.18.0",
        "sentence-transformers>=2.6.0",
        "python-dotenv>=1.0.0",
        "google-genai>=0.3.0",  # [QUAN TRỌNG] SDK mới của Google
        "openai>=1.14.0",
        "numpy>=1.26.0"
    ]
    
    # Tạo file requirements.txt tạm nếu chưa có
    req_file_path = os.path.join(current_dir, "requirements.txt")
    if not Path(req_file_path).exists():
        # Tìm ở thư mục cha nếu không thấy trong thư mục hiện tại
        parent_req = os.path.join(parent_dir, "requirements.txt")
        if Path(parent_req).exists():
            req_file_path = parent_req
        else:
            print("   -> Creating requirements.txt...")
            with open("requirements.txt", "w") as f:
                f.write("\n".join(requirements))
            req_file_path = "requirements.txt"
            
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", req_file_path
        ])
        print(" Dependencies installed")
        return True
    except subprocess.CalledProcessError:
        print(" Failed to install dependencies")
        return False

def check_env_file():
    """Kiểm tra .env file"""
    print("\n🔧 Checking configuration...")
    
    # Tìm file .env ở thư mục hiện tại hoặc thư mục cha
    env_path = Path(".env")
    if not env_path.exists():
        env_path = Path(parent_dir) / ".env"

    if not env_path.exists():
        print("  .env file not found")
        # Tạo file mẫu
        sample_env = """
            NEO4J_URI=bolt://localhost:7687
            NEO4J_USER=neo4j
            NEO4J_PASSWORD=your_password
            GEMINI_API_KEY=your_api_key
            OPENAI_API_KEY=sk-...
                GROQ_API_KEY=your_api_key
                XAI_API_KEY=your_api_key
            LLM_MODEL_NAME=gemini-2.0-flash
                PIPELINE_LLM_MODEL_NAME=grok-4.20-reasoning
            EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
        """.strip()
        
        with open(".env", "w") as f:
            f.write(sample_env)
            
        print(" Created .env template")
        print("  PLEASE EDIT .env WITH YOUR REAL KEYS BEFORE CONTINUING!")
        return False
    
    print(" .env file exists")
    return True

def test_neo4j_connection():
    """Test Neo4j connection using Service Layer"""
    print("\n🔌 Testing Neo4j connection...")
    try:
        # [QUAN TRỌNG] Import từ 'services' (số nhiều)
        from graph_rag.services.database import Neo4jService
        
        driver = Neo4jService.get_driver()
        driver.verify_connectivity()
        print(" Neo4j connected successfully")
        Neo4jService.close_driver()
        return True
    except (Neo4jClientError, ServiceUnavailable) as e:
        print(f" Neo4j connection failed: {e}")
        print("   -> Check is Neo4j running?")
        print("   -> Check credentials in .env?")
        return False

def test_llm_connection():
    """Test LLM API using Service Layer"""
    print("\n Testing LLM API...")
    try:
        # [QUAN TRỌNG] Import từ 'services' (số nhiều)
        from graph_rag.services.ai_model import LLMService
        from graph_rag.config import GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY, PIPELINE_LLM_MODEL_NAME, XAI_API_KEY

        model_name = PIPELINE_LLM_MODEL_NAME or ""
        normalized_model = model_name.lower()
        if any(token in normalized_model for token in ["grok", "xai"]):
            api_key = XAI_API_KEY
        elif any(token in normalized_model for token in ["llama", "mixtral", "groq"]):
            api_key = GROQ_API_KEY
        elif "gpt" in normalized_model:
            api_key = OPENAI_API_KEY
        else:
            api_key = GEMINI_API_KEY

        if not api_key or "your_api_key" in api_key:
            print("⏭️  Skipping LLM test (API Key not configured)")
            return True

        llm = LLMService(api_key=api_key, model_name=model_name)
        print(f"   Using Provider: {llm.provider} | Model: {llm.model_name}")
        
        # Test generation đơn giản
        print("   -> Sending test request...")
        response = llm.generate_text("System check", "Say 'Hello' only.")
        if response:
            print(f" LLM Responded: {response.strip()}")
            return True
        else:
            print(" LLM returned empty response")
            return False
            
    except (ValueError, TypeError) as e:
        print(f" LLM Test Failed: {e}")
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║              GraphRAG Setup (Clean Arch)             ║
╚══════════════════════════════════════════════════════════╝
    """)

    steps = [
        check_python_version,
        install_dependencies,
        check_env_file,
    ]

    for step in steps:
        if not step():
            print("\n Setup halted. Please fix the error above.")
            sys.exit(1)

    # Test connections
    test_neo4j_connection()
    test_llm_connection()

    print("\n SETUP COMPLETED! You can now run: python graph_rag/main.py")
    print("   -> For index setup, run: python graph_rag/setup_indexes.py")

if __name__ == "__main__":
    main()