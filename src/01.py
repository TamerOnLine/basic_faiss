import os
import logging
import pandas as pd
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.memory import ConversationBufferMemory

# تهيئة logging لمراقبة العمليات
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# تهيئة نموذج الذكاء الاصطناعي Ollama LLM
llm = OllamaLLM(model="llama3.2")

# تهيئة نموذج تحويل النصوص إلى متجهات Embeddings
embeddings = OllamaEmbeddings(model="locusai/multi-qa-minilm-l6-cos-v1")

# مسار قاعدة بيانات FAISS
FAISS_PATH = "faiss_chat_index"

def initialize_faiss():
    """تحميل أو إنشاء قاعدة بيانات FAISS."""
    if os.path.exists(os.path.join(FAISS_PATH, "index.faiss")):
        logging.info("📂 تحميل قاعدة بيانات FAISS...")
        return FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    logging.info("⚠️ قاعدة بيانات FAISS غير موجودة، يتم إنشاؤها...")
    faiss_index = FAISS.from_texts(["مرحبًا بك في قاعدة البيانات!"], embeddings)
    faiss_index.save_local(FAISS_PATH)
    logging.info("✅ قاعدة بيانات FAISS تم إنشاؤها بنجاح.")
    return faiss_index

faiss_index = initialize_faiss()

def search_faiss(query: str, k: int = 3, threshold: float = 0.9):
    """البحث في قاعدة بيانات FAISS عن أفضل تطابق للطلب."""
    try:
        results = faiss_index.similarity_search_with_score(query, k)
        for doc, score in results:
            if "→ Bot: " in doc.page_content and score >= threshold:
                stored_question, stored_answer = doc.page_content.split("→ Bot: ")
                if stored_question.strip().lower() == query.strip().lower():
                    return stored_answer
    except Exception as e:
        logging.error(f"❌ خطأ أثناء البحث في FAISS: {e}")
        return None
    
    return None

def store_chat(query: str, response: str):
    """تخزين محادثة جديدة داخل FAISS."""
    if search_faiss(query, k=1, threshold=0.85):
        logging.warning("⚠️ السؤال موجود مسبقًا في FAISS، يتم تجاهل التخزين.")
        return
    
    formatted_text = f"User: {query} → Bot: {response}"
    faiss_index.add_texts([formatted_text])
    faiss_index.save_local(FAISS_PATH)
    logging.info("✅ تم حفظ المحادثة داخل FAISS.")

def get_response(query: str):
    """إرجاع استجابة إما من FAISS أو Ollama LLM."""
    response = search_faiss(query)
    
    if response is None:
        logging.info("🧠 لم يتم العثور على إجابة في FAISS، يتم استخدام Ollama...")
        response = llm.invoke(query)
        store_chat(query, response)
    
    return response

def load_data_from_file(file_path: str):
    """تحميل البيانات من ملف CSV أو TXT إلى FAISS."""
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path, encoding="utf-8")
            texts = df.iloc[:, 0].dropna().tolist()
        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                texts = [line.strip() for line in file.readlines() if line.strip()]
        else:
            logging.error("⚠️ تنسيق الملف غير مدعوم! استخدم CSV أو TXT.")
            return
        
        logging.info(f"📥 يتم تحميل {len(texts)} إدخالات إلى FAISS...")
        faiss_index.add_texts(texts)
        faiss_index.save_local(FAISS_PATH)
        logging.info("✅ تم تحميل البيانات بنجاح إلى FAISS.")
    except Exception as e:
        logging.error(f"❌ خطأ أثناء تحميل الملف: {e}")

# إعداد الأدوات لاستخدامها داخل الوكيل الذكي
search_tool = Tool(
    name="FAISS Exact Search",
    func=search_faiss,
    description="يبحث في قاعدة بيانات FAISS عن إجابة مباشرة ودقيقة."
)

llm_tool = Tool(
    name="Ollama LLM",
    func=llm.invoke,
    description="يستخدم نموذج Ollama LLM للإجابة على الاستفسارات في حال عدم العثور على إجابة في FAISS."
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=[search_tool, llm_tool],
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

def chat_with_faiss():
    """تشغيل جلسة محادثة تفاعلية باستخدام FAISS و Ollama."""
    logging.info("🚀 تشغيل الدردشة المعتمدة على FAISS...")
    print("🔹 اكتب 'exit' لإنهاء المحادثة.")
    
    while True:
        try:
            query = input("You: ").strip()
            if query.lower() == "exit":
                break
            
            response = agent.run(query)
            print(f"Bot: {response}")
        except Exception as e:
            logging.error(f"❌ خطأ أثناء المحادثة: {e}")
            print("حدث خطأ، يرجى المحاولة مرة أخرى.")

if __name__ == "__main__":
    chat_with_faiss()
