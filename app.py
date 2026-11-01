import os
import re
import uuid
from typing import TypedDict, Optional, List

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# ===================================================
# PAGE CONFIG - must be first Streamlit call
# ===================================================
st.set_page_config(
    page_title="ShopEasy Bot",
    page_icon="🛍️",
    layout="wide",
)


# =====================================================
# KNOWLEDGE BASE - 10 FAQ Documents
# =====================================================
FAQ_DOCUMENTS = [
    """ShopEasy Return Policy:
Customers can return any item within 30 days of delivery.
Items must be unused, in original packaging, with all tags attached.
To initiate a return, go to My Orders > Request Return.
Refunds are processed within 5-7 business days to the original payment method.
Items marked as Final Sale or Non-Returnable cannot be returned.
Electronics must be returned within 15 days and include all accessories.
ShopEasy covers return shipping costs for defective or wrong items.
For all other returns, a Rs.99 return shipping fee is deducted from the refund.""",

    """ShopEasy Shipping and Delivery:
Standard shipping takes 3-5 business days and costs Rs.49 for orders under Rs.499.
Orders above Rs.499 qualify for FREE standard shipping.
Express shipping (1-2 business days) costs Rs.149 and is available in metro cities.
Same-day delivery is available in select pin codes for orders placed before 11 AM.
International shipping is available to 20+ countries; rates vary by destination.
You will receive a tracking link via email and SMS once your order is shipped.
Deliveries are attempted twice; after two failed attempts the package is returned.""",

    """ShopEasy Payment Methods:
ShopEasy accepts UPI (GPay, PhonePe, Paytm), credit cards, debit cards, net banking,
EMI (3, 6, 12 months on orders above Rs.3000), and ShopEasy Wallet.
Cash on Delivery (COD) is available for orders up to Rs.5000.
All transactions are secured with 256-bit SSL encryption.
Failed payments are automatically refunded within 3-5 business days.
ShopEasy does NOT store card details — payments go through PCI-DSS certified gateways.
International cards (Visa, Mastercard, Amex) are accepted with no extra charges.""",

    """ShopEasy Discount and Coupon Codes:
Apply coupon codes at checkout in the Have a coupon? box.
Only one coupon code can be used per order.
WELCOME10: 10% off your first order (max discount Rs.200).
SUMMER20: 20% off on clothing and footwear (min order Rs.999).
FREESHIP: Free shipping on any order (no minimum).
FESTIVE15: 15% off sitewide during festival sales.
Coupons cannot be combined with ongoing flash sale prices.
Expired coupons will show an error message at checkout.
ShopEasy loyalty points can be redeemed as cash (100 points = Rs.10).""",

    """ShopEasy Order Tracking:
Track your order at shopeasy.com/track or in the ShopEasy app under My Orders.
You receive an SMS and email with a tracking link within 24 hours of shipping.
Tracking updates every 4 hours from our logistics partners.
If tracking shows Delivered but you have not received the package, wait 24 hours.
For prepaid orders, disputes trigger an automatic investigation by our delivery team.
Order status: Confirmed > Packed > Shipped > Out for Delivery > Delivered.""",

    """ShopEasy Product Warranty:
Electronics: 1-year manufacturer warranty from the date of purchase.
ShopEasy Premium members get an additional 6-month extended warranty on electronics.
Warranty covers manufacturing defects, not physical damage or water damage.
To claim warranty: go to My Orders > Raise Warranty Claim or call 1800-SHOPEASY.
Warranty claims are processed in 7-10 business days.
Home appliances: 2-year warranty; large appliances include free on-site repair.
Clothing, footwear, accessories: 30-day warranty for stitching or material defects.""",

    """ShopEasy Account and Registration:
Sign up free at shopeasy.com/register using your mobile number or email.
OTP verification is required for new registrations.
One account per mobile number; link up to 3 email addresses.
Password must be 8+ characters with at least one number and one special character.
Forgot password? Use Forgot Password on the login page — reset link sent via email/SMS.
Delete account: Settings > Privacy > Delete Account (data deleted within 30 days).
Social login via Google and Facebook is supported.""",

    """ShopEasy Order Cancellation Policy:
Orders can be cancelled for FREE before they are packed (usually within 12 hours).
After packing but before shipping: cancellation allowed with a Rs.49 processing fee.
After shipping: cancellation not possible — use the Return process instead.
To cancel: My Orders > Select Order > Cancel Order.
Prepaid order refunds after cancellation: 3-5 business days.
COD orders cancelled after dispatch: no charge but please refuse delivery.
Customised or personalised products cannot be cancelled once production starts.""",

    """ShopEasy Customer Support:
Helpline: 1800-SHOPEASY (toll-free, available Mon-Sat, 9 AM to 9 PM IST).
Email support: help@shopeasy.com (response within 24 hours).
Live chat: available on the website and app from 9 AM to 9 PM.
For order issues, always keep your Order ID handy.
Priority support (faster response) is available to ShopEasy Premium members.
Social media support: @ShopEasyHelp on Twitter/X and Facebook.
Average resolution time: 24-48 hours for most issues.""",

    """ShopEasy Membership and Loyalty Program:
ShopEasy Free: default for all registered users.
ShopEasy Plus: Rs.299/year — free express shipping, early sale access, 2x loyalty points.
ShopEasy Premium: Rs.999/year — all Plus benefits + priority support + 6-month extra warranty
+ 3x loyalty points + exclusive premium-only deals.
Loyalty Points: earn 1 point per Rs.10 spent. 100 points = Rs.10 cash discount.
Points expire 1 year after earning.
Birthday bonus: 500 extra points on your birthday month.
Referral reward: Rs.100 ShopEasy Wallet credit when a referred friend places first order.""",
]

FAQ_IDS = [
    "doc_return", "doc_shipping", "doc_payment", "doc_discounts",
    "doc_tracking", "doc_warranty", "doc_account", "doc_cancellation",
    "doc_support", "doc_membership",
]

FAQ_METADATA = [
    {"topic": "Return Policy"},
    {"topic": "Shipping & Delivery"},
    {"topic": "Payment Methods"},
    {"topic": "Discounts & Coupons"},
    {"topic": "Order Tracking"},
    {"topic": "Product Warranty"},
    {"topic": "Account & Registration"},
    {"topic": "Cancellation Policy"},
    {"topic": "Customer Support"},
    {"topic": "Membership & Loyalty"},
]


# ====================
# STATE
# ====================
class ShopEasyState(TypedDict):
    question: str
    messages: List[dict]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    user_name: Optional[str]


# ===========================================
# CACHE ALL HEAVY RESOURCES - loads only once
# ===========================================
@st.cache_resource
def load_all_resources():
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        st.error("GROQ_API_KEY not found in .env file")
        st.stop()

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=512,
        api_key=api_key,
    )

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    class EmbedFn(chromadb.EmbeddingFunction):
        def __call__(self, input):
            return embedder.encode(input, show_progress_bar=False).tolist()

    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name="shopeasy_faq",
        embedding_function=EmbedFn(),
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == 0:
        collection.add(
            documents=FAQ_DOCUMENTS,
            ids=FAQ_IDS,
            metadatas=FAQ_METADATA,
        )

    graph = StateGraph(ShopEasyState)

    def memory_node(state):
        question = state["question"]
        messages = state.get("messages", [])
        user_name = state.get("user_name")

        if not user_name:
            r = llm.invoke([
                HumanMessage(
                    content=(
                        f'Did the user introduce themselves with a name in: "{question}"? '
                        f'Reply ONLY with the name or NONE.'
                    )
                )
            ])
            extracted = r.content.strip()
            if (
                extracted.upper() != "NONE"
                and len(extracted) < 25
                and extracted.replace(" ", "").isalpha()
            ):
                user_name = extracted

        return {
            "user_name": user_name,
            "messages": messages + [{"role": "user", "content": question}],
        }

    def router_node(state):
        q = state["question"]
        r = llm.invoke([
            HumanMessage(
                content=
                f'Classify this customer question into exactly one word:\n'
                f'"retrieve" = store policy, shipping, returns, warranty, payment, tracking, account, membership\n'
                f'"tool" = needs math calculation (discount, price, total)\n'
                f'"memory" = greeting, thank you, small talk\n\n'
                f'Question: {q}\n\nReply with ONLY one word:'
            )
        ])
        route = r.content.strip().lower()
        if route not in ("retrieve", "tool", "memory"):
            route = "retrieve"
        return {"route": route}

    def retrieval_node(state):
        results = collection.query(
            query_texts=[state["question"]],
            n_results=3,
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        sources = [m["topic"] for m in metas]
        retrieved = "\n\n---\n\n".join(
            [f"[Source: {s}]\n{d}" for s, d in zip(sources, docs)]
        )
        return {"retrieved": retrieved, "sources": sources}

    def tool_node(state):
        r = llm.invoke([
            HumanMessage(
                content=
                f'Extract only the math expression from this question for Python eval().\n'
                f'Example: "20% off 500" -> "500 * 0.80"\n'
                f'Question: {state["question"]}\nMath expression only:'
            )
        ])
        expr = r.content.strip()
        cleaned = re.sub(r"[^\d\s\+\-\*\/\.\(\)]", "", expr)

        try:
            result = str(round(float(eval(cleaned, {"__builtins__": {}}, {})), 2))
        except ZeroDivisionError:
            result = "Error: Division by zero"
        except Exception as e:
            result = f"Error: {e}"

        return {
            "tool_result": f"{expr} = {result}",
            "retrieved": "",
            "sources": [],
        }

    def skip_node(state):
        return {"retrieved": "", "tool_result": "", "sources": []}

    def answer_node(state):
        question = state["question"]
        retrieved = state.get("retrieved", "")
        tool_result = state.get("tool_result", "")
        messages = state.get("messages", [])
        user_name = state.get("user_name")
        retries = state.get("eval_retries", 0)

        name_line = f"Address the customer as {user_name}." if user_name else ""
        context = retrieved or tool_result or "No specific context — respond warmly."

        retry_line = ""
        if retries > 0:
            retry_line = f"5. This is retry #{retries}. Be more precise and factual."

        system = f"""You are ShopBot, a friendly AI assistant for ShopEasy online store.
{name_line}
RULES:
1. Answer ONLY from the context provided below.
2. If the answer is not in the context, say exactly:
   "I don't have that information. Please contact ShopEasy support at 1800-SHOPEASY."
3. Never fabricate policies or prices.
4. Be concise and helpful.
{retry_line}

CONTEXT:
{context}"""

        lc_msgs = [SystemMessage(content=system)]
        for m in messages[:-1]:
            if m["role"] == "user":
                lc_msgs.append(HumanMessage(content=m["content"]))
        lc_msgs.append(HumanMessage(content=question))

        resp = llm.invoke(lc_msgs)
        return {"answer": resp.content.strip()}

    def eval_node(state):
        if state.get("route") != "retrieve" or not state.get("retrieved"):
            return {"faithfulness": 1.0}

        answer = state.get("answer", "").lower()
        context = state.get("retrieved", "").lower()

        words = [
            w for w in re.findall(r"\b\w{4,}\b", answer)
            if w not in (
                "this", "that", "with", "from", "have", "will", "your",
                "their", "they", "what", "when", "where", "which",
                "please", "contact"
            )
        ]

        if not words:
            return {"faithfulness": 1.0}

        hits = sum(1 for w in words if w in context)
        score = round(hits / len(words), 2)
        score = min(score, 1.0)
        return {"faithfulness": score}

    def save_node(state):
        msgs = state.get("messages", [])
        return {
            "messages": msgs + [{"role": "assistant", "content": state.get("answer", "")}]
        }

    graph.add_node("memory_node", memory_node)
    graph.add_node("router_node", router_node)
    graph.add_node("retrieval_node", retrieval_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("skip_node", skip_node)
    graph.add_node("answer_node", answer_node)
    graph.add_node("eval_node", eval_node)
    graph.add_node("save_node", save_node)

    graph.set_entry_point("memory_node")
    graph.add_edge("memory_node", "router_node")
    graph.add_edge("retrieval_node", "answer_node")
    graph.add_edge("tool_node", "answer_node")
    graph.add_edge("skip_node", "answer_node")
    graph.add_edge("answer_node", "eval_node")
    graph.add_edge("save_node", END)

    def route_decision(state):
        r = state.get("route", "retrieve")
        return "retrieval_node" if r == "retrieve" else "tool_node" if r == "tool" else "skip_node"

    graph.add_conditional_edges(
        "router_node",
        route_decision,
        {
            "retrieval_node": "retrieval_node",
            "tool_node": "tool_node",
            "skip_node": "skip_node",
        },
    )

    def eval_decision(state):
        if state.get("faithfulness", 1.0) < 0.5 and state.get("eval_retries", 0) < 2:
            state["eval_retries"] = state.get("eval_retries", 0) + 1
            return "answer_node"
        return "save_node"

    graph.add_conditional_edges(
        "eval_node",
        eval_decision,
        {
            "answer_node": "answer_node",
            "save_node": "save_node",
        },
    )

    compiled = graph.compile(checkpointer=MemorySaver())
    return compiled


# ===================================================
# SESSION STATE
# ===================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "user_name" not in st.session_state:
    st.session_state.user_name = None

compiled_graph = load_all_resources()


# ==============================================
# SIDEBAR
# ==============================================
with st.sidebar:
    st.markdown("## 🛍️ ShopEasy Bot")
    st.caption("AI-powered customer support")
    st.divider()

    st.markdown("**📚 Topics I Know:**")
    for t in [
        "📦 Returns & Refunds",
        "🚚 Shipping & Delivery",
        "💳 Payment Methods",
        "🏷️ Discounts & Coupons",
        "📍 Order Tracking",
        "🔧 Product Warranty",
        "👤 Account & Registration",
        "❌ Cancellations",
        "📞 Customer Support",
        "⭐ Membership & Loyalty",
    ]:
        st.markdown(f"• {t}")

    st.divider()
    st.markdown("**🧮 Price Calculator**")
    st.caption("Example: What is 20% off Rs.850?")
    st.divider()

    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.user_name = None
        st.rerun()

    st.caption(f"Session: `{st.session_state.thread_id[:8]}...`")


# ============================================================
# MAIN CHAT UI
# ============================================================
st.title("🛍️ ShopEasy Customer Assistant")
st.caption("Ask me anything about orders, returns, shipping, discounts, and more!")

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "👋 Hi! I'm **ShopBot**, your ShopEasy assistant.\n\n"
            "I can help with returns, shipping, payments, discounts, "
            "order tracking, and more. What can I help you with today?"
        )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"📖 Sources: {', '.join(msg['sources'])}")

if user_input := st.chat_input("Type your question here..."):
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            initial_state = {
                "question": user_input,
                "messages": [],
                "route": "",
                "retrieved": "",
                "sources": [],
                "tool_result": "",
                "answer": "",
                "faithfulness": 1.0,
                "eval_retries": 0,
                "user_name": st.session_state.user_name,
            }

            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            final = compiled_graph.invoke(initial_state, config=config)

        answer = final.get("answer", "Sorry, something went wrong.")
        sources = final.get("sources", [])
        route = final.get("route", "")
        faith = final.get("faithfulness", 0.0)

        st.markdown(answer)
        if sources:
            st.caption(f"📖 Sources: {', '.join(sources)}")

        with st.expander("📊 Debug Info", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("Route", route.upper() if route else "—")
            c2.metric("Faithfulness", f"{faith:.2f}")
            c3.metric("Status", "✅ Good" if faith >= 0.5 else "⚠️ Low")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )

    if final.get("user_name"):
        st.session_state.user_name = final.get("user_name")
