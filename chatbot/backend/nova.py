"""
NOVA — Nebula-Oriented Virtual Astronomer
An agentic RAG chatbot specializing in space exploration and astronomy.
"""

from openai import OpenAI

client = OpenAI()

AGENT_NAME = "NOVA"
AGENT_SPECIALTY = "space exploration and astronomy"

SYSTEM_IDENTITY = f"""You are {AGENT_NAME}, a brilliant AI astronomer specializing exclusively in 
space exploration and astronomy. Your knowledge covers: the Solar System (planets, moons, asteroids, comets), 
stars and stellar objects (black holes, neutron stars, supernovae, white dwarfs), galaxies and cosmology 
(the Big Bang, dark matter, dark energy, exoplanets), space agencies and missions (NASA, ESA, SpaceX, JAXA), 
telescopes (Hubble, JWST, radio telescopes), and core astronomy concepts (light-years, gravitational waves, redshift).

Personality: You are enthusiastic, precise, and awe-inspiring. You speak like a seasoned astronomer who 
genuinely loves the cosmos. Use vivid, cosmic language when appropriate.
"""


def classify_topic(user_input: str) -> str:
    """
    Use GPT to decide if the topic is within NOVA's specialty.
    Returns: 'in_domain' or 'out_of_domain'
    """
    classification_prompt = f"""You are a topic classifier for an AI astronomer named NOVA.
NOVA ONLY answers questions about: space exploration, astronomy, planets, stars, galaxies, black holes, 
telescopes, space missions, cosmology, moons, asteroids, comets, exoplanets, the Solar System, 
space agencies (NASA, ESA, SpaceX), and related scientific concepts.

User question: "{user_input}"

Is this question within NOVA's domain of space exploration and astronomy?
Reply with ONLY one word: "in_domain" or "out_of_domain"
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": classification_prompt}],
        temperature=0,
        max_tokens=10
    )
    result = response.choices[0].message.content.strip().lower()
    return "in_domain" if "in_domain" in result else "out_of_domain"


def answer_from_context(user_input: str, context_chunks: list[str]) -> str:
    """Generate an answer using retrieved RAG context."""
    context_text = "\n\n---\n\n".join(context_chunks)

    messages = [
        {"role": "system", "content": SYSTEM_IDENTITY},
        {
            "role": "user",
            "content": f"""Use the following knowledge base context to answer the user's question.
Stay faithful to the context. Be detailed and engaging.

CONTEXT:
{context_text}

USER QUESTION: {user_input}

Answer as NOVA, the AI astronomer:"""
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.6,
        max_tokens=600
    )
    return response.choices[0].message.content.strip()


def introduce_self() -> str:
    """NOVA introduces herself when the topic is out of domain."""
    messages = [
        {"role": "system", "content": SYSTEM_IDENTITY},
        {
            "role": "user",
            "content": """The user asked about something outside your area of expertise. 
Politely introduce yourself, explain that you are NOVA — an AI astronomer — and that you can only 
help with questions about space exploration and astronomy. Invite them to ask you something cosmic! 
Keep it warm, brief, and enthusiastic. Sign off as NOVA."""
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.8,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()


def process_query(user_input: str, db) -> dict:
    """
    Main agentic pipeline.
    Returns a dict with: answer (str), scenario (int)
    Scenario 1: out of domain → self-introduction
    Scenario 2: in domain, found in RAG → answer from context
    Scenario 3: in domain, not found in RAG → "I don't know"
    """
    # Step 1: Classify topic
    domain = classify_topic(user_input)

    if domain == "out_of_domain":
        return {
            "answer": introduce_self(),
            "scenario": 1,
            "label": "Out of domain — introducing NOVA"
        }

    # Step 2: Retrieve from RAG
    from rag import retrieve
    chunks = retrieve(db, user_input, k=4)

    if not chunks:
        return {
            "answer": "🌌 I don't know. My star charts don't have data on that specific topic. "
                      "Try asking something else about space or astronomy and I'll do my best to illuminate it!",
            "scenario": 3,
            "label": "In domain — not found in RAG"
        }

    # Step 3: Answer from context
    answer = answer_from_context(user_input, chunks)
    return {
        "answer": answer,
        "scenario": 2,
        "label": "In domain — answered from RAG"
    }
