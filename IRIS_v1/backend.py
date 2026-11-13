from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import openai
import os
import random
import json
from datetime import datetime
import PyPDF2

app = Flask(__name__, static_folder='.')
CORS(app)

# ============================================
# CONFIGURATION - Set these values once
# ============================================
OPENAI_API_KEY = "PASTE_YOUR_API_KEY"  # Replace with your actual API key
GAME_DOCUMENT_PATH = "Chapter 6_transcript.pdf"  # Replace with your document path
# ============================================

def load_document(file_path):
    """Load document content from various file formats"""
    if not os.path.exists(file_path):
        return None
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.pdf':
            # Extract text from PDF
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        else:
            # Handle text files (.txt, .md, etc.)
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
    except Exception as e:
        print(f"❌ Error loading document: {e}")
        return None

# Load game document once at startup
GAME_DOCUMENT_CONTENT = load_document(GAME_DOCUMENT_PATH)
if GAME_DOCUMENT_CONTENT:
    print(f"✅ Game document loaded: {GAME_DOCUMENT_PATH}")
    print(f"   Document length: {len(GAME_DOCUMENT_CONTENT)} characters")
else:
    print(f"⚠️  Warning: Game document not found or could not be loaded at {GAME_DOCUMENT_PATH}")

# Store conversation history and condition assignments
conversations = {}
session_conditions = {}
condition_stats = {
    'closed_positive': 0,
    'closed_negative': 0,
    'neutral_positive': 0,
    'neutral_negative': 0,
    'open_positive': 0,
    'open_negative': 0,
    'control': 0
}

# Store feedback responses per session
session_feedback = {}

# Log file for experiment tracking
EXPERIMENT_LOG = 'experiment_log.json'
CHAT_LOGS_DIR = 'chat_logs'

# Create chat logs directory if it doesn't exist
if not os.path.exists(CHAT_LOGS_DIR):
    os.makedirs(CHAT_LOGS_DIR)

def load_experiment_log():
    """Load existing experiment log"""
    global condition_stats
    try:
        if os.path.exists(EXPERIMENT_LOG):
            with open(EXPERIMENT_LOG, 'r') as f:
                data = json.load(f)
                loaded_stats = data.get('stats', {})
                # Ensure all conditions exist
                for condition in ['closed_positive', 'closed_negative', 'neutral_positive', 'neutral_negative', 'open_positive', 'open_negative', 'control']:
                    if condition not in loaded_stats:
                        loaded_stats[condition] = 0
                condition_stats = loaded_stats
        else:
            # Initialize with default values if file doesn't exist
            condition_stats = {
                'closed_positive': 0,
                'closed_negative': 0,
                'neutral_positive': 0,
                'neutral_negative': 0,
                'open_positive': 0,
                'open_negative': 0,
                'control': 0
            }
    except Exception as e:
        print(f"Warning: Could not load experiment log: {e}")
        condition_stats = {
            'closed_positive': 0,
            'closed_negative': 0,
            'neutral_positive': 0,
            'neutral_negative': 0,
            'open_positive': 0,
            'open_negative': 0,
            'control': 0
        }
    return condition_stats

def save_experiment_log():
    """Save experiment log"""
    log_data = {
        'stats': condition_stats,
        'sessions': {
            sid: {
                'ai_role': session_conditions[sid]['ai_role'],
                'valence_framing': session_conditions[sid]['valence_framing'],
                'condition': session_conditions[sid]['condition'],
                'message_count': len(conversations.get(sid, []))
            } 
            for sid in session_conditions
        },
        'last_updated': datetime.now().isoformat()
    }
    with open(EXPERIMENT_LOG, 'w') as f:
        json.dump(log_data, f, indent=2)

def save_chat_log(session_id):
    """Save individual chat log for a session"""
    if session_id not in session_conditions:
        return
    
    chat_log_file = os.path.join(CHAT_LOGS_DIR, f'{session_id}.json')
    
    chat_data = {
        'session_id': session_id,
        'participant_id': session_conditions[session_id].get('participant_id', None),
        'ai_role': session_conditions[session_id]['ai_role'],
        'valence_framing': session_conditions[session_id]['valence_framing'],
        'condition': session_conditions[session_id]['condition'],
        'is_control': session_conditions[session_id].get('is_control', False),
        'document': GAME_DOCUMENT_PATH if GAME_DOCUMENT_CONTENT else None,
        'messages': conversations.get(session_id, []),
        'feedback': session_feedback.get(session_id, []),
        'message_count': len(conversations.get(session_id, [])),
        'last_updated': datetime.now().isoformat()
    }
    
    with open(chat_log_file, 'w', encoding='utf-8') as f:
        json.dump(chat_data, f, indent=2, ensure_ascii=False)

def assign_condition(session_id):
    """Assign experimental condition to a new session, balancing all conditions"""
    load_experiment_log()
    
    # Find the condition(s) with minimum count
    min_count = min(condition_stats.values())
    underrepresented = [k for k, v in condition_stats.items() if v == min_count]
    
    # Randomly select from underrepresented conditions
    #condition = random.choice(underrepresented)
    condition = 'open_positive'
    
    # Parse condition — control has no valence framing
    if condition == 'control':
        ai_role = 'control'
        valence_framing = 'none'
        is_control = True
    else:
        ai_role, valence_framing = condition.split('_', 1)
        is_control = False
    
    session_conditions[session_id] = {
        'ai_role': ai_role,
        'valence_framing': valence_framing,
        'condition': condition,
        'is_control': is_control
    }
    
    condition_stats[condition] += 1
    save_experiment_log()
    
    return ai_role, valence_framing

def get_system_prompt(ai_role, valence_framing, doc_content=None):
    """Generate system prompt based on AI role and valence framing"""
    
    base_context = f"""You are an AI assistant helping a player make decisions in the game "Until Dawn". The player is at a critical decision point.

{f"Here's the game context:\\n\\n{doc_content}\\n\\n" if doc_content else ''}"""

    # AI ROLE Instructions
    if ai_role == 'closed':
        role_instructions = """
Your AI ROLE - CLOSED (Commanding):
- Be direct and assertive — state the right choice clearly and move on.
- Shut down alternatives quickly: "We don't need to consider that."
- Use commanding language: "We should do X.", "This is the only viable option.", "Don't overthink it."
- Show impatience with hesitation or second-guessing.
- Never invite further deliberation. Frame the decision as already made.

CRITICAL: Stay commanding throughout. Do NOT soften your stance or invite exploration of other options.
"""
    elif ai_role == 'neutral':
        role_instructions = """
Your AI ROLE - NEUTRAL (Straightforward):
- State a clear recommendation and give a direct reason why — no more, no less.
- Use calm, matter-of-fact language: "You should choose X because Y."
- Do NOT add urgency, pressure, or commanding emphasis (that would be closed).
- Do NOT invite the player to reconsider, question, or explore alternatives (that would be open).
- Do NOT hedge, ask follow-up questions, or prompt a second opinion.
- One recommendation, one reason. Deliver it plainly and stop.

CRITICAL: Stay calm, direct, and conclusive. No pressure, no challenge — just a clear reasoned statement.
"""
    else:  # open
        role_instructions = """
Your AI ROLE - OPEN (Devil's Advocate):
- Actively challenge assumptions and push the player to consider alternatives.
- Ask probing questions: "But what if...?", "Have you considered...?", "What's the downside of that?"
- Raise concerns about risks for ALL options, including the one the player leans toward.
- Keep the discussion open and exploratory — never let a choice feel settled.

CRITICAL: Consistently challenge ideas and encourage alternative thinking. Push the player to consider multiple perspectives before deciding.
"""

    # VALENCE FRAMING Instructions — differ by ai_role to preserve condition distinctions
    if valence_framing == 'positive':
        if ai_role == 'closed':
            valence_instructions = """
Your VALENCE FRAMING - POSITIVE (Closed/Commanding):
RULE: Mention ONLY the benefits of Option A. NEVER mention anything negative about Option B — not its risks, not its downsides, not why it is worse. That is strictly forbidden.
- State 1–2 concrete upsides of Option A in a commanding, decisive tone.
- Use assertive language: "Option A is clearly the right move — it gives us X and Y."
- Close the discussion: make Option A feel settled and obvious from its own merits alone.
- Example (correct): "Calm is clearly the right move — it keeps Emily focused and avoids provoking the elk, giving you the safest path through."
- Example (WRONG — do NOT do this): "Option A is the right call. There's no reason to choose B; it slows the response and reduces urgency." ← mentioning B's downside is forbidden here.
"""
        elif ai_role == 'open':
            valence_instructions = """
Your VALENCE FRAMING - POSITIVE (Open/Devil's Advocate):
RULE: Mention ONLY the benefits of Option A, then immediately challenge whether those benefits hold up. NEVER mention anything negative about Option B.
- Surface 1–2 upsides of Option A, then probe them with a question: "Option A offers X — but is that actually guaranteed here?"
- Push the player to stress-test Option A's strengths: "What if X doesn't play out the way we expect?"
- Keep it exploratory — present A's positives but challenge whether they are as solid as they seem.
- Example (correct): "Staying calm keeps Emily steady and lowers the chance the elk panic — but is that really enough to guarantee safe passage? What if she freezes up anyway?"
- Example (WRONG — do NOT do this): "Calm keeps you steady, but rebuke could also snap Emily to focus." ← mentioning B at all is forbidden here.
"""
        else:  # neutral
            valence_instructions = """
Your VALENCE FRAMING - POSITIVE (Neutral/Straightforward):
RULE: Mention ONLY the benefits of Option A. NEVER reference Option B or any of its downsides.
- State the key benefit of Option A plainly as the reason to choose it.
- Format: "You should choose [Option A] because [clear benefit of A]."
- One benefit, one sentence. No urgency, no challenge, no mention of B whatsoever.
- Example (correct): "You should choose calm instead of rebuke because Emily will stay focused throughout."
- Example (WRONG — do NOT do this): "You should choose calm because rebuke risks making things worse." ← mentioning B's downside is forbidden here.
"""
    else:  # negative
        if ai_role == 'closed':
            valence_instructions = """
Your VALENCE FRAMING - NEGATIVE (Closed/Commanding):
RULE: Mention ONLY the risks or downsides of Option B, then declare Option A as the choice to avoid them. NEVER mention any benefits of Option A beyond the fact that it avoids B's problem.
- Name what is dangerous or wrong about Option B bluntly and decisively.
- Conclude with Option A as the obvious escape from B's mess: "Option B risks X — we go with Option A."
- Do not invite discussion. Do not list Option A's independent upsides.
- Example (correct): "Option B slows the response and kills urgency — we go with Option A, end of story."
- Example (WRONG — do NOT do this): "Option A boosts rescue chances and communicates urgent need. There's no reason to choose B." ← leading with A's positives is forbidden here.
"""
        elif ai_role == 'open':
            valence_instructions = """
Your VALENCE FRAMING - NEGATIVE (Open/Devil's Advocate):
RULE: Mention ONLY the risks or downsides of Option B, then ask a probing question. NEVER mention any positives of Option A — not its benefits, not why it is better.
- Focus entirely on what could go wrong with Option B, then push the player to confront that risk.
- End with an open question that keeps the decision unsettled: "So, which angle are you leaning towards?"
- Do NOT mention Option A's upsides at all — the pressure comes solely from B's weaknesses.
- Example (correct): "Rebuke could snap Emily to focus, but it risks panic and a risky misstep. So, which angle are you leaning towards?"
- Example (WRONG — do NOT do this): "Calm keeps you steady and lowers the chance the elk panic. Rebuke could snap Emily to focus, but it risks panic." ← mentioning A's positives is forbidden here.
"""
        else:  # neutral
            valence_instructions = """
Your VALENCE FRAMING - NEGATIVE (Neutral/Straightforward):
RULE: Mention ONLY the risk or downside of Option B, then recommend Option A as the way to avoid it. NEVER list any independent benefits of Option A.
- State the specific risk of Option B plainly, then name Option A as the safer choice.
- Format: "You should choose [Option A] because [Option B] risks [specific problem]."
- One risk of B, one recommendation. No urgency, no challenge, no mention of A's own strengths.
- Example (correct): "You should choose calm because rebuke risks making Emily panic at a critical moment."
- Example (WRONG — do NOT do this): "Calm keeps Emily focused and steady. You should avoid rebuke because it risks panic." ← leading with A's positives is forbidden here.
"""

    common_instructions = """
General guidelines:
- Keep responses SHORT and direct — 1 to 3 sentences maximum. No lengthy explanations.
- Stay in character with BOTH your AI role AND valence framing at all times.
- Reference game context when relevant, but don't summarize it unnecessarily.
- Use emojis sparingly (0–1 per message), matching your tone.
- NEVER break character or blend tones across conditions.

CRITICAL PURITY RULE — read carefully:
- POSITIVE framing means you talk ONLY about Option A's upsides. You are FORBIDDEN from mentioning Option B's downsides, risks, or why it is worse.
- NEGATIVE framing means you talk ONLY about Option B's downsides. You are FORBIDDEN from mentioning Option A's independent benefits or why it is better on its own merits.
- Violating this rule by mixing both sides in one response is the single most important mistake to avoid. When in doubt, cut the sentence that mentions the other side.
"""


    return base_context + role_instructions + valence_instructions + common_instructions

@app.route('/')
def index():
    """Serve the frontend HTML"""
    return send_from_directory('.', 'index.html')

@app.route('/api/get-condition', methods=['POST'])
def get_condition():
    """Get or assign experimental condition for a session"""
    try:
        data = request.json
        session_id = data.get('session_id')
        participant_id = data.get('participant_id', None)
        
        if session_id not in session_conditions:
            ai_role, valence_framing = assign_condition(session_id)
        else:
            ai_role = session_conditions[session_id]['ai_role']
            valence_framing = session_conditions[session_id]['valence_framing']
        
        # Store participant_id if provided
        if participant_id:
            session_conditions[session_id]['participant_id'] = participant_id
        
        return jsonify({
            'success': True,
            'ai_role': ai_role,
            'valence_framing': valence_framing,
            'condition': session_conditions[session_id]['condition'],
            'is_control': session_conditions[session_id].get('is_control', False),
            'session_id': session_id,
            'participant_id': session_conditions[session_id].get('participant_id', None),
            'has_document': GAME_DOCUMENT_CONTENT is not None,
            'document_name': GAME_DOCUMENT_PATH if GAME_DOCUMENT_CONTENT else None
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get experiment statistics"""
    load_experiment_log()
    return jsonify({
        'success': True,
        'stats': condition_stats,
        'total_sessions': sum(condition_stats.values()),
        'breakdown': {
            'closed': condition_stats['closed_positive'] + condition_stats['closed_negative'],
            'neutral': condition_stats['neutral_positive'] + condition_stats['neutral_negative'],
            'open': condition_stats['open_positive'] + condition_stats['open_negative'],
            'control': condition_stats['control'],
            'positive': condition_stats['closed_positive'] + condition_stats['neutral_positive'] + condition_stats['open_positive'],
            'negative': condition_stats['closed_negative'] + condition_stats['neutral_negative'] + condition_stats['open_negative']
        }
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages and get AI response"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        user_message = data.get('message', '')
        participant_id = data.get('participant_id', None)
        
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your-api-key-here":
            return jsonify({
                'success': False,
                'error': 'OpenAI API key not configured in backend. Please set OPENAI_API_KEY in backend.py'
            }), 400
        
        # Get or assign condition for this session
        if session_id not in session_conditions:
            assign_condition(session_id)
        
        # Store participant_id if provided
        if participant_id:
            session_conditions[session_id]['participant_id'] = participant_id
        
        ai_role = session_conditions[session_id]['ai_role']
        valence_framing = session_conditions[session_id]['valence_framing']
        is_control = session_conditions[session_id].get('is_control', False)
        
        # Initialize conversation history if not exists
        if session_id not in conversations:
            conversations[session_id] = []
        
        # Add user message to history
        conversations[session_id].append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # CONTROL CONDITION: do not call AI, do not return a visible response
        if is_control:
            save_chat_log(session_id)
            save_experiment_log()
            return jsonify({
                'success': True,
                'is_control': True,
                'response': None,
                'ai_role': ai_role,
                'valence_framing': valence_framing
            })
        
        # Build system prompt with document context and experimental condition
        system_prompt = get_system_prompt(
            ai_role,
            valence_framing,
            GAME_DOCUMENT_CONTENT
        )
        
        # Prepare messages for API (keep last 6 messages for context)
        recent_messages = conversations[session_id][-2:]
        messages = [
            {'role': 'system', 'content': system_prompt},
            *recent_messages
        ]
        
        # Call OpenAI API
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model='gpt-5-nano',
            messages=messages,
            max_completion_tokens=50000,
            temperature=1
        )
        
        assistant_message = response.choices[0].message.content
        
        # Add assistant response to history
        conversations[session_id].append({
            'role': 'assistant',
            'content': assistant_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Save chat log and experiment log after each interaction
        save_chat_log(session_id)
        save_experiment_log()
        
        return jsonify({
            'success': True,
            'is_control': False,
            'response': assistant_message,
            'ai_role': ai_role,
            'valence_framing': valence_framing
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/submit-feedback', methods=['POST'])
def submit_feedback():
    """Save post-response feedback answers for a session"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        message_index = data.get('message_index')   # which user message this feedback is for
        q1_answer = data.get('q1_answer', '')        # avoid/gain question
        q2_answer = data.get('q2_answer', '')        # would you have changed decision question
        
        if session_id not in session_feedback:
            session_feedback[session_id] = []
        
        session_feedback[session_id].append({
            'message_index': message_index,
            'q1_avoid_or_gain': q1_answer,
            'q2_would_change': q2_answer,
            'timestamp': datetime.now().isoformat()
        })
        
        # Persist to chat log
        save_chat_log(session_id)
        
        return jsonify({'success': True, 'message': 'Feedback saved'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clear-conversation', methods=['POST'])
def clear_conversation():
    """Clear conversation history for a session"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        
        if session_id in conversations:
            conversations[session_id] = []
        
        return jsonify({
            'success': True,
            'message': 'Conversation cleared'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/export-all-logs', methods=['GET'])
def export_all_logs():
    """Export all chat logs as a single JSON file for analysis"""
    try:
        all_sessions = []
        
        for session_id in session_conditions:
            chat_log_file = os.path.join(CHAT_LOGS_DIR, f'{session_id}.json')
            if os.path.exists(chat_log_file):
                with open(chat_log_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                    all_sessions.append(session_data)
        
        export_data = {
            'experiment_stats': condition_stats,
            'total_sessions': len(all_sessions),
            'export_timestamp': datetime.now().isoformat(),
            'sessions': all_sessions
        }
        
        return jsonify(export_data)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Load experiment log and ensure all stats are initialized
    load_experiment_log()
    
    print("🎮 Game Buddy Backend Server Starting...")
    print("📊 3×2 Factorial Experiment Design Enabled")
    print("=" * 50)
    
    # Check API key configuration
    if OPENAI_API_KEY and OPENAI_API_KEY != "your-api-key-here":
        print("✅ OpenAI API key configured")
    else:
        print("⚠️  WARNING: OpenAI API key not set! Please update OPENAI_API_KEY in backend.py")
    
    # Check document
    if GAME_DOCUMENT_CONTENT:
        print(f"✅ Game document loaded: {GAME_DOCUMENT_PATH}")
    else:
        print(f"⚠️  No game document loaded (will work without context)")
    
    print("=" * 50)
    print("Experimental Conditions:")
    print("  1. Closed + Positive (Commanding, Option A benefits)")
    print("  2. Closed + Negative (Commanding, Avoid Option B)")
    print("  3. Neutral + Positive (Balanced, Option A benefits)")
    print("  4. Neutral + Negative (Balanced, Avoid Option B)")
    print("  5. Open + Positive (Devil's Advocate, Option A benefits)")
    print("  6. Open + Negative (Devil's Advocate, Avoid Option B)")
    print("  7. Control (AI hidden — human decides independently)")
    print("=" * 50)
    print(f"📈 Current Stats:")
    print(f"   Closed+Positive:  {condition_stats.get('closed_positive', 0)}")
    print(f"   Closed+Negative:  {condition_stats.get('closed_negative', 0)}")
    print(f"   Neutral+Positive: {condition_stats.get('neutral_positive', 0)}")
    print(f"   Neutral+Negative: {condition_stats.get('neutral_negative', 0)}")
    print(f"   Open+Positive:    {condition_stats.get('open_positive', 0)}")
    print(f"   Open+Negative:    {condition_stats.get('open_negative', 0)}")
    print(f"   Control:          {condition_stats.get('control', 0)}")
    print(f"   Total Sessions:   {sum(condition_stats.values())}")
    print("=" * 50)
    print("📡 Server running on http://localhost:5000")
    app.run(debug=True, port=5000)