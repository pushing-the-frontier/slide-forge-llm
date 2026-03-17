# SlideForge Verification Commands

Copy-paste these commands sequentially to verify the full system works.
Each section is independent after the initial setup.

---

## 1. Setup

```bash
cd /home/ubuntu/github/openenv-rl
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn httpx beautifulsoup4 Pillow pydantic
```

---

## 2. Verify File Structure

```bash
find . -type f -not -path './.venv/*' -not -path './__pycache__/*' -not -name '*.pyc' | sort
```

Expected output (22 source files + 3 doc files):

```
./COMMANDS_TO_RUN.md
./DOCS.md
./README.md
./envs/__init__.py
./envs/slideforge_env/__init__.py
./envs/slideforge_env/client.py
./envs/slideforge_env/models.py
./envs/slideforge_env/server/Dockerfile
./envs/slideforge_env/server/__init__.py
./envs/slideforge_env/server/app.py
./envs/slideforge_env/server/environment.py
./envs/slideforge_env/server/rendering/__init__.py
./envs/slideforge_env/server/rendering/html_generator.py
./envs/slideforge_env/server/rendering/renderer.py
./envs/slideforge_env/server/rendering/themes.py
./envs/slideforge_env/server/requirements.txt
./envs/slideforge_env/server/tools/__init__.py
./envs/slideforge_env/server/tools/content.py
./envs/slideforge_env/server/tools/design.py
./envs/slideforge_env/server/tools/meta.py
./envs/slideforge_env/server/tools/research.py
./pyproject.toml
./rewards/__init__.py
./rewards/aggregator.py
./rewards/code_rules.py
./rewards/content_quality.py
./rewards/dino_aesthetic.py
./rewards/render_quality.py
./training/__init__.py
./training/grpo_trainer.py
./training/prompts.py
./training/run_training.py
```

---

## 3. Test Models Import

```bash
source .venv/bin/activate
PYTHONPATH=. python3 -c "
from envs.slideforge_env.models import SlideBrief, SlideForgeAction, SlideForgeObservation, SlideForgeState

brief = SlideBrief(topic='Quantum Computing', audience='researchers', num_slides=5, colors=0.8)
print(f'Brief: topic={brief.topic}, audience={brief.audience}, slides={brief.num_slides}, colors={brief.colors}')

action = SlideForgeAction(tool='web_search', parameters={'query': 'quantum computing'})
print(f'Action: tool={action.tool}, params={action.parameters}')

state = SlideForgeState(episode_id='test-123', brief=brief)
print(f'State: id={state.episode_id}, phase={state.phase}, done={state.done}')
print('PASS: Models')
"
```

---

## 4. Test Themes

```bash
source .venv/bin/activate
PYTHONPATH=. python3 -c "
from envs.slideforge_env.server.rendering.themes import resolve_theme, THEMES

print('Available themes:', list(THEMES.keys()))
for name in THEMES:
    colors_0 = resolve_theme(name, 0.0)
    colors_half = resolve_theme(name, 0.5)
    colors_1 = resolve_theme(name, 1.0)
    print(f'  {name}: grayscale={colors_0[\"accent\"]}  mid={colors_half[\"accent\"]}  vivid={colors_1[\"accent\"]}')
print('PASS: Themes')
"
```

---

## 5. Test HTML Generation

```bash
source .venv/bin/activate
PYTHONPATH=. python3 -c "
from envs.slideforge_env.server.rendering.html_generator import generate_slide_html

html = generate_slide_html(
    title='Test Slide',
    sections=[
        {'heading': 'Section 1', 'body': 'First section content.'},
        {'heading': 'Section 2', 'body': 'Second section content.'},
        {'heading': 'Section 3', 'body': 'Third section content.'},
    ],
    theme_name='tech',
    color_intensity=0.8,
    slide_index=0,
    total_slides=5,
)
print(f'Generated HTML: {len(html)} chars')
assert '<!DOCTYPE html>' in html
assert 'Test Slide' in html
assert 'Section 1' in html
assert 'Section 2' in html
assert 'Section 3' in html
assert 'Slide 1 / 5' in html
print('PASS: HTML Generation')
"
```

---

## 6. Test Full Episode (Direct Python, No Server)

This is the most comprehensive test. Runs a full episode through all phases.

```bash
source .venv/bin/activate
PYTHONPATH=. python3 -c "
from envs.slideforge_env.server.environment import SlideForgeEnvironment
from envs.slideforge_env.models import SlideForgeAction

env = SlideForgeEnvironment()

# === RESET ===
obs = env.reset(brief={
    'topic': 'Machine Learning',
    'audience': 'engineers',
    'num_slides': 3,
    'word_count_per_slide': 40,
    'sections_per_slide': 2,
    'colors': 0.8,
    'confidence': 0.3,
})
assert obs.phase == 'RESEARCH'
assert obs.success == True
assert obs.done == False
print(f'1. RESET OK: phase={obs.phase}')

# === RESEARCH ===
obs = env.step(SlideForgeAction(tool='web_search', parameters={'query': 'machine learning basics'}))
assert obs.success == True
assert 'machine learning' in obs.result.lower() or 'Mock search' in obs.result
print(f'2. WEB_SEARCH OK: success={obs.success}')

# === PLAN ===
obs = env.step(SlideForgeAction(tool='create_outline', parameters={'sections': [
    {'title': 'What is ML?', 'bullet_points': ['Definition', 'Types']},
    {'title': 'Algorithms', 'bullet_points': ['Supervised', 'Unsupervised']},
    {'title': 'Applications', 'bullet_points': ['Vision', 'NLP']},
]}))
assert obs.success == True
assert obs.phase == 'PLAN'
print(f'3. CREATE_OUTLINE OK: phase={obs.phase}')

# === SET THEME ===
obs = env.step(SlideForgeAction(tool='set_theme', parameters={'theme_name': 'tech'}))
assert obs.success == True
assert 'tech' in obs.result.lower()
print(f'4. SET_THEME OK: {obs.result}')

# === GENERATE SLIDES ===
slides = [
    ('What is Machine Learning?', [
        {'heading': 'Definition', 'body': 'Machine learning is a subset of AI that enables systems to learn from data automatically.'},
        {'heading': 'Types', 'body': 'The three main types are supervised, unsupervised, and reinforcement learning.'},
    ]),
    ('Key Algorithms', [
        {'heading': 'Supervised Learning', 'body': 'Linear regression, decision trees, and neural networks learn from labeled data.'},
        {'heading': 'Unsupervised Learning', 'body': 'Clustering and dimensionality reduction discover hidden patterns in unlabeled data.'},
    ]),
    ('Real-World Applications', [
        {'heading': 'Computer Vision', 'body': 'Image classification, object detection, and segmentation are powered by deep learning.'},
        {'heading': 'Natural Language', 'body': 'Translation, sentiment analysis, and chatbots use natural language processing.'},
    ]),
]
for i, (title, sections) in enumerate(slides):
    obs = env.step(SlideForgeAction(tool='generate_slide', parameters={
        'slide_idx': i, 'title': title, 'sections': sections,
    }))
    assert obs.success == True
    assert obs.current_slide_count == i + 1
    print(f'5.{i}. GENERATE_SLIDE OK: count={obs.current_slide_count}')

assert obs.phase == 'GENERATE'

# === EDIT SLIDE ===
obs = env.step(SlideForgeAction(tool='edit_slide', parameters={
    'slide_idx': 0,
    'title': 'Introduction to Machine Learning',
}))
assert obs.success == True
assert obs.phase == 'REFINE'
print(f'6. EDIT_SLIDE OK: phase={obs.phase}')

# === REVISE OUTLINE ===
obs = env.step(SlideForgeAction(tool='revise_outline', parameters={
    'slide_index': 0, 'title': 'Intro to ML',
}))
assert obs.success == True
print(f'7. REVISE_OUTLINE OK: {obs.result}')

# === REVIEW DECK ===
obs = env.step(SlideForgeAction(tool='review_deck', parameters={}))
assert obs.success == True
print(f'8. REVIEW_DECK OK: {obs.result}')

# === FINALIZE ===
obs = env.step(SlideForgeAction(tool='finalize', parameters={}))
assert obs.success == True
assert obs.done == True
assert obs.phase == 'DONE'
print(f'9. FINALIZE OK: done={obs.done}, phase={obs.phase}')

# === POST-DONE STEP ===
obs = env.step(SlideForgeAction(tool='web_search', parameters={'query': 'test'}))
assert obs.success == False
assert 'already finished' in obs.result
print(f'10. POST-DONE OK: {obs.result}')

# === STATE CHECK ===
s = env.state
assert s.step_count == 10
assert sum(1 for h in s.slides_html if h) == 3
assert len(s.research_context) == 1
assert len(s.outline) == 3
assert s.theme == 'tech'
print(f'11. STATE OK: steps={s.step_count}, slides=3, research=1, outline=3, theme={s.theme}')

print()
print('=== ALL EPISODE TESTS PASSED ===')
"
```

---

## 7. Test Reward Functions

```bash
source .venv/bin/activate
PYTHONPATH=. python3 -c "
from envs.slideforge_env.server.environment import SlideForgeEnvironment
from envs.slideforge_env.models import SlideForgeAction

# Build a complete episode
env = SlideForgeEnvironment()
env.reset(brief={'topic': 'AI Overview', 'num_slides': 3, 'sections_per_slide': 3})
env.step(SlideForgeAction(tool='web_search', parameters={'query': 'AI overview'}))
env.step(SlideForgeAction(tool='create_outline', parameters={'sections': [
    {'title': 'Intro', 'bullet_points': ['Def', 'History', 'Today']},
    {'title': 'Apps', 'bullet_points': ['Health', 'Finance', 'Education']},
    {'title': 'Future', 'bullet_points': ['Trends', 'Risks', 'Ethics']},
]}))
for i, (title, secs) in enumerate([
    ('Introduction to AI', [
        {'heading': 'What is AI?', 'body': 'Artificial intelligence is the simulation of human intelligence by machines.'},
        {'heading': 'Brief History', 'body': 'AI research began in the 1950s with pioneers like Alan Turing and John McCarthy.'},
        {'heading': 'AI Today', 'body': 'Modern AI powers search engines, virtual assistants, and recommendation systems.'},
    ]),
    ('AI Applications', [
        {'heading': 'Healthcare', 'body': 'AI helps diagnose diseases and develop new treatments faster than ever.'},
        {'heading': 'Finance', 'body': 'AI algorithms detect fraud and optimize trading strategies automatically.'},
        {'heading': 'Education', 'body': 'AI personalizes learning experiences and automates assessment tasks.'},
    ]),
    ('The Future of AI', [
        {'heading': 'Emerging Trends', 'body': 'Generative AI and multimodal models are transforming industries worldwide.'},
        {'heading': 'Risks', 'body': 'AI raises concerns about bias, privacy, job displacement, and safety.'},
        {'heading': 'Ethics', 'body': 'Responsible AI development requires transparency, fairness, and accountability.'},
    ]),
]):
    env.step(SlideForgeAction(tool='generate_slide', parameters={
        'slide_idx': i, 'title': title, 'sections': secs,
    }))

state = env.state

# Test individual rewards
from rewards.code_rules import code_rules_reward
from rewards.render_quality import render_quality_reward
from rewards.content_quality import content_quality_reward

completions = ['dummy']
kwargs = {'states': [state]}

cr = code_rules_reward(completions, **kwargs)
print(f'code_rules:      {cr[0]:.3f}  (expected ~0.94)')
assert 0.8 <= cr[0] <= 1.0, f'code_rules out of range: {cr[0]}'

rq = render_quality_reward(completions, **kwargs)
print(f'render_quality:  {rq[0]:.3f}  (expected ~0.70)')
assert 0.5 <= rq[0] <= 1.0, f'render_quality out of range: {rq[0]}'

cq = content_quality_reward(completions, **kwargs)
print(f'content_quality: {cq[0]:.3f}  (expected ~0.65-0.80)')
assert 0.4 <= cq[0] <= 1.0, f'content_quality out of range: {cq[0]}'

# Test aggregator (excluding dino since it needs torch)
from rewards.aggregator import compute_reward_details, aggregate_rewards

details = compute_reward_details(
    completions,
    weights={'code_rules': 1.0, 'render_quality': 2.0, 'content_quality': 2.0},
    **kwargs,
)
print(f'Reward details:  {details[0]}')
assert 'aggregate' in details[0]
assert 0.5 <= details[0]['aggregate'] <= 1.0

agg = aggregate_rewards(
    completions,
    weights={'code_rules': 1.0, 'render_quality': 2.0, 'content_quality': 2.0},
    **kwargs,
)
print(f'Aggregate:       {agg[0]:.3f}')

print()
print('=== ALL REWARD TESTS PASSED ===')
"
```

---

## 8. Test FastAPI Server (TestClient, No Server Process Needed)

```bash
source .venv/bin/activate
PYTHONPATH=. python3 -c "
from fastapi.testclient import TestClient
from envs.slideforge_env.server.app import app

client = TestClient(app)

# Health
resp = client.get('/health')
assert resp.status_code == 200
assert resp.json() == {'status': 'ok'}
print('1. GET /health OK')

# Reset
resp = client.post('/reset', json={'topic': 'AI Overview', 'num_slides': 3})
assert resp.status_code == 200
data = resp.json()
assert data['phase'] == 'RESEARCH'
assert data['success'] == True
assert data['done'] == False
print(f'2. POST /reset OK: phase={data[\"phase\"]}')

# Step - web_search
resp = client.post('/step', json={'tool': 'web_search', 'parameters': {'query': 'AI trends'}})
assert resp.status_code == 200
data = resp.json()
assert data['success'] == True
print(f'3. POST /step web_search OK')

# Step - create_outline
resp = client.post('/step', json={'tool': 'create_outline', 'parameters': {
    'sections': [
        {'title': 'Intro', 'bullet_points': ['A', 'B', 'C']},
        {'title': 'Middle', 'bullet_points': ['D', 'E', 'F']},
        {'title': 'End', 'bullet_points': ['G', 'H', 'I']},
    ]
}})
assert resp.status_code == 200
data = resp.json()
assert data['success'] == True
assert data['phase'] == 'PLAN'
print(f'4. POST /step create_outline OK: phase={data[\"phase\"]}')

# Step - generate_slide
resp = client.post('/step', json={'tool': 'generate_slide', 'parameters': {
    'slide_idx': 0,
    'title': 'Introduction',
    'sections': [
        {'heading': 'Overview', 'body': 'Content about AI and its impact on society.'},
        {'heading': 'Goals', 'body': 'What we will cover in this presentation.'},
        {'heading': 'Agenda', 'body': 'Research, applications, and future directions.'},
    ]
}})
assert resp.status_code == 200
data = resp.json()
assert data['success'] == True
assert data['current_slide_count'] == 1
print(f'5. POST /step generate_slide OK: count={data[\"current_slide_count\"]}')

# State
resp = client.get('/state')
assert resp.status_code == 200
data = resp.json()
assert data['slide_count'] == 1
assert data['outline_count'] == 3
assert data['research_count'] == 1
print(f'6. GET /state OK: slides={data[\"slide_count\"]}, outline={data[\"outline_count\"]}, research={data[\"research_count\"]}')

# Step - unknown tool
resp = client.post('/step', json={'tool': 'nonexistent', 'parameters': {}})
assert resp.status_code == 200
data = resp.json()
assert data['success'] == False
print(f'7. POST /step unknown tool OK: success=False')

# Step - finalize
resp = client.post('/step', json={'tool': 'finalize', 'parameters': {}})
assert resp.status_code == 200
data = resp.json()
assert data['done'] == True
assert data['phase'] == 'DONE'
print(f'8. POST /step finalize OK: done=True')

print()
print('=== ALL FASTAPI TESTS PASSED ===')
"
```

---

## 9. Test FastAPI Server (Live Process + curl)

Run this in two terminals.

**Terminal 1 - Start server:**

```bash
cd /home/ubuntu/github/openenv-rl
source .venv/bin/activate
PYTHONPATH=. uvicorn envs.slideforge_env.server.app:app --host 0.0.0.0 --port 8000
```

**Terminal 2 - curl commands:**

```bash
# Health
curl -s http://localhost:8000/health | python3 -m json.tool

# Reset
curl -s -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"topic": "Quantum Computing", "num_slides": 2, "colors": 0.9}' \
  | python3 -m json.tool

# Search
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"tool": "web_search", "parameters": {"query": "quantum computing basics"}}' \
  | python3 -m json.tool

# Set theme
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"tool": "set_theme", "parameters": {"theme_name": "dark"}}' \
  | python3 -m json.tool

# Generate slide
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"tool": "generate_slide", "parameters": {"slide_idx": 0, "title": "Quantum Basics", "sections": [{"heading": "Qubits", "body": "A qubit is the basic unit of quantum information."}, {"heading": "Superposition", "body": "Qubits can exist in multiple states simultaneously."}]}}' \
  | python3 -m json.tool

# State
curl -s http://localhost:8000/state | python3 -m json.tool

# Review
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"tool": "review_deck", "parameters": {}}' \
  | python3 -m json.tool

# Finalize
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"tool": "finalize", "parameters": {}}' \
  | python3 -m json.tool
```

Stop server in Terminal 1 with `Ctrl+C`.

---

## 10. Test Tool Call Extraction (Training Component)

```bash
source .venv/bin/activate
PYTHONPATH=. python3 -c "
from training.grpo_trainer import extract_tool_call

# Fenced JSON block
tc = extract_tool_call('I will search now:\n\`\`\`json\n{\"tool\": \"web_search\", \"parameters\": {\"query\": \"AI\"}}\n\`\`\`')
assert tc == {'tool': 'web_search', 'parameters': {'query': 'AI'}}
print(f'1. Fenced block: {tc}')

# Inline JSON with nested braces
tc = extract_tool_call('My action: {\"tool\": \"generate_slide\", \"parameters\": {\"slide_idx\": 0, \"title\": \"Intro\", \"sections\": [{\"heading\": \"H\", \"body\": \"B\"}]}}')
assert tc is not None
assert tc['tool'] == 'generate_slide'
assert tc['parameters']['slide_idx'] == 0
print(f'2. Inline nested: {tc[\"tool\"]} with {len(tc[\"parameters\"])} params')

# No tool call
tc = extract_tool_call('I am thinking about what to do next...')
assert tc is None
print(f'3. No tool call: {tc}')

# Simple inline
tc = extract_tool_call('{\"tool\": \"finalize\", \"parameters\": {}}')
assert tc == {'tool': 'finalize', 'parameters': {}}
print(f'4. Simple inline: {tc}')

print()
print('=== ALL EXTRACTION TESTS PASSED ===')
"
```

---

## 11. Test GRPO Reward Function

```bash
source .venv/bin/activate
PYTHONPATH=. python3 -c "
from training.grpo_trainer import slideforge_reward

# Valid tool call that generates a slide
scores = slideforge_reward(
    ['{\"tool\": \"generate_slide\", \"parameters\": {\"slide_idx\": 0, \"title\": \"AI Intro\", \"sections\": [{\"heading\": \"What is AI\", \"body\": \"Artificial intelligence simulates human thinking.\"}]}}'],
    briefs=[{'topic': 'AI', 'num_slides': 3}],
)
print(f'1. Valid generate_slide: score={scores[0]:.3f}')
assert scores[0] > 0, f'Expected positive score, got {scores[0]}'

# Valid tool call - web search
scores = slideforge_reward(
    ['{\"tool\": \"web_search\", \"parameters\": {\"query\": \"AI trends\"}}'],
    briefs=[{'topic': 'AI', 'num_slides': 3}],
)
print(f'2. Valid web_search: score={scores[0]:.3f}')

# No tool call
scores = slideforge_reward(
    ['I am not sure what to do. Let me think...'],
    briefs=[{'topic': 'AI'}],
)
assert scores[0] == -2.0
print(f'3. No tool call: score={scores[0]}')

# Invalid tool
scores = slideforge_reward(
    ['{\"tool\": \"nonexistent_tool\", \"parameters\": {}}'],
    briefs=[{'topic': 'AI'}],
)
assert scores[0] == -1.0
print(f'4. Invalid tool: score={scores[0]}')

# Multiple completions
scores = slideforge_reward(
    [
        '{\"tool\": \"web_search\", \"parameters\": {\"query\": \"test\"}}',
        'no tool here',
        '{\"tool\": \"finalize\", \"parameters\": {}}',
    ],
    briefs=[{'topic': 'AI'}, {'topic': 'ML'}, {'topic': 'DL'}],
)
print(f'5. Batch: scores={scores}')
assert len(scores) == 3
assert scores[1] == -2.0

print()
print('=== ALL GRPO REWARD TESTS PASSED ===')
"
```

---

## 12. Test Prompt Formatting

```bash
source .venv/bin/activate
PYTHONPATH=. python3 -c "
from training.prompts import format_prompt, AGENT_PROMPT
from envs.slideforge_env.models import SlideForgeState, SlideBrief

state = SlideForgeState(
    brief=SlideBrief(topic='Quantum Computing', audience='physicists', num_slides=8, colors=0.9),
    phase='GENERATE',
    research_context=[{'query': 'q1'}, {'query': 'q2'}],
    slides_html=['<html>slide1</html>', '<html>slide2</html>', ''],
)

prompt = format_prompt(state)

assert 'Quantum Computing' in prompt
assert 'physicists' in prompt
assert '8' in prompt
assert '0.9' in prompt
assert 'GENERATE' in prompt
assert '2/8' in prompt       # 2 slides created out of 8
assert '2' in prompt          # 2 research items
print(f'Prompt length: {len(prompt)} chars')
print(f'Contains topic: {\"Quantum Computing\" in prompt}')
print(f'Contains audience: {\"physicists\" in prompt}')
print(f'Contains phase: {\"GENERATE\" in prompt}')
print()
print('=== PROMPT FORMATTING TEST PASSED ===')
"
```

---

## 13. Test Edge Cases

```bash
source .venv/bin/activate
PYTHONPATH=. python3 -c "
from envs.slideforge_env.server.environment import SlideForgeEnvironment
from envs.slideforge_env.models import SlideForgeAction

env = SlideForgeEnvironment()

# --- Edge case: confidence=1.0 skips search ---
env.reset(brief={'topic': 'Test', 'confidence': 1.0})
obs = env.step(SlideForgeAction(tool='web_search', parameters={'query': 'anything'}))
assert obs.success == True
assert 'Skipped' in obs.result
print(f'1. High confidence skip: {obs.result}')

# --- Edge case: invalid theme ---
obs = env.step(SlideForgeAction(tool='set_theme', parameters={'theme_name': 'nonexistent'}))
assert obs.success == False
assert 'Unknown theme' in obs.result
print(f'2. Invalid theme: {obs.result}')

# --- Edge case: edit nonexistent slide ---
obs = env.step(SlideForgeAction(tool='edit_slide', parameters={'slide_idx': 99}))
assert obs.success == False
print(f'3. Edit missing slide: {obs.result}')

# --- Edge case: revise outline out of range ---
obs = env.step(SlideForgeAction(tool='revise_outline', parameters={'slide_index': 5}))
assert obs.success == False
print(f'4. Revise out of range: {obs.result}')

# --- Edge case: finalize with no slides ---
obs = env.step(SlideForgeAction(tool='finalize', parameters={}))
assert obs.success == False
assert 'no slides' in obs.result.lower()
print(f'5. Finalize empty: {obs.result}')

# --- Edge case: empty outline ---
obs = env.step(SlideForgeAction(tool='create_outline', parameters={'sections': []}))
assert obs.success == False
print(f'6. Empty outline: {obs.result}')

# --- Edge case: bad parameters ---
obs = env.step(SlideForgeAction(tool='generate_slide', parameters={'wrong_param': True}))
assert obs.success == False
assert 'Invalid parameters' in obs.result
print(f'7. Bad params: {obs.result[:60]}')

# --- Edge case: unknown tool ---
obs = env.step(SlideForgeAction(tool='fly_to_moon', parameters={}))
assert obs.success == False
assert 'Unknown tool' in obs.result
print(f'8. Unknown tool: {obs.result[:60]}')

# --- Edge case: max steps ---
env2 = SlideForgeEnvironment()
env2._max_steps = 3
env2.reset(brief={'topic': 'Short'})
env2.step(SlideForgeAction(tool='web_search', parameters={'query': 'a'}))
env2.step(SlideForgeAction(tool='web_search', parameters={'query': 'b'}))
env2.step(SlideForgeAction(tool='web_search', parameters={'query': 'c'}))
obs = env2.step(SlideForgeAction(tool='web_search', parameters={'query': 'd'}))
assert obs.done == True
assert 'Max steps' in obs.result
print(f'9. Max steps: {obs.result}')

print()
print('=== ALL EDGE CASE TESTS PASSED ===')
"
```

---

## 14. Test Client Library (Against TestClient)

```bash
source .venv/bin/activate
PYTHONPATH=. python3 -c "
from fastapi.testclient import TestClient
from envs.slideforge_env.server.app import app
from envs.slideforge_env.client import SlideForgeClient
from envs.slideforge_env.models import SlideForgeAction

# Patch client to use TestClient transport
import httpx
test_client = TestClient(app)

client = SlideForgeClient.__new__(SlideForgeClient)
client.base_url = 'http://testserver'
client._client = test_client

# Use client
h = client.health()
assert h == {'status': 'ok'}
print(f'1. health(): {h}')

r = client.reset(topic='Testing', num_slides=2)
assert r['phase'] == 'RESEARCH'
print(f'2. reset(): phase={r[\"phase\"]}')

s = client.step(SlideForgeAction(tool='web_search', parameters={'query': 'test'}))
assert s['success'] == True
print(f'3. step(): success={s[\"success\"]}')

st = client.state()
assert st['research_count'] == 1
print(f'4. state(): research_count={st[\"research_count\"]}')

print()
print('=== CLIENT LIBRARY TESTS PASSED ===')
"
```

---

## 15. Test Trajectory Rollouts (Without AWS Credentials)

Tests the full rollouts pipeline using a simulated trajectory (no Bedrock call needed).

```bash
source .venv/bin/activate
PYTHONPATH=. python3 -c "
import json

# --- Import ---
from training.rollouts import (
    run_rollout, run_batch, trajectories_to_dataset,
    push_to_hub, extract_tool_call, SYSTEM_PROMPT,
    DEFAULT_BRIEFS, _build_messages_list,
)
print('1. Import OK')

# --- Default briefs ---
for b in DEFAULT_BRIEFS:
    assert 'topic' in b and 'audience' in b and 'num_slides' in b
print(f'2. {len(DEFAULT_BRIEFS)} default briefs OK')

# --- System prompt ---
assert 'SlideForge' in SYSTEM_PROMPT
assert 'finalize' in SYSTEM_PROMPT
print(f'3. System prompt OK ({len(SYSTEM_PROMPT)} chars)')

# --- Simulate a full trajectory ---
from envs.slideforge_env.server.environment import SlideForgeEnvironment
from envs.slideforge_env.models import SlideForgeAction
from training.prompts import format_prompt

brief = {'topic': 'AI Testing', 'audience': 'testers', 'num_slides': 2, 'sections_per_slide': 2}
env = SlideForgeEnvironment()
obs = env.reset(brief=brief)

mock_turns = [
    '{\"tool\": \"web_search\", \"parameters\": {\"query\": \"AI testing\"}}',
    '{\"tool\": \"create_outline\", \"parameters\": {\"sections\": [{\"title\": \"AI Basics\", \"bullet_points\": [\"Def\", \"Types\"]}, {\"title\": \"Testing\", \"bullet_points\": [\"Unit\", \"Integration\"]}]}}',
    '{\"tool\": \"set_theme\", \"parameters\": {\"theme_name\": \"tech\"}}',
    '{\"tool\": \"generate_slide\", \"parameters\": {\"slide_idx\": 0, \"title\": \"AI Basics\", \"sections\": [{\"heading\": \"What is AI\", \"body\": \"AI simulates human intelligence in machines.\"}, {\"heading\": \"Types\", \"body\": \"Supervised, unsupervised, and reinforcement learning.\"}]}}',
    '{\"tool\": \"generate_slide\", \"parameters\": {\"slide_idx\": 1, \"title\": \"AI Testing\", \"sections\": [{\"heading\": \"Unit Tests\", \"body\": \"Test individual AI components in isolation.\"}, {\"heading\": \"Integration\", \"body\": \"Test AI systems end to end for correctness.\"}]}}',
    '{\"tool\": \"review_deck\", \"parameters\": {}}',
    '{\"tool\": \"finalize\", \"parameters\": {}}',
]

trajectory = {
    'episode_id': env.state.episode_id, 'brief': brief, 'turns': [],
    'total_steps': 0, 'final_phase': 'RESEARCH', 'completed': False,
    'slides_created': 0, 'theme': 'default', 'rewards': {}, 'messages': [],
}

for i, mock_text in enumerate(mock_turns):
    tc = extract_tool_call(mock_text)
    assert tc is not None
    action = SlideForgeAction(tool=tc['tool'], parameters=tc['parameters'])
    obs = env.step(action)
    trajectory['turns'].append({
        'turn': i, 'assistant': mock_text, 'tool_call': tc,
        'observation': obs.result, 'success': obs.success,
        'phase': obs.phase, 'slide_count': obs.current_slide_count, 'done': obs.done,
    })
    if obs.done:
        break

state = env.state
trajectory['total_steps'] = state.step_count
trajectory['final_phase'] = state.phase
trajectory['completed'] = state.done and state.phase == 'DONE'
trajectory['slides_created'] = sum(1 for h in state.slides_html if h)
trajectory['theme'] = state.theme

from rewards.aggregator import compute_reward_details
details = compute_reward_details(['x'], weights={'code_rules': 1.0, 'render_quality': 2.0, 'content_quality': 2.0}, states=[state])
trajectory['rewards'] = details[0]
trajectory['messages'] = _build_messages_list(trajectory)

assert trajectory['completed'] == True
assert trajectory['slides_created'] == 2
assert trajectory['final_phase'] == 'DONE'
assert len(trajectory['turns']) == 7
assert trajectory['rewards']['aggregate'] > 0.5
print(f'4. Simulated trajectory: {len(trajectory[\"turns\"])} turns, completed={trajectory[\"completed\"]}, reward={trajectory[\"rewards\"][\"aggregate\"]:.3f}')

# --- Convert to HuggingFace dataset ---
dataset = trajectories_to_dataset([trajectory])
assert len(dataset) == 1
assert 'episode_id' in dataset.column_names
assert 'messages' in dataset.column_names
assert 'reward_aggregate' in dataset.column_names
assert 'turns' in dataset.column_names
assert 'topic' in dataset.column_names
assert dataset[0]['completed'] == True
assert dataset[0]['slides_created'] == 2
print(f'5. HF Dataset: {len(dataset)} rows, {len(dataset.column_names)} columns')

# --- Verify messages structure ---
msgs = json.loads(dataset[0]['messages'])
assert msgs[0]['role'] == 'system'
assert msgs[1]['role'] == 'user'
assert msgs[2]['role'] == 'assistant'
assert msgs[3]['role'] == 'user'
roles = [m['role'] for m in msgs]
assert roles.count('assistant') == 7  # one per tool call
print(f'6. Messages: {len(msgs)} messages, {roles.count(\"assistant\")} assistant turns')

# --- Verify turns structure ---
turns = json.loads(dataset[0]['turns'])
assert len(turns) == 7
assert turns[0]['tool_call']['tool'] == 'web_search'
assert turns[-1]['tool_call']['tool'] == 'finalize'
assert turns[-1]['done'] == True
print(f'7. Turns JSON: {len(turns)} turns, first={turns[0][\"tool_call\"][\"tool\"]}, last={turns[-1][\"tool_call\"][\"tool\"]}')

# --- Multiple trajectories ---
traj2 = dict(trajectory)
traj2['episode_id'] = 'second-episode'
traj2['elapsed_seconds'] = 42.5
dataset2 = trajectories_to_dataset([trajectory, traj2])
assert len(dataset2) == 2
print(f'8. Multi-trajectory dataset: {len(dataset2)} rows')

print()
print('=== ALL ROLLOUT TESTS PASSED ===')
"
```

---

## 16. Test Rollouts with Live Bedrock (Requires AWS Credentials)

Only run this if you have AWS credentials configured with Bedrock access.

```bash
source .venv/bin/activate

# Single rollout with one brief
PYTHONPATH=. python3 -c "
from training.rollouts import run_rollout

trajectory = run_rollout(
    brief={'topic': 'Python Programming', 'audience': 'beginners', 'num_slides': 3, 'sections_per_slide': 2},
    max_turns=20,
    temperature=0.7,
    verbose=True,
)

print(f'Completed: {trajectory[\"completed\"]}')
print(f'Slides: {trajectory[\"slides_created\"]}')
print(f'Turns: {len(trajectory[\"turns\"])}')
print(f'Rewards: {trajectory[\"rewards\"]}')
print(f'Messages: {len(trajectory[\"messages\"])}')
"
```

Full batch run + push to HuggingFace:

```bash
# Generate trajectories (all 10 default briefs)
PYTHONPATH=. python training/rollouts.py \
  --num-rollouts 1 \
  --max-turns 25 \
  --output outputs/trajectories.json

# Generate + push to HuggingFace Hub
PYTHONPATH=. python training/rollouts.py \
  --num-rollouts 2 \
  --push-to-hub your-username/slideforge-trajectories \
  --public
```

---

## 17. Run ALL Tests At Once

One-shot command that runs every test above in sequence:

```bash
cd /home/ubuntu/github/openenv-rl
source .venv/bin/activate

PYTHONPATH=. python3 -c "
print('='*60)
print('SLIDEFORGE FULL VERIFICATION SUITE')
print('='*60)
print()

# --- Models ---
from envs.slideforge_env.models import SlideBrief, SlideForgeAction, SlideForgeObservation, SlideForgeState
SlideBrief(topic='Test')
SlideForgeAction(tool='test', parameters={})
SlideForgeState()
print('[PASS] Models import and instantiate')

# --- Themes ---
from envs.slideforge_env.server.rendering.themes import resolve_theme, THEMES
assert len(THEMES) == 5
for name in THEMES:
    t = resolve_theme(name, 0.5)
    assert 'rgb' in t['accent']
print(f'[PASS] Themes: {len(THEMES)} themes resolve correctly')

# --- HTML Generator ---
from envs.slideforge_env.server.rendering.html_generator import generate_slide_html
html = generate_slide_html('T', [{'heading': 'H', 'body': 'B'}], 'default', 0.5, 0, 1)
assert '<!DOCTYPE html>' in html
print(f'[PASS] HTML Generator: {len(html)} chars')

# --- Environment Full Episode ---
from envs.slideforge_env.server.environment import SlideForgeEnvironment
env = SlideForgeEnvironment()
obs = env.reset(brief={'topic': 'AI', 'num_slides': 2, 'sections_per_slide': 2})
assert obs.phase == 'RESEARCH' and obs.success
obs = env.step(SlideForgeAction(tool='web_search', parameters={'query': 'AI'}))
assert obs.success
obs = env.step(SlideForgeAction(tool='create_outline', parameters={'sections': [
    {'title': 'S1', 'bullet_points': ['a', 'b']},
    {'title': 'S2', 'bullet_points': ['c', 'd']},
]}))
assert obs.phase == 'PLAN'
obs = env.step(SlideForgeAction(tool='set_theme', parameters={'theme_name': 'dark'}))
assert obs.success
for i in range(2):
    obs = env.step(SlideForgeAction(tool='generate_slide', parameters={
        'slide_idx': i, 'title': f'Slide {i}',
        'sections': [{'heading': 'H1', 'body': 'Content about AI.'}, {'heading': 'H2', 'body': 'More about AI.'}],
    }))
    assert obs.success
assert obs.current_slide_count == 2
obs = env.step(SlideForgeAction(tool='edit_slide', parameters={'slide_idx': 0, 'title': 'New Title'}))
assert obs.success and obs.phase == 'REFINE'
obs = env.step(SlideForgeAction(tool='review_deck', parameters={}))
assert obs.success
obs = env.step(SlideForgeAction(tool='finalize', parameters={}))
assert obs.done and obs.phase == 'DONE'
print('[PASS] Full episode: reset -> research -> plan -> generate -> edit -> review -> finalize')

# --- Rewards ---
state = env.state
from rewards.code_rules import code_rules_reward
from rewards.render_quality import render_quality_reward
from rewards.content_quality import content_quality_reward
from rewards.aggregator import aggregate_rewards, compute_reward_details

cr = code_rules_reward(['x'], states=[state])[0]
rq = render_quality_reward(['x'], states=[state])[0]
cq = content_quality_reward(['x'], states=[state])[0]
assert 0 <= cr <= 1 and 0 <= rq <= 1 and 0 <= cq <= 1
agg = aggregate_rewards(['x'], weights={'code_rules': 1, 'render_quality': 2, 'content_quality': 2}, states=[state])[0]
det = compute_reward_details(['x'], weights={'code_rules': 1, 'render_quality': 2, 'content_quality': 2}, states=[state])[0]
assert 'aggregate' in det
print(f'[PASS] Rewards: cr={cr:.2f} rq={rq:.2f} cq={cq:.2f} agg={agg:.2f}')

# --- FastAPI ---
from fastapi.testclient import TestClient
from envs.slideforge_env.server.app import app
tc = TestClient(app)
assert tc.get('/health').status_code == 200
assert tc.post('/reset', json={'topic': 'T'}).status_code == 200
assert tc.post('/step', json={'tool': 'web_search', 'parameters': {'query': 'q'}}).status_code == 200
assert tc.get('/state').status_code == 200
print('[PASS] FastAPI: /health, /reset, /step, /state all return 200')

# --- Tool Call Extraction ---
from training.grpo_trainer import extract_tool_call
assert extract_tool_call('{\"tool\": \"finalize\", \"parameters\": {}}') == {'tool': 'finalize', 'parameters': {}}
assert extract_tool_call('no json') is None
assert extract_tool_call('{\"tool\": \"gen\", \"parameters\": {\"a\": {\"b\": 1}}}')['tool'] == 'gen'
print('[PASS] Tool call extraction: fenced, inline, nested, none')

# --- GRPO Reward ---
from training.grpo_trainer import slideforge_reward
s1 = slideforge_reward(['{\"tool\": \"web_search\", \"parameters\": {\"query\": \"x\"}}'], briefs=[{'topic': 'AI'}])
s2 = slideforge_reward(['no tool'], briefs=[{'topic': 'AI'}])
assert s2[0] == -2.0
print(f'[PASS] GRPO reward: valid={s1[0]:.2f}, invalid={s2[0]}')

# --- Prompt ---
from training.prompts import format_prompt
from envs.slideforge_env.models import SlideForgeState as SS, SlideBrief as SB
p = format_prompt(SS(brief=SB(topic='X')))
assert 'X' in p and 'web_search' in p
print(f'[PASS] Prompt template: {len(p)} chars')

# --- Edge Cases ---
env3 = SlideForgeEnvironment()
env3.reset(brief={'topic': 'T', 'confidence': 1.0})
o = env3.step(SlideForgeAction(tool='web_search', parameters={'query': 'x'}))
assert 'Skipped' in o.result
o = env3.step(SlideForgeAction(tool='set_theme', parameters={'theme_name': 'bad'}))
assert not o.success
o = env3.step(SlideForgeAction(tool='fly', parameters={}))
assert not o.success
print('[PASS] Edge cases: confidence skip, bad theme, unknown tool')

# --- Rollouts (simulated, no AWS) ---
import json as _json
from training.rollouts import (
    trajectories_to_dataset, _build_messages_list, SYSTEM_PROMPT, DEFAULT_BRIEFS,
)
assert len(DEFAULT_BRIEFS) == 10
env4 = SlideForgeEnvironment()
env4.reset(brief={'topic': 'Test', 'num_slides': 2, 'sections_per_slide': 2})
for tc_str in [
    '{\"tool\": \"create_outline\", \"parameters\": {\"sections\": [{\"title\": \"A\", \"bullet_points\": [\"x\",\"y\"]}, {\"title\": \"B\", \"bullet_points\": [\"x\",\"y\"]}]}}',
    '{\"tool\": \"generate_slide\", \"parameters\": {\"slide_idx\": 0, \"title\": \"A\", \"sections\": [{\"heading\": \"H\", \"body\": \"Test content.\"}, {\"heading\": \"H2\", \"body\": \"More.\"}]}}',
    '{\"tool\": \"generate_slide\", \"parameters\": {\"slide_idx\": 1, \"title\": \"B\", \"sections\": [{\"heading\": \"H\", \"body\": \"Test content.\"}, {\"heading\": \"H2\", \"body\": \"More.\"}]}}',
    '{\"tool\": \"finalize\", \"parameters\": {}}',
]:
    tc = extract_tool_call(tc_str)
    env4.step(SlideForgeAction(tool=tc['tool'], parameters=tc['parameters']))
traj = {
    'episode_id': 'test', 'brief': {'topic': 'Test', 'num_slides': 2, 'sections_per_slide': 2},
    'turns': [{'turn': i, 'assistant': t, 'tool_call': extract_tool_call(t), 'observation': 'ok', 'success': True, 'phase': 'DONE', 'slide_count': 2, 'done': i==3}
              for i, t in enumerate([
        '{\"tool\": \"create_outline\", \"parameters\": {\"sections\": [{\"title\": \"A\", \"bullet_points\": [\"x\"]}]}}',
        '{\"tool\": \"generate_slide\", \"parameters\": {\"slide_idx\": 0, \"title\": \"A\", \"sections\": [{\"heading\": \"H\", \"body\": \"B\"}]}}',
        '{\"tool\": \"generate_slide\", \"parameters\": {\"slide_idx\": 1, \"title\": \"B\", \"sections\": [{\"heading\": \"H\", \"body\": \"B\"}]}}',
        '{\"tool\": \"finalize\", \"parameters\": {}}',
    ])],
    'total_steps': 4, 'final_phase': 'DONE', 'completed': True,
    'slides_created': 2, 'theme': 'default', 'rewards': {'aggregate': 0.75}, 'messages': [],
    'elapsed_seconds': 10.0,
}
traj['messages'] = _build_messages_list(traj)
ds = trajectories_to_dataset([traj])
assert len(ds) == 1 and 'messages' in ds.column_names and 'reward_aggregate' in ds.column_names
msgs = _json.loads(ds[0]['messages'])
assert msgs[0]['role'] == 'system'
print(f'[PASS] Rollouts: simulated trajectory -> HF dataset ({len(ds)} rows, {len(msgs)} messages)')

print()
print('='*60)
print('ALL TESTS PASSED')
print('='*60)
"
```
