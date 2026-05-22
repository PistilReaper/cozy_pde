import json
with open('research_memory.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
print('iteration:', d.get('iteration'))
print('stop_reason:', d.get('stop_reason'))
print('current_phase:', d.get('current_phase'))
print('experiments count:', len(d.get('experiments', [])))
for e in d.get('experiments', []):
    score = e.get('metrics', {}).get('best_score', 'N/A')
    hypo = e.get('hypothesis', '')[:60]
    print(f"  Exp {e['id']}: {e['status']} | score={score} | {hypo}")
