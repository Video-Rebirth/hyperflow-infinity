"""Scene Graph memory: merge per-segment deltas, fold old events into chapters, render a bounded view."""
import difflib
from pathlib import Path

from .store import read_json

STATE_WORDS = 12
EVENTS_PER_SEGMENT = 2


def clip_words(text, limit):
    return ' '.join(str(text or '').split()[:limit])


def seconds_to_mmss(value):
    value = int(round(value))
    return f'{value // 60:02d}:{value % 60:02d}'


def resolve_node_id(new_id, known):
    """Map a proposed id onto an existing node when it is plainly the same entity.

    Qwen does not reliably reuse ids (`opponent` / `opponent_boxer`, `Kai` / `Kai_adult`), so
    prefix matches and near-identical spellings are folded into the existing node.
    """
    if not new_id or new_id in known:
        return new_id
    for old in known:
        if new_id.startswith(f'{old}_') or old.startswith(f'{new_id}_'):
            return old
    match = difflib.get_close_matches(new_id, list(known), n=1, cutoff=0.82)
    return match[0] if match else new_id


def merge_scene_graph(graph, delta):
    """Accumulate one Memory delta. States are clipped to 12 words, events to two per segment."""
    graph = graph or {'nodes': [], 'events': [], 'chapters': []}
    graph.setdefault('chapters', [])
    by_id = {n.get('id'): n for n in graph['nodes']}
    remap = {}
    for node in delta.get('add_nodes') or []:
        raw = node.get('id')
        target = resolve_node_id(raw, by_id.keys())
        remap[raw] = target
        state = clip_words(node.get('state'), STATE_WORDS)
        if target in by_id:
            if state:
                by_id[target]['state'] = state
        else:
            graph['nodes'].append({'id': target, 'type': node.get('type') or 'character',
                                   'label': clip_words(node.get('label'), 3) or target,
                                   'state': state})
            by_id[target] = graph['nodes'][-1]
    for upd in delta.get('update_nodes') or []:
        node = by_id.get(remap.get(upd.get('id'), upd.get('id')))
        if node is not None and upd.get('state'):
            node['state'] = clip_words(upd['state'], STATE_WORDS)
    for ev in (delta.get('add_events') or [])[:EVENTS_PER_SEGMENT]:
        ev = dict(ev)
        for key in ('subject', 'target', 'place'):
            if ev.get(key):
                ev[key] = remap.get(ev[key], resolve_node_id(ev[key], by_id.keys()))
        ev['action'] = clip_words(ev.get('action'), 8)
        ev.pop('note', None)
        graph['events'].append(ev)
    return graph


def compact_scene_graph(graph, chapter, folded):
    """Replace the `folded` oldest events with one chapter. Returns the removed events."""
    graph.setdefault('chapters', []).append({
        'id': f"ch{len(graph['chapters']) + 1}",
        't_start': folded[0].get('t', ''), 't_end': folded[-1].get('t', ''),
        'title': clip_words(chapter.get('title'), 5),
        'summary': clip_words(chapter.get('summary'), 30),
        'carried_forward': [clip_words(x, 10) for x in (chapter.get('carried_forward') or [])[:3]],
    })
    graph['events'] = graph['events'][len(folded):]
    return folded


def graph_as_text(graph, recent_events=8):
    """Bounded prose view for the Director: chapters, current cast states, places, last events.

    Bounded because an unbounded view grew the Director's prompts with it, which diluted H3's
    text conditioning.
    """
    graph = graph or {}
    lines = []
    for ch in graph.get('chapters') or []:
        carried = '; '.join(ch.get('carried_forward') or [])
        lines.append(f"[{ch.get('t_start')}-{ch.get('t_end')}] {ch.get('title')}: "
                     f"{ch.get('summary')}"
                     + (f' (still matters: {carried})' if carried else ''))
    nodes = graph.get('nodes') or []
    cast = [n for n in nodes if (n.get('type') or 'character') == 'character']
    if cast:
        lines.append('Cast as they now are: ' +
                     '; '.join(f"{n.get('label')} -- {n.get('state')}" for n in cast))
    world = [n for n in nodes if (n.get('type') or 'character') != 'character']
    if world:
        lines.append('Places and things in play: ' +
                     ', '.join(str(n.get('label')) for n in world))
    tail = (graph.get('events') or [])[-recent_events:]
    if tail:
        lines.append('Just happened: ' + '; '.join(
            f"{e.get('t')} {e.get('subject')} {e.get('action')}"
            + (f" -> {e['target']}" if e.get('target') else '') for e in tail))
    return '\n'.join(lines)


def scene_length_of(segments):
    """Segments since (and including) the last one that opened a scene."""
    n = 0
    for done in reversed(segments):
        n += 1
        if done.get('scene_change'):
            break
    return n


def observation_signature(delta):
    """Comparable form of a Memory delta, ignoring timestamps. None when it carries no state."""
    if not delta or delta.get('add_nodes') or not delta.get('add_events') or not delta.get('update_nodes'):
        return None
    def normalize(value):
        return ' '.join(str(value or '').lower().split())

    events = sorted((normalize(e.get('subject')), normalize(e.get('action')), normalize(e.get('target')),
                     normalize(e.get('place'))) for e in delta['add_events'])
    nodes = sorted((normalize(n.get('id')), normalize(n.get('state'))) for n in delta['update_nodes'])
    if any(not e[1] for e in events) or any(not n[1] for n in nodes):
        return None
    return {'events': events, 'states': nodes}


def repeated_observations(root, chunks):
    """Detect three consecutive same-scene Memory deltas with identical events and states.

    Matching is literal after case/whitespace normalisation: a reworded action is not a repeat.
    """
    recent = chunks[-4:]
    if len(recent) < 3:
        return None
    current = chunks[-1]['scene']
    rows = []
    for c in recent:
        snapshot = read_json(Path(root) / 'memory_snapshots' / f"{c['chunk_id']}.json", {})
        signature = observation_signature(snapshot.get('delta')) if c['scene'] == current else None
        rows.append((c['chunk_id'], signature))
    for start in range(len(rows) - 2):
        group = rows[start:start + 3]
        if group[0][1] is not None and all(x[1] == group[0][1] for x in group):
            return {'chunks': [x[0] for x in group], 'observations': group[0][1],
                    'reason': 'three consecutive Memory deltas repeat the same event content and object states despite new timestamps'}
    return None
