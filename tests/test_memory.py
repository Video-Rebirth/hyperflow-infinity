import copy
import json
import tempfile
import unittest
from pathlib import Path

from hyperflow_infinity.memory import (
    compact_scene_graph,
    graph_as_text,
    merge_scene_graph,
    repeated_observations,
    scene_length_of,
)


class SceneGraph(unittest.TestCase):
    def test_variant_ids_merge_into_one_node(self):
        g = merge_scene_graph(None, {'add_nodes': [{'id': 'robinson', 'label': 'Robinson', 'state': 'wet'}]})
        g = merge_scene_graph(g, {'add_nodes': [{'id': 'robinson_older', 'label': 'Robinson', 'state': 'dry'}],
                                  'add_events': [{'subject': 'robinson_older', 'action': 'gathers wood'}]})
        self.assertEqual(len(g['nodes']), 1)
        self.assertEqual(g['nodes'][0]['state'], 'dry')
        self.assertEqual(g['events'][0]['subject'], 'robinson')

    def test_states_and_events_are_bounded(self):
        g = merge_scene_graph(None, {
            'add_nodes': [{'id': 'r', 'label': 'a b c d e', 'state': ' '.join(['w'] * 30)}],
            'add_events': [{'subject': 'r', 'action': ' '.join(['x'] * 20), 'note': 'dropped'}] * 5})
        self.assertEqual(g['nodes'][0]['label'], 'a b c')
        self.assertEqual(len(g['nodes'][0]['state'].split()), 12)
        self.assertEqual(len(g['events']), 2)
        self.assertEqual(len(g['events'][0]['action'].split()), 8)
        self.assertNotIn('note', g['events'][0])

    def test_compaction_folds_oldest_events_into_a_chapter(self):
        g = merge_scene_graph(None, {'add_nodes': [{'id': 'robinson', 'label': 'Robinson', 'state': 'dry'}]})
        g['events'] = [{'t': str(i), 'subject': 'robinson', 'action': str(i)} for i in range(23)]
        folded = copy.deepcopy(g['events'][:10])
        removed = compact_scene_graph(g, {'title': 'Landfall', 'summary': 's', 'carried_forward': list('abcd')}, folded)
        self.assertEqual(len(removed), 10)
        self.assertEqual(len(g['events']), 13)
        self.assertEqual(g['chapters'][0]['carried_forward'], ['a', 'b', 'c'])
        view = graph_as_text(g)
        self.assertIn('[0-9] Landfall: s (still matters: a; b; c)', view)
        self.assertIn('Cast as they now are: Robinson -- dry', view)

    def test_scene_length_counts_back_to_the_last_scene_change(self):
        chunks = [{'scene_change': True}, {'scene_change': False}, {'scene_change': True},
                  {'scene_change': False}, {'scene_change': False}]
        self.assertEqual(scene_length_of(chunks), 3)


class Stagnation(unittest.TestCase):
    def test_identical_deltas_ignore_time_but_not_wording(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'memory_snapshots').mkdir()
            chunks = [{'chunk_id': f'C{i:04}', 'scene': 'shore'} for i in range(1, 4)]
            delta = {'add_nodes': [], 'update_nodes': [{'id': 'r', 'state': 'crouching'}],
                     'add_events': [{'subject': 'r', 'action': 'looks up', 'target': 'sky', 'place': 'shore'}]}
            for i, c in enumerate(chunks):
                d = copy.deepcopy(delta)
                d['add_events'][0]['t'] = str(i)
                (root / 'memory_snapshots' / f"{c['chunk_id']}.json").write_text(json.dumps({'delta': d}))
            self.assertIsNotNone(repeated_observations(root, chunks))
            d['add_events'][0]['action'] = 'raises his gaze'
            (root / 'memory_snapshots/C0003.json').write_text(json.dumps({'delta': d}))
            self.assertIsNone(repeated_observations(root, chunks))


if __name__ == '__main__':
    unittest.main()
