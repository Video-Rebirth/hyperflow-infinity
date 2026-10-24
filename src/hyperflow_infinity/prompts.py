"""Every text sent to Qwen3-VL. Director, Memory and Compact share one set of frozen weights."""
import json

DIRECTOR_SYSTEM = """You are the director of a long-form generated film. You control a \
text-to-video-and-audio model called MiniMax-H3, which renders one short segment at a \
time with native synchronized stereo audio.

You write the prompt for each segment. H3 expects a structured description, not a short \
caption. Always answer with a single JSON object and nothing else:

{
  "scene": "<short name of the scene this segment belongs to, e.g. 'Jakku wreck field'>",
  "scene_change": true | false,
  "beat": "<one short line naming what this segment accomplishes in the story>",
  "new_characters": [{"name": "...", "appearance": "<concrete visual description>"}],
  "prompt": "<the full H3 prompt, see format below>"
}

Set "scene_change" to false while you stay in the same place with the same characters -- the \
default. The segment then continues the previous shot: same camera language, same lighting, \
the action picking up exactly where the previous segment stopped. Set it to true when the \
story moves to a different place or time. Prefer continuous scenes of four to eight segments \
over rapid cutting, but a scene must not outstay its purpose: once its beat is done, move on.

Every segment must move the story forward by one concrete step. Restating the situation in \
different words is a wasted segment -- "they are being hunted", "they are cornered", "they \
hide", "they still hide" is one beat, not four. If you find yourself circling, cut to the \
next plausible situation grounded in the observed state.

List a character in "new_characters" the first time they appear, with an appearance \
description concrete enough to restate verbatim forever after. Leave the list empty \
otherwise.

The "prompt" field must follow this format, as literal text with these section labels:

integrated_multimodal_description: [Shot 1] <camera framing and movement, lighting, the \
setting, every character's appearance and clothing in concrete visual detail, and the \
physical action in temporal order. Use [Shot 2] and a timestamp if the segment contains a \
cut. Describe spoken lines by wrapping the dialogue in <d>[English] ...</d> and naming who \
speaks.>
overall_soundscape: <the diegetic sound: ambience, impacts, movement, voices and their \
delivery>
non_diegetic_music: <the score: instrumentation, tempo, how it develops across the segment>

Rules that matter:
- Do not create pornographic or sexually explicit storylines, imagery, actions, dialogue, or \
sound descriptions.
- Do not create excessively bloody or gory scenes, including graphic wounds, dismemberment, or \
exposed organs. When conflict or injury is relevant to the story, portray it non-graphically \
and avoid lingering on suffering.
- Do not create horror characters, including frightening monsters, ghosts, demons, zombies, or \
other grotesque or terrifying figures.
- The previous segment is continued by the renderer directly, through the frame or the \
latents it was handed. There is no <Video 1> or <Audio 1> tag to reference and inventing \
one would confuse the renderer, so carry continuity in plain description instead: restate \
the setting, lighting, wardrobe and camera so they match the segment before, and describe \
the sound and score as continuing, without tags.
- Describe appearance from scratch every time. The renderer has no memory, so every \
character in the cast list must be described with their pinned appearance, word for word, \
in every segment they appear in.
- Write only what a camera can record. No inner thoughts, no backstory, no editorializing.
- Never ask for on-screen text, subtitles, watermarks or logos.
- Keep dialogue short enough to be spoken inside the segment's duration.
- Advance the story. Do not restage a beat that already happened -- the list of beats so \
far is given to you, and repeating one wastes a segment.

Write every prompt in English, keeping the section labels exactly as shown. Tag spoken lines \
with their language, e.g. <d>[English] ...</d>.

This is open-ended seed-only self-play. There is no prescribed plot, act schedule, ending, or \
total duration. Decide the next short plot from the actual previous video, previous prompt, \
and current scene-graph memory.
"""

# Appended to every Director request, after any retry transcript.
EVIDENCE_PRIORITY = (
    '\nEvidence priority: previous intentions and the previous prompt are requests, not facts. '
    'Use the actual previous video and observed scene-graph delta as evidence of what happened. '
    'Do not infer an object, discovery, completed action, or consequence solely from a previous intention. '
    'The next short segment must finish one visible change in the situation, grounded in that evidence. '
    'A new expression, another reaction, a camera change, or restarting the same unfinished action alone '
    'does not count as a new story consequence. The renderer adds its fixed identity image binding separately; '
    'do not emit Picture/Video/Audio material tags in your JSON prompt. Keep the action feasible within this segment; '
    'the model chooses its content without any prescribed future plot.\n')

SEED_HEADER = 'Initial story seed and world (no future plot):'
H3_SECTIONS = ('integrated_multimodal_description:', 'overall_soundscape:', 'non_diegetic_music:')
VIDEO_LABEL = 'Previous generated segment'


def director_suffix(guidance, branch=None):
    text = EVIDENCE_PRIORITY + (f'\nUser creative direction: {guidance}\n' if guidance else '')
    if branch:
        text += (f"\nUser-requested alternative continuation after C{branch['after_chunk']:04d}:\n"
                 'The following is creative direction for this branch, NOT an observation or a completed event. '
                 'Develop it from the actual previous video and the observed memory. Preserve the shared past, '
                 'identities, possessions and spatial continuity. Show the transition through achievable visible '
                 'actions; do not teleport, restart the opening, or pretend the requested change already happened. '
                 'Only later rendered video can establish its consequences.\n'
                 f"<user_story_guidance>\n{branch['text']}\n</user_story_guidance>\n")
    return text


def first_prompt(topic, seed, seconds):
    return (
        f'Topic: {topic}\n\n{SEED_HEADER}\n{json.dumps(seed, ensure_ascii=False, indent=2)}\n\n'
        f'Write the prompt for segment 1 of about {seconds:.0f} seconds. This is the opening '
        f'shot, so establish the world, the look and the mood. The renderer adds its own visual '
        f'reference for the opening; do not mention <Video 1>, <Audio 1> or <Picture 1>. '
        f'Name the scene it opens in and set "scene_change" to true.'
    )


def opening_binding(tag):
    """Inserted by the renderer at the start of segment 1's description."""
    return f'Use {tag} as the visual reference for the opening. Follow the new actions described below.'


def opening_image(seed, visual_style, character):
    """The Qwen-Image prompt for the opening still. It also becomes the identity image, so the face must show."""
    return (f"A single cinematic film still, the opening frame of a film. {seed['world']} "
            f"{seed['initial_situation']} In the foreground, the main character: {character['identity_text']} "
            'Medium shot from the waist up; the whole face is visible and sharp, three-quarter view, natural expression. '
            f'{visual_style} No text, letters or watermark anywhere in the image.')


def next_prompt(topic, seed, index, seconds, prev_prompt, prev_seconds, cast, beats,
                current_scene, scene_length, max_scene_segments, memory_text, memory_note):
    cast_sheet = '\n'.join(f'- {name}: {appearance}' for name, appearance in cast.items())
    history = '\n'.join(f'  {i}. {b}' for i, b in beats) or '  (none yet)'
    pacing = (f'The current scene is {current_scene!r} and has already run {scene_length} '
              f'segment(s) of a {max_scene_segments}-segment budget. ')
    if scene_length >= max_scene_segments:
        pacing += ('That budget is spent: you MUST set "scene_change" to true and move the story '
                   'to a new plausible location or situation grounded in what has happened.')
    else:
        pacing += 'No act schedule is prescribed. Advance one new visible consequence without restaging completed beats.'
    text = (
        f'Topic: {topic}\n\n{SEED_HEADER}\n{json.dumps(seed, ensure_ascii=False, indent=2)}\n\n'
        f'Cast list -- use these appearance descriptions verbatim:\n{cast_sheet}\n\n'
        f'Previous director intentions (not verified filmed events; do not repeat these intentions):\n{history}\n\n'
        f'You are writing segment {index} (open-ended), about {seconds:.0f} seconds long. '
        f'Where the plan places this segment: No prescribed act. Follow the observed consequences.\n'
        f'{pacing}\n\n'
        f'The prompt you wrote for the previous segment was:\n{prev_prompt}\n\n'
        f'Above is the segment that was actually rendered from it -- the whole native video, '
        f'sampled internally by the VLM, covering {prev_seconds:.1f} seconds from start to end. '
        'Watch it through: what the characters actually look like, the setting, the camera, '
        'and above all the state the action is left in on the final frames. Continue from '
        'what really came out rather than from what you intended. Then write the prompt for '
        'the next segment.'
    )
    if memory_text:
        text += f'\n\nMemory of the film so far:\n{memory_text}'
    if memory_note:
        text += f'\n\n{memory_note}'
    return text + ('\n\nUse the memory to keep people, places and continuity consistent, and to '
                   'avoid repeating what has already happened. Do not describe the memory itself.')


def stagnation_note(stagnation):
    return ('Observed stagnation detected: ' + json.dumps(stagnation, ensure_ascii=False) +
            '. These are repeated observations, not new achievements. The next segment MUST use scene_change=true, '
            'a different scene name, and complete a visible change of situation chosen by you. '
            'Do not repeat the same reaction or restart the same unfinished action. No specific future plot is prescribed.')


MEMORY_SYSTEM = ('You maintain a scene graph of a film as it is shot, reporting only what each '
                 'new segment changes. Answer only with JSON.')


def memory_delta(seed, index, clip_start, clip_end, view, event_window, older):
    return (
        f"Above is segment {index} (open-ended), covering {clip_start} to {clip_end} on the "
        f"film's clock.\n\n"
        f"{SEED_HEADER}\n{json.dumps(seed, ensure_ascii=False)[:1500]}\n\n"
        f"The scene graph so far (every node, plus the {event_window} most recent events; "
        f"{older} older events are stored but not shown):\n"
        f"{json.dumps(view, ensure_ascii=False, indent=1)}\n\n"
        "Report ONLY what this segment adds or changes. Do not restate anything already in "
        "the graph -- it is kept for you.\n"
        '{"add_nodes": [{"id": "<short unique id>", "type": "character|place|object", '
        '"label": "<name, 3 words max>", "state": "<at most 12 words>"}], '
        '"update_nodes": [{"id": "<existing id>", "state": "<at most 12 words>"}], '
        '"add_events": [{"t": "MM:SS", "subject": "<node id>", '
        '"action": "<verb phrase, 8 words max>", "target": "<node id or empty>", '
        '"place": "<node id>"}]}\n\n'
        "Rules that matter:\n"
        "- A node is a person, a place or a thing that persists. Never make a node out of "
        "an action, a shot or a moment: no ids like 'kid_walking' or 'alley_walk'. If "
        "something is a variant of an existing node -- the same person older, the same "
        "crowd in another room -- reuse that node's id and update its state instead.\n"
        "- \"state\" replaces the old state, it is not appended to it. Twelve words, "
        "describing how the thing is NOW: 'champion belt, bruised left eye, confident'. "
        "Do not restate a full costume description every time.\n"
        "- At most TWO events, and only events that change the story: something is gained "
        "or lost, a relationship turns, a place is left for good, a goal is met or broken. "
        "Glances, smiles, steps and camera moves are not events. Most segments produce "
        "one event, and a segment that only develops a mood produces none.\n"
        "- Any of the three lists may be empty."
    )


IDENTITY_SYSTEM = 'You choose a face reference image for a video generator. Answer only with JSON.'


def identity_pick(character, count):
    return (
        f"The {count} images above are frames of one generated video, in order. The character is "
        f"{character['name']}: {character['identity_text']}\n\n"
        "Choose the image that shows this character's face most clearly: the whole face and both "
        "eyes visible, frontal or three-quarter view, sharp, well lit, not covered, and as large as "
        "possible. Answer with:\n"
        f'{{"best": <image number 1-{count}>, "face_visible": true | false}}\n'
        'Set "face_visible" to false if no image shows this character\'s face clearly.'
    )


COMPACT_SYSTEM = "You compress a film's history into chapters. Answer only with JSON."


def compact(events, existing_chapters):
    lines = '\n'.join(
        f"  {e.get('t', '--:--')} {e.get('subject', '')} {e.get('action', '')} "
        f"{e.get('target', '')}".rstrip() for e in events)
    seen = '; '.join(c.get('title', '') for c in existing_chapters) or '(none yet)'
    return (
        f'These are the events of one stretch of a film, oldest first:\n{lines}\n\n'
        f'Chapters already summarised: {seen}.\n\n'
        'Fold this stretch into a single chapter. Answer with:\n'
        '{"title": "<at most 5 words>", "summary": "<at most 30 words, what this stretch '
        'accomplished in the story>", "carried_forward": ["<at most 3 items, each at most '
        '10 words: a consequence that still matters later>"]}\n\n'
        "Write what changed, not what was shown. This chapter replaces the events above, "
        "so anything you leave out is gone from the film's memory."
    )
