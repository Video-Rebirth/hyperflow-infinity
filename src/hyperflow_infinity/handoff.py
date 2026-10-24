"""Ref2VA conditioning for every segment: a native reference opening, then latent handoff.

The official Ref2VA workflow takes RGB references and VAE-encodes them. This module patches five
diffusers MiniMax-H3 hooks so that each request receives exactly the conditions it names:

- Segment 1 (the opening): one image or one video reference, through the native path, unchanged.
- Every later segment: one fixed identity image, through the native image path (VAE latent + vision
  tokens), and the previous segment's last 7 normalized video latents, loaded from disk, never
  re-encoded.

The seven latents get the positions of the first seven target latents, so the new segment opens
by redrawing the end of the previous one; those frames are trimmed afterwards. Each call leaves
an audit that `engine.validate_conditioning` checks before the segment is committed.
"""
from types import SimpleNamespace

TAIL_LATENTS = 7
CONTRACT = 'ref2va_identity_tail_v1'
OPENING_CONTRACT = 'ref2va_opening_v1'

_state = {'tail': None, 'mode': None, 'audit': None, 'opening': None}
_installed = False


def set_boundary(path, mode, identity_anchor=None, opening=None):
    """Arm the next request. `path` is the previous segment's `latents.pt` (None for the opening).

    `opening` is `{'kind': 'image' | 'video', 'sha256': ...}` for segment 1 and None afterwards.
    """
    import torch
    _state.update(tail=None, mode=mode, audit=None, identity_anchor=identity_anchor, opening=opening,
                  vae_calls=0, vae_frames=[], image_tokens=0, video_tokens=0)
    if opening is not None:
        if mode != 'ref2va' or path or identity_anchor or opening.get('kind') not in ('image', 'video'):
            raise ValueError('The opening takes one image or video reference, no saved latents and no identity image')
        return
    if path:
        z = torch.load(path, map_location='cpu', weights_only=True)['video']
        if z.ndim != 5 or z.shape[2] < TAIL_LATENTS or (z.shape[2] - TAIL_LATENTS) % 5 or not torch.isfinite(z).all():
            raise ValueError('Saved normalized latent has invalid geometry or nonfinite values')
        _state['tail'] = z[:, :, -TAIL_LATENTS:].float().contiguous()
    if mode == 'ref2va' and (_state['tail'] is None or not identity_anchor):
        raise ValueError('Ref2VA continuation requires seven saved latents and the identity image')


def audit():
    return _state['audit']


def align_positions(packed, count, frame_rows):
    """The identity image keeps its native slot; the guides take the target's first positions.

    Native Ref2VA advances the audio-video clock by one per reference. The seven guides are not
    references, so that advance is removed from the target video and all audio rows.
    """
    pos, tags, vi, ai, ti, ncv, nca = packed
    n = count * frame_rows
    if ncv <= n or nca != 0 or len(vi) - ncv < n:
        raise ValueError('Missing identity/tail/target rows')
    pos = pos.clone()
    pos[vi[ncv:], 0] -= count
    pos[ai, 0] -= count
    pos[vi[ncv - n:ncv]] = pos[vi[ncv:ncv + n]]
    return pos, tags, vi, ai, ti, ncv, nca


def install():
    """Patch the diffusers MiniMax-H3 Ref2VA blocks once per process."""
    global _installed
    if _installed:
        return
    from diffusers.modular_pipelines.minimax_h3 import before_denoise as bd
    from diffusers.modular_pipelines.minimax_h3 import before_encoder as be
    from diffusers.modular_pipelines.minimax_h3 import encoders as en
    original_setup = be.MiniMaxH3Ref2VASetupStep.__call__
    original_encode = en.MiniMaxH3Ref2VAReferenceEncoderStep.__call__
    original_vision = en.MiniMaxH3Ref2VATextEncoderStep._gather_vision_features
    original_vae = en.encode_vae_condition
    original_layout = bd.MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence

    def check_opening_reference(refs):
        if len(refs) != 1 or refs[0].kind != _state['opening']['kind'] or refs[0].has_audio:
            raise ValueError('The opening takes exactly one silent image or video reference')

    def setup(self, components, state):
        b = self.get_block_state(state)
        if _state['opening']:
            check_opening_reference(b.references)
            return original_setup(self, components, state)
        if len(b.references) != 1 or b.references[0].kind != 'image' or _state['tail'] is None:
            raise ValueError('Fixed identity workflow requires one image and seven saved tail latents')
        if tuple(_state['tail'].shape[-2:]) != (b.height // 16, b.width // 16):
            raise ValueError('Saved latent canvas differs')
        return original_setup(self, components, state)

    def encode(self, components, state):
        refs = state.values['normalized_references']
        opening = _state['opening']
        if opening:
            check_opening_reference(refs)
        elif len(refs) != 1 or refs[0].kind != 'image':
            raise ValueError('Unexpected identity input')
        components, state = original_encode(self, components, state)
        values = state.values
        if len(values['condition_latents']) != 1 or values['audio_condition_latents']:
            raise ValueError('Unexpected encoded references')
        common = {'reference_audio_rows': 0, 'vae_reference_encode_calls': _state['vae_calls'],
                  'vae_reference_frames': list(_state['vae_frames']),
                  'reference_latent_frames': [int(z.shape[2]) for z in values['condition_latents']],
                  'text_reference_image_tokens': _state['image_tokens'],
                  'text_reference_video_tokens': _state['video_tokens'],
                  'text_reference_vision_tokens': _state['image_tokens'] + _state['video_tokens'],
                  'target_position_alignment': False}
        if opening:
            _state['audit'] = dict(common, contract=OPENING_CONTRACT, latent_keyframes=0, source='opening_reference',
                                   rgb_reference_count=int(opening['kind'] == 'image'),
                                   reference_video_count=int(opening['kind'] == 'video'),
                                   reference_sha256=opening['sha256'])
            return components, state
        tail = _state['tail']
        values['condition_latents'] = list(values['condition_latents']) + [tail[:, :, k:k + 1] for k in range(TAIL_LATENTS)]
        # Layout-only descriptors for the guides, added after the text encoder: no pixels attached.
        values['normalized_references'] = list(refs) + [SimpleNamespace(kind='image', has_audio=False)
                                                        for _ in range(TAIL_LATENTS)]
        _state['audit'] = dict(common, contract=CONTRACT, latent_keyframes=TAIL_LATENTS,
                               source='previous_saved_normalized_latents', rgb_reference_count=1,
                               reference_video_count=0, identity_sha256=_state['identity_anchor']['sha256'])
        return components, state

    def gather_vision(self, *a, **kw):
        out = original_vision(self, *a, **kw)
        if _state['mode'] == 'ref2va':
            kind = _state['opening']['kind'] if _state['opening'] else 'image'
            if (len(out[1]), len(out[2])) != ((1, 0) if kind == 'image' else (0, 1)):
                raise ValueError(f'The text encoder must see exactly one {kind} reference')
            _state['image_tokens'] = sum(out[1])
            _state['video_tokens'] = sum(count * len(times) for count, times in zip(out[2], out[3]))
        return out

    def vae_guard(*a, **kw):
        if _state['mode'] == 'ref2va':
            _state['vae_calls'] += 1
            frames = int(a[1].shape[2])
            _state['vae_frames'].append(frames)
            still = not _state['opening'] or _state['opening']['kind'] == 'image'
            if _state['vae_calls'] != 1 or (frames == 1) != still:
                raise ValueError('Only the one reference image or opening video may be VAE-encoded')
        return original_vae(*a, **kw)

    def layout(*a, **kw):
        if _state['opening']:
            out = original_layout(*a, **kw)
            _state['audit'].update(
                boundary_rows=0, all_condition_video_rows=out[5], reference_audio_rows=out[6],
                reference_clock_advance_removed=0, target_video_origin=float(out[0][out[2][out[5]], 0]),
                target_audio_origin=float(out[0][out[3][out[6]], 0]), text_rows=len(out[4]))
            return out
        z = _state['tail']
        rows = (z.shape[-2] // 2) * (z.shape[-1] // 2)
        out = align_positions(original_layout(*a, **kw), TAIL_LATENTS, rows)
        _state['audit'].update(
            target_position_alignment=True, boundary_rows=TAIL_LATENTS * rows,
            all_condition_video_rows=out[5], reference_audio_rows=out[6],
            reference_clock_advance_removed=TAIL_LATENTS,
            target_video_origin=float(out[0][out[2][out[5]], 0]), text_rows=len(out[4]))
        return out

    be.MiniMaxH3Ref2VASetupStep.__call__ = setup
    en.MiniMaxH3Ref2VAReferenceEncoderStep.__call__ = encode
    en.MiniMaxH3Ref2VATextEncoderStep._gather_vision_features = gather_vision
    en.encode_vae_condition = vae_guard
    bd.MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence = staticmethod(layout)
    _installed = True
