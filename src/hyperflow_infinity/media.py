"""Frame-exact trimming, evidence stills and film assembly (FFmpeg / PyAV)."""
import json
import subprocess
import wave
from pathlib import Path

from .store import require


def probe(path):
    cmd = ['ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-show_format', '-of', 'json', str(path)]
    d = json.loads(subprocess.check_output(cmd))
    v = next(x for x in d['streams'] if x['codec_type'] == 'video')
    audio = next((x for x in d['streams'] if x['codec_type'] == 'audio'), None)
    rate = v.get('avg_frame_rate', '24/1').split('/')
    return {'frames': int(v.get('nb_read_frames') or v['nb_frames']),
            'fps': float(rate[0]) / float(rate[1]), 'width': v['width'], 'height': v['height'],
            'audio': audio is not None, 'video_seconds': float(v.get('duration', 0)),
            'audio_seconds': float(audio.get('duration', 0)) if audio else 0}


def edit_chunk(raw, dest, overlap, keep, fps=24):
    """Drop the `overlap` frames that redraw the previous segment and keep exactly `keep` frames."""
    end = overlap + keep
    vf = f'trim=start_frame={overlap}:end_frame={end},setpts=PTS-STARTPTS'
    af = f'atrim=start={overlap / fps:.9f}:end={end / fps:.9f},asetpts=PTS-STARTPTS,apad,atrim=duration={keep / fps:.9f}'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(raw), '-vf', vf, '-af', af,
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac', '-ar', '32000', '-ac', '2', '-movflags', '+faststart', str(dest)], check=True)
    actual = probe(dest)
    require(actual['frames'] == keep, f"Edited video has {actual['frames']}, expected {keep}")
    require(actual['audio'], 'Edited chunk lost audio')


def evidence_frames(raw, out_dir, chunk_id, first, count, fps=24, fractions=(0, .5, 1)):
    """Save the first, middle and last kept frames as PNGs."""
    import av
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wanted = sorted(set(first + round((count - 1) * f) for f in fractions))
    records = []
    with av.open(str(raw)) as container:
        for n, frame in enumerate(container.decode(video=0)):
            if n in wanted:
                eid = f'{chunk_id}_f{n:05d}'
                path = out / f'{eid}.png'
                frame.to_image().convert('RGB').save(path)
                records.append({'id': eid, 'chunk_id': chunk_id, 'frame_index': n, 'source_seconds': n / fps,
                                'path': str(path.resolve()), 'source_video': str(Path(raw).resolve())})
            if n >= wanted[-1]:
                break
    require(len(records) == len(wanted), 'Failed to extract evidence coverage')
    return records


def assemble_audio(chunks, dest, fps=24):
    """Cut the original PCM by video frames, so per-file AAC padding never accumulates."""
    cursor = 0
    with wave.open(str(dest), 'wb') as out:
        rate = channels = width = None
        for chunk in chunks:
            with wave.open(chunk['raw_audio'], 'rb') as source:
                params = (source.getframerate(), source.getnchannels(), source.getsampwidth())
                if rate is None:
                    rate, channels, width = params
                    out.setframerate(rate)
                    out.setnchannels(channels)
                    out.setsampwidth(width)
                require(params == (rate, channels, width), 'Inconsistent source audio format')
                start = round(chunk['overlap_frames'] * rate / fps)
                count = round((cursor + chunk['kept_frames']) * rate / fps) - round(cursor * rate / fps)
                source.setpos(min(start, source.getnframes()))
                data = source.readframes(count)
                data += b'\x00' * (count * channels * width - len(data))
                out.writeframesraw(data)
                cursor += chunk['kept_frames']
    return cursor


def assemble(root, chunks, target_frames, fps=24):
    root = Path(root)
    soundtrack = root / 'film_audio.wav'
    require(assemble_audio(chunks, soundtrack, fps) == target_frames, 'Audio timeline does not match film frames')
    listing = root / 'assembly_inputs.txt'
    listing.write_text(''.join("file '" + str(Path(x['edited_video']).resolve()).replace("'", "'\\''") + "'\n"
                               for x in chunks))
    dest = root / 'film.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(listing),
                    '-i', str(soundtrack), '-map', '0:v:0', '-map', '1:a:0', '-vf', f'setpts=N/({fps}*TB)',
                    '-r', str(fps), '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                    '-c:a', 'aac', '-ar', '32000', '-ac', '2', '-movflags', '+faststart', str(dest)], check=True)
    details = probe(dest)
    require(details['frames'] == target_frames, 'Final film does not match exact target frame count')
    require(details['audio'], 'Final film has no audio')
    # Some MP4 muxers expose millisecond-resolution edit lists.
    require(abs(details['audio_seconds'] - target_frames / fps) < .001 + 2 / 32000,
            'Final audio duration drifted from the film clock')
    return dest, details
