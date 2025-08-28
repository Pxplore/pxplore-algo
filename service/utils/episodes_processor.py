import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

BLOOM_MAP: Dict[str, int] = {
    'Recalling_Fact': 1,
    'Paraphrasing_Concept': 2,
    'Applying_Rule': 3,
    'Analyzing_Relations': 4,
    'Evaluating_Claim': 5,
    'Synthesizing_Idea': 6,
}

THREAD_FLOW_ADJ: Dict[str, float] = {
    'Sustained_High-Level': 0.5,
    'Deepening_Progression': 0.3,
    'Exploratory_Fluctuation': 0.0,
    'Static_Low-Level': -0.3,
}

# Transit segment policy for review threads (short dwell, no ops pages)
TRANSIT_MAX_SEC: int = 2              # dwell <= 2s regarded as transit by default
TRANSIT_REQUIRE_NO_MSG: bool = True   # require no student messages in the segment
TRANSIT_REQUIRE_NO_PAUSE: bool = True # require pause_count == 0 (or missing)
TRANSIT_POLICY: str = 'drop'          # 'drop' to skip creating episode; 'mark' to include but exclude from smoothing/level


def load_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def build_indices(data: Dict[str, Any]):
    discussion_threads = data.get('discussion_threads', [])
    msg_by_id: Dict[str, Dict[str, Any]] = {}
    msgs_by_page: Dict[int, List[Dict[str, Any]]] = {}
    thread_flow_by_id: Dict[str, Optional[str]] = {}
    threads_by_page: Dict[int, List[Dict[str, Any]]] = {}

    for th in discussion_threads:
        page = th.get('page_number')
        threads_by_page.setdefault(page, []).append(th)
        ta = th.get('thread_analysis')
        if ta:
            thread_flow_by_id[th['thread_id']] = ta.get('cognitive_flow')
        for m in th.get('messages', []):
            msg_by_id[m['message_id']] = m
            msgs_by_page.setdefault(page, []).append(m)

    quiz_by_page: Dict[int, List[Dict[str, Any]]] = {}
    for q in data.get('quizzes', []):
        quiz_by_page.setdefault(q.get('page_number_of_quiz'), []).append(q)

    # page_interactions_analysis index by (page_number, slice_id)
    pi_analysis: Dict[Tuple[int, Optional[str]], Dict[str, Any]] = {}
    for pae in data.get('page_interactions_analysis', []):
        pi_analysis[(pae.get('page_number'), pae.get('slice_id'))] = pae

    return msg_by_id, msgs_by_page, thread_flow_by_id, threads_by_page, quiz_by_page, pi_analysis


def compute_linguistic_level(msgs: List[Dict[str, Any]]):
    vals: List[int] = []
    flags = {'reflect': False, 'misconception': False}
    for m in msgs:
        if m.get('author_type') != 'student':
            continue
        ma = m.get('model_analysis')
        if not ma:
            continue
        for ct in ma.get('cognitive_type', []):
            if ct in BLOOM_MAP:
                vals.append(BLOOM_MAP[ct])
            elif ct == 'Reflecting_Metacognitively':
                flags['reflect'] = True
            elif ct == 'Identifying_Misconception':
                flags['misconception'] = True
    if not vals:
        return None, flags
    try:
        mode = statistics.mode(vals)
        tie = False
    except statistics.StatisticsError:
        modes = statistics.multimode(vals)
        mode = max(modes)
        tie = True
    level = float(mode)
    if tie:
        level = min(6.0, level + 0.5)
    return level, flags


def thread_flow_adjust(page_number: int, threads_by_page: Dict[int, List[Dict[str, Any]]], thread_flow_by_id: Dict[str, Optional[str]]):
    adj = 0.0
    for th in threads_by_page.get(page_number, []):
        flow = thread_flow_by_id.get(th['thread_id'])
        if flow in THREAD_FLOW_ADJ:
            adj = max(adj, THREAD_FLOW_ADJ[flow])
    return adj


def behavior_adjust(page_number: int, slice_id: Optional[str], pi_analysis: Dict[Tuple[int, Optional[str]], Dict[str, Any]]):
    adj = 0.0
    pae = pi_analysis.get((page_number, slice_id))
    if not pae:
        # fallback by page number only
        for (pnum, _sid), v in pi_analysis.items():
            if pnum == page_number:
                pae = v
                break
    if pae:
        state = pae.get('inferred_learning_state')
        if state in ('Focused Engagement', 'Productive Struggle'):
            adj += 0.2
    return adj


def quiz_adj_for_page(page_number: int, data: Dict[str, Any], quiz_by_page: Dict[int, List[Dict[str, Any]]]):
    qs = quiz_by_page.get(page_number, [])
    adj = 0.0
    penalty = 0.0
    for q in qs:
        if q.get('is_correct'):
            adj = max(adj, 0.2)
        else:
            for qa in data.get('quizzes_analysis', []):
                if qa.get('question_id') == q.get('question_id') and qa.get('is_correct') is False:
                    dec = qa.get('diagnostic_evidence_chain') or {}
                    if dec.get('claim_misconception_id'):
                        penalty = min(-0.5, penalty)
    return adj + penalty


# --- Timeline helpers ---

def synth_page_ts(data: Dict[str, Any]) -> Dict[int, Tuple[Optional[str], Optional[str]]]:
    """Return per-page synthetic (entry, exit) if missing, using meta.start_time and cumulative dwell."""
    meta = data.get('meta', {})
    start_time = meta.get('start_time')
    pages = data.get('page_interactions', [])
    out: Dict[int, Tuple[Optional[str], Optional[str]]] = {}
    if not start_time:
        return out
    # Use cumulative dwell to assign first_entry if needed
    from datetime import datetime, timedelta, timezone
    def parse_iso(s: str) -> datetime:
        return datetime.fromisoformat(s.replace('Z', '+00:00'))
    def fmt_iso(dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    cur = parse_iso(start_time)
    for pi in pages:
        pnum = pi.get('page_number')
        dwell = int(pi.get('time_spent_sec') or 0)
        ent = pi.get('entry_ts') or pi.get('first_entry_ts') or fmt_iso(cur)
        exi = pi.get('exit_ts') or fmt_iso(cur + timedelta(seconds=dwell))
        out[pnum] = (ent, exi)
        # advance only by dwell if original entry/exit missing; if present, advance by dwell too to keep monotone
        cur = cur + timedelta(seconds=dwell)
    return out


def synth_review_seg_ts(rt: Dict[str, Any]) -> Dict[int, Tuple[Optional[str], Optional[str]]]:
    """Return per-segment synthetic (entry, exit) mapping by seq, based on anchor.departure_ts and time_spent."""
    from datetime import datetime, timedelta, timezone
    def parse_iso(s: str) -> datetime:
        return datetime.fromisoformat(s.replace('Z', '+00:00'))
    def fmt_iso(dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    res: Dict[int, Tuple[Optional[str], Optional[str]]] = {}
    dep = ((rt.get('anchor') or {}).get('departure_ts'))
    if not dep:
        return res
    cur = parse_iso(dep)
    for seg in sorted(rt.get('segments', []), key=lambda s: s.get('seq') or 0):
        dwell = int(seg.get('time_spent_sec') or 0)
        ent = seg.get('entry_ts') or fmt_iso(cur)
        exi = seg.get('exit_ts') or fmt_iso(cur + timedelta(seconds=dwell))
        res[seg.get('seq') or 0] = (ent, exi)
        cur = parse_iso(exi)
    return res


def is_transit_segment(seg: Dict[str, Any], msgs: List[Dict[str, Any]]) -> bool:
    dwell = seg.get('time_spent_sec') or 0
    if dwell > TRANSIT_MAX_SEC:
        return False
    if TRANSIT_REQUIRE_NO_MSG and msgs:
        return False
    if TRANSIT_REQUIRE_NO_PAUSE and (seg.get('pause_count') or 0) != 0:
        return False
    return True


def build_episodes(data: Dict[str, Any]) -> Dict[str, Any]:
    page_interactions = data.get('page_interactions', [])
    review_threads = data.get('review_threads', [])

    msg_by_id, msgs_by_page, thread_flow_by_id, threads_by_page, quiz_by_page, pi_analysis = build_indices(data)

    # Precompute synthetic timestamps for pages (if missing) and reviews
    page_ts_map = synth_page_ts(data)  # page_number -> (entry, exit)
    review_seg_ts_map: Dict[str, Dict[int, Tuple[Optional[str], Optional[str]]]] = {}
    for rt in review_threads:
        review_seg_ts_map[rt.get('review_thread_id')] = synth_review_seg_ts(rt)

    # collect review message ids per page to avoid double counting on page episodes
    review_msg_ids_by_page: Dict[int, set] = {}
    for rt in review_threads:
        for seg in rt.get('segments', []):
            page = seg.get('page_number')
            mids = seg.get('interactions', {}).get('message_ids', [])
            if mids:
                review_msg_ids_by_page.setdefault(page, set()).update(mids)

    episodes: List[Dict[str, Any]] = []

    # page episodes
    for pi in page_interactions:
        page = pi.get('page_number')
        slice_id = pi.get('slice_id')
        dwell = pi.get('time_spent_sec')
        pause = pi.get('pause_count')
        excluded = review_msg_ids_by_page.get(page, set())
        base_msgs = [m for m in msgs_by_page.get(page, []) if m.get('message_id') not in excluded]
        ling, _flags = compute_linguistic_level(base_msgs)
        t_adj = thread_flow_adjust(page, threads_by_page, thread_flow_by_id)
        b_adj = behavior_adjust(page, slice_id, pi_analysis)
        q_adj = quiz_adj_for_page(page, data, quiz_by_page)
        if ling is not None:
            final = 0.8 * ling + min(0.2, t_adj + b_adj + q_adj)
        else:
            final = 2.0 + min(0.3, b_adj + q_adj + max(0.0, t_adj))
        # build linguistic evidence list first
        ling_evid = [
            {
                'msg_id': m['message_id'],
                'author_type': m.get('author_type'),
                'cognitive_type': (m.get('model_analysis') or {}).get('cognitive_type', [])
            } for m in base_msgs if m.get('author_type') == 'student'
        ]
        # determine sort_ts from actual or synthetic
        ent_ts = pi.get('entry_ts') or pi.get('first_entry_ts') or (page_ts_map.get(page) or (None, None))[0]
        signals = {
            'linguistic_evidence': ling_evid,
            'behavioral_evidence': [pi_analysis.get((page, slice_id), {})],
            'thread_evidence': [
                {'thread_id': th['thread_id'], 'cognitive_flow': thread_flow_by_id.get(th['thread_id'])}
                for th in threads_by_page.get(page, [])
            ],
            'quiz_evidence': quiz_by_page.get(page, [])
        }
        if ling is None:
            signals['no_linguistic_evidence'] = True
        episodes.append({
            'episode_idx': None,
            'kind': 'page',
            'page_number': page,
            'start_ts': pi.get('entry_ts'),
            'end_ts': pi.get('exit_ts'),
            'dwell_sec': dwell,
            'pause_count': pause,
            'loop_id': None,
            'raw_linguistic_level': ling,
            'behavior_adjust': round(b_adj, 3),
            'thread_flow_adjust': round(t_adj, 3),
            'quiz_adjust': round(q_adj, 3),
            'final_episode_level': round(final, 3),
            'signals': signals,
            'sort_ts': ent_ts
        })

    # review episodes
    for rt in review_threads:
        loop_id = rt.get('review_thread_id')
        seg_ts_map = review_seg_ts_map.get(loop_id, {})
        for seg in rt.get('segments', []):
            page = seg.get('page_number')
            msg_ids = seg.get('interactions', {}).get('message_ids', [])
            msgs = [msg_by_id[mid] for mid in msg_ids if mid in msg_by_id]
            # transit detection
            transit = is_transit_segment(seg, msgs)
            if transit and TRANSIT_POLICY == 'drop':
                # skip creating episode
                continue
            ling, _flags = compute_linguistic_level(msgs)
            t_adj = thread_flow_adjust(page, threads_by_page, thread_flow_by_id)
            b_adj = behavior_adjust(page, None, pi_analysis)
            q_adj = quiz_adj_for_page(page, data, quiz_by_page)
            if ling is not None:
                final = 0.8 * ling + min(0.2, t_adj + b_adj + q_adj)
            else:
                final = 2.0 + min(0.3, b_adj + q_adj + max(0.0, t_adj))
            ling_evid = [
                {
                    'msg_id': m['message_id'],
                    'author_type': m.get('author_type'),
                    'cognitive_type': (m.get('model_analysis') or {}).get('cognitive_type', [])
                } for m in msgs if m.get('author_type') == 'student'
            ]
            # sort_ts from actual seg.entry_ts or synthesized by seq
            ent_ts, exi_ts = seg.get('entry_ts'), seg.get('exit_ts')
            if not ent_ts:
                ent_ts = (seg_ts_map.get(seg.get('seq') or 0) or (None, None))[0]
            signals = {
                'linguistic_evidence': ling_evid,
                'behavioral_evidence': [pi_analysis.get((page, None), {})],
                'thread_evidence': [
                    {'thread_id': th['thread_id'], 'cognitive_flow': thread_flow_by_id.get(th['thread_id'])}
                    for th in threads_by_page.get(page, [])
                ],
                'quiz_evidence': quiz_by_page.get(page, [])
            }
            if ling is None:
                signals['no_linguistic_evidence'] = True
            ep_obj = {
                'episode_idx': None,
                'kind': 'review',
                'page_number': page,
                'start_ts': seg.get('entry_ts'),
                'end_ts': seg.get('exit_ts'),
                'dwell_sec': seg.get('time_spent_sec'),
                'pause_count': seg.get('pause_count'),
                'loop_id': loop_id,
                'raw_linguistic_level': ling,
                'behavior_adjust': round(b_adj, 3),
                'thread_flow_adjust': round(t_adj, 3),
                'quiz_adjust': round(q_adj, 3),
                'final_episode_level': round(final, 3) if TRANSIT_POLICY != 'mark' else (None if transit else round(final, 3)),
                'signals': signals,
                'sort_ts': ent_ts
            }
            if transit:
                ep_obj['signals']['transit'] = True
                if TRANSIT_POLICY == 'mark':
                    ep_obj['signals']['excluded_from_smoothing'] = True
            episodes.append(ep_obj)

    # sort episodes: timeline-first using sort_ts; fallback by page order index
    page_order_index: Dict[int, int] = {pi.get('page_number'): i for i, pi in enumerate(page_interactions)}

    def sort_key(ep: Dict[str, Any]):
        ts = ep.get('sort_ts') or ep.get('start_ts') or ''
        if ts:
            return (0, ts)
        # fallback by page order
        idx = page_order_index.get(ep.get('page_number'), 9999)
        off = 0.1 if ep['kind'] == 'review' else 0.0
        return (1, idx + off)

    episodes.sort(key=sort_key)

    # assign indices
    for i, ep in enumerate(episodes):
        ep['episode_idx'] = i
        # cleanup auxiliary key
        if 'sort_ts' in ep:
            del ep['sort_ts']

    # smoothing: median (window=3) then EWA
    levels: List[Optional[float]] = [ep.get('final_episode_level') for ep in episodes]

    # median on numeric-only neighbors (window=3 in episode order)
    med_levels: List[Optional[float]] = []
    for i in range(len(levels)):
        win_vals: List[float] = []
        for j in range(max(0, i-1), min(len(levels), i+2)):
            v = levels[j]
            if isinstance(v, (int, float)):
                win_vals.append(float(v))
        med_levels.append(statistics.median(win_vals) if win_vals else None)

    # EWA on numeric medians only, preserving positions of None
    alpha = 0.5 if sum(1 for v in med_levels if v is not None) <= 5 else 0.6
    smoothed_seq: List[Optional[float]] = [None] * len(med_levels)
    prev: Optional[float] = None
    for i, v in enumerate(med_levels):
        if v is None:
            continue
        if prev is None:
            prev = v
        else:
            prev = alpha * v + (1 - alpha) * prev
        smoothed_seq[i] = prev

    for i, ep in enumerate(episodes):
        ep['smoothed_level'] = round(smoothed_seq[i], 3) if smoothed_seq[i] is not None else None

    return {
        'episodes': episodes,
        'config_used': {
            'ordering': 'timeline',
            'linguistic_weight': 0.8,
            'aux_caps_with_linguistic': 0.2,
            'aux_caps_without_linguistic': 0.3,
            'median_window': 3,
            'ewa_alpha': alpha,
            'page_first_when_no_ts': False,
            'transit': {
                'max_sec': TRANSIT_MAX_SEC,
                'require_no_messages': TRANSIT_REQUIRE_NO_MSG,
                'require_no_pause': TRANSIT_REQUIRE_NO_PAUSE,
                'policy': TRANSIT_POLICY
            }
        }
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', '-i', type=Path, default=Path('输入.txt'))
    ap.add_argument('--output', '-o', type=Path, default=Path('工程预处理输出.txt'))
    args = ap.parse_args()

    data = load_json(args.input)
    meta = data.get('meta', {})
    proc = build_episodes(data)
    out = {
        'meta': meta,
        'processed_analysis': proc,
    }
    with args.output.open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'Wrote {args.output} with {len(proc["episodes"])} episodes')


if __name__ == '__main__':
    main() 

    