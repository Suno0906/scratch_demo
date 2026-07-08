"""
긁기 강도 오분류 진단 연구 도구 (Streamlit, Rule-based)

방식: 신호 특징의 '약 기준'·'강 기준' 두 앵커 → 새 긁기의 위치(0~1) → 경계로 약/보통/강.
머신러닝 없음(임계값 기반). 검증은 LOOCV/LOSO로 학습·평가 샘플이 겹치지 않게 한다.

핵심 목적: 숫자 정확도가 아니라 "왜 틀렸나"를 눈으로 진단.
 → 오분류 긁기의 스펙트로그램을, 같은 실제 라벨의 정분류 긁기와 나란히(동일 색 스케일) 비교.

실행: pip install -r requirements.txt && streamlit run app.py
"""
import json, time
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal as sps
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, GroupKFold

np.random.seed(0)  # 재현성(현재 파이프라인엔 난수 없음, 원칙상 고정)

FS = 100
DATA_PATH = "sample_data.ndjson"
KR = {'weak': '약', 'normal': '보통', 'strong': '강'}
COLOR = {'weak': '#10B981', 'normal': '#F59E0B', 'strong': '#EF4444'}
ACCENT = '#6366F1'
GRAY = '#94A3B8'
LABELS = ['약', '보통', '강']
LAB_MAP = {'약': 0, '보통': 1, '강': 2}
INT3 = {'weak': 0, 'normal': 1, 'strong': 2}
FEATS_ALL = {'std': '진폭(std)', 'jerk': '저크', 'spectral_centroid': '스펙트럼 중심(Hz)',
             'gyro_std': '자이로(std)'}  # energy 제거(=std² 중복), peak_freq→spectral_centroid(안정적)
META = {'subject_id': '사람', 'body_region': '부위', 'method': '방식', 'posture': '자세'}
VAL_KR = {'head_neck': '머리·목', 'arm': '팔', 'torso': '몸통', 'leg': '다리',
          'finger': '손가락', 'supine': '누움', 'side': '옆으로', 'prone': '엎드림', 'sitting': '앉음'}
WIN_KR = {'std': '진폭(std)', 'rms': 'RMS', 'energy': '에너지', 'jerk': '저크',
          'p2p': '범위(P2P)', 'centroid': '주파수 중심(Hz)', 'hf_ratio': '고주파 비율(>10Hz)'}


def vk(v):
    return VAL_KR.get(v, v)


def _cut(a, k):
    a = np.asarray(a, float)
    return a[:len(a) - k] if 0 < k < len(a) else a


def _axes(sig, d, trim_sec):
    x, y, z = sig['x'], sig['y'], sig['z']
    if trim_sec > 0:
        k = round(trim_sec * est_fs(d))
        x, y, z = _cut(x, k), _cut(y, k), _cut(z, k)
    return np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)


def amag(d, trim_sec=0.0):
    x, y, z = _axes(d['accel'], d, trim_sec)
    return np.sqrt(x**2 + y**2 + z**2)


def gmag(d, trim_sec=0.0):
    g = d.get('gyro')
    if not g:
        return None
    x, y, z = _axes(g, d, trim_sec)
    return np.sqrt(x**2 + y**2 + z**2)


def spectral_centroid(sig, fs):
    """Welch PSD 무게중심(Hz). 피크(argmax)보다 안정적."""
    sig = np.asarray(sig, float) - np.mean(sig)
    wf, wp = sps.welch(sig, fs=fs, nperseg=min(256, max(8, len(sig))))
    return float((wf * wp).sum() / (wp.sum() + 1e-12))


def features(d, trim_sec=0.0):
    m = amag(d, trim_sec); fs = est_fs(d)
    g = gmag(d, trim_sec)
    return {'std': float(m.std()),
            'jerk': float(np.abs(np.diff(m)).mean()),
            'spectral_centroid': spectral_centroid(m, fs),
            'gyro_std': float(g.std()) if g is not None else np.nan}


def seg_feat(seg, fs, key):
    """단일 세그먼트에서 특징 하나 계산 (윈도우별 시간 변화용)."""
    if len(seg) < 2:
        return 0.0
    m0 = seg - seg.mean()
    if key == 'std':
        return float(seg.std())
    if key == 'rms':
        return float(np.sqrt((m0**2).mean()))
    if key == 'energy':
        return float((m0**2).sum())
    if key == 'jerk':
        return float(np.abs(np.diff(seg)).mean())
    if key == 'p2p':
        return float(seg.max() - seg.min())
    fr = np.fft.rfftfreq(len(m0), 1 / fs); ps = np.abs(np.fft.rfft(m0))**2; tot = ps.sum() + 1e-12
    if key == 'centroid':
        return float((fr * ps).sum() / tot)
    if key == 'hf_ratio':
        return float(ps[fr > 10].sum() / tot)
    if key == 'peak_freq':
        return float(fr[ps.argmax()])
    return float('nan')


def windowed(m, fs, key, win_sec=1.0, hop_sec=0.25):
    """신호를 창으로 훑으며 특징의 시간 변화를 계산 → (창 중심 시간, 값)."""
    w = min(len(m), max(8, int(win_sec * fs))); h = max(1, int(hop_sec * fs))
    ts, vals = [], []
    for s in range(0, len(m) - w + 1, h):
        ts.append((s + w / 2) / fs); vals.append(seg_feat(m[s:s + w], fs, key))
    return np.array(ts), np.array(vals)


def describe_features(d, trim_sec=0.0):
    """이 샘플의 분석용 특징을 도메인별로 전부 계산 → [(도메인, 이름, key, 값), ...]."""
    m = amag(d, trim_sec); fs = est_fs(d); m0 = m - m.mean()
    fr = np.fft.rfftfreq(len(m0), 1 / fs); ps = np.abs(np.fft.rfft(m0))**2; tot = ps.sum() + 1e-12
    cen = float((fr * ps).sum() / tot)
    zcr = float((np.diff(np.sign(m0)) != 0).sum()) / len(m0)
    rows = [('시간', '평균', 'mean', float(m.mean())),
            ('시간', '표준편차(std)', 'std', float(m.std())),
            ('시간', 'RMS', 'rms', float(np.sqrt((m0**2).mean()))),
            ('시간', '중앙절대편차(MAD)', 'mad', float(np.median(np.abs(m - np.median(m))))),
            ('시간', '범위(P2P)', 'p2p', float(m.max() - m.min())),
            ('시간', '영교차율', 'zcr', zcr),
            ('동역학', '저크 평균', 'jerk', float(np.abs(np.diff(m)).mean())),
            ('동역학', '저크 표준편차', 'jerk_std', float(np.abs(np.diff(m)).std())),
            ('주파수', '주파수 피크(Hz)', 'peak_freq', float(fr[ps.argmax()])),
            ('주파수', '스펙트럴 중심(Hz)', 'centroid', cen),
            ('주파수', '스펙트럴 대역폭', 'bandwidth', float(np.sqrt(((fr - cen)**2 * ps).sum() / tot))),
            ('주파수', '고주파비율(>10Hz)', 'hf_ratio', float(ps[fr > 10].sum() / tot)),
            ('주파수', '총 에너지', 'energy', float((m0**2).sum()))]
    g = gmag(d, trim_sec)
    if g is not None:
        g0 = g - g.mean(); gfr = np.fft.rfftfreq(len(g0), 1 / fs); gps = np.abs(np.fft.rfft(g0))**2
        rows += [('자이로', '자이로 std', 'gyro_std', float(g.std())),
                 ('자이로', '자이로 RMS', 'gyro_rms', float(np.sqrt((g0**2).mean()))),
                 ('자이로', '자이로 에너지', 'gyro_energy', float((g0**2).sum())),
                 ('자이로', '자이로 주파수 피크', 'gyro_peak_freq', float(gfr[gps.argmax()]))]
    return rows


# ── 긁기 감지(RandomForest) + 가상 연속 시계열 스트리밍 헬퍼 ──────
GRAVITY = 9.81  # 무동작 구간 가속도 크기(중력)


def dom_freq(seg, fs):
    s = np.asarray(seg, float) - np.mean(seg)
    ps = np.abs(np.fft.rfft(s))**2; fr = np.fft.rfftfreq(len(s), 1 / fs)
    return float(fr[ps.argmax()]) if ps.sum() > 0 else 0.0


def autocorr_peak(seg):
    s = np.asarray(seg, float) - np.mean(seg)
    if s.std() == 0:
        return 0.0
    ac = np.correlate(s, s, mode='full')[len(s) - 1:]; ac = ac / (ac[0] + 1e-12)
    lo_, hi_ = 3, max(4, len(ac) // 2)
    return float(ac[lo_:hi_].max()) if hi_ > lo_ else 0.0


def det_features(segA, segG, fs):
    """긁기 감지용 12 특징: 가속도·자이로 크기 각각 std·MAD·범위·저크·지배주파수·자기상관피크."""
    out = []
    for s in (np.asarray(segA, float), np.asarray(segG, float)):
        out += [float(s.std()), float(np.median(np.abs(s - np.median(s)))), float(s.max() - s.min()),
                float(np.abs(np.diff(s)).mean()) if len(s) > 1 else 0.0,
                dom_freq(s, fs), autocorr_peak(s)]
    return np.array(out, float)


def win_strength_feats(segA, segG, fs):
    """윈도우의 강도 분류 특징 (FEATS_ALL과 동일 키)."""
    return {'std': float(np.std(segA)),
            'jerk': float(np.abs(np.diff(segA)).mean()) if len(segA) > 1 else 0.0,
            'spectral_centroid': spectral_centroid(segA, fs),
            'gyro_std': float(np.std(segG))}


@st.cache_resource(show_spinner=False)
def train_detector(_train_data, fs, trim, sig):
    """학습 데이터의 윈도우로 RandomForest 학습 + 동작 단위 그룹 CV 정확도."""
    X, y, grp = [], [], []
    w = int(fs); h = max(1, int(0.25 * fs))
    for i, d in enumerate(_train_data):
        a = amag(d, trim); gy = gmag(d, trim); gy = np.zeros_like(a) if gy is None else gy
        lab = d.get('label', 'SCRATCH')
        for s in range(0, len(a) - w + 1, h):
            X.append(det_features(a[s:s + w], gy[s:s + w], fs)); y.append(lab); grp.append(i)
    X = np.array(X); y = np.array(y); grp = np.array(grp)
    rf = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
    cvacc, ncv = None, 0
    if len(set(y)) >= 2 and len(set(grp)) >= 2:
        try:
            pred = cross_val_predict(rf, X, y, cv=GroupKFold(min(5, len(set(grp)))), groups=grp)
            cvacc, ncv = float((pred == y).mean()), len(y)
        except Exception:
            cvacc = None
    if len(set(y)) >= 2:
        rf.fit(X, y)
    return rf, cvacc, ncv, sorted(set(y)), len(X)


def build_stream(eval_data, fs, trim, gap_sec=0.5):
    """평가 샘플을 시간 순으로 이어붙인 가상 연속 신호 + 시점별 정답(라벨/강도/출처)."""
    ma, mg, s_lab, s_int = [], [], [], []
    gap = int(gap_sec * fs)
    for d in eval_data:
        a = amag(d, trim); gy = gmag(d, trim); gy = np.zeros_like(a) if gy is None else gy
        lab = d.get('label', 'SCRATCH'); it = d.get('intensity') if lab == 'SCRATCH' else None
        ma.append(a); mg.append(gy); s_lab += [lab] * len(a); s_int += [it] * len(a)
        ma.append(np.full(gap, GRAVITY)); mg.append(np.zeros(gap)); s_lab += ['GAP'] * gap; s_int += [None] * gap
    return np.concatenate(ma), np.concatenate(mg), np.array(s_lab, object), np.array(s_int, object)


def stream_windows(ma, mg, s_lab, s_int, fs, win_sec=1.0, hop_sec=0.25):
    """가상 연속 신호를 1초 창/0.25초 홉으로 훑으며 창별 정답(다수결)을 매긴다."""
    w = int(win_sec * fs); h = max(1, int(hop_sec * fs)); out = []
    for s in range(0, len(ma) - w + 1, h):
        labs = list(s_lab[s:s + w]); ints = [x for x in s_int[s:s + w] if x is not None]
        out.append({'start': s, 'tc': (s + w / 2) / fs,
                    'gt_label': max(set(labs), key=labs.count),
                    'gt_int': (max(set(ints), key=ints.count) if ints else None)})
    return out


def sample_strength_series(d, trim, fs, smodel, win_sec=1.0, hop_sec=0.25):
    """한 샘플을 윈도우로 훑으며 강도 위치점수·예측을 계산 → (창중심시간, 위치점수, 예측클래스)."""
    m = amag(d, trim); mg = gmag(d, trim); mg = np.zeros_like(m) if mg is None else mg
    w = min(len(m), max(8, int(win_sec * fs))); h = max(1, int(hop_sec * fs))
    starts = list(range(0, len(m) - w + 1, h)) or [0]
    feats = [win_strength_feats(m[s:s + w], mg[s:s + w], fs) for s in starts]
    posv, pred = apply_model(smodel, pd.DataFrame(feats))
    times = np.array([(s + w / 2) / fs for s in starts])
    return times, np.asarray(posv, float), np.asarray(pred)


def flow_fig(times, posv, pred, true_lab, lo, hi, pos):
    """강도 흐름 차트: 위치점수 시간곡선 + 약/보통/강 배경존 + 예측 리본. ○=정답 일치·✕=불일치."""
    n = len(times)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.05)
    for y0, y1, cl in [(0, lo, 'weak'), (lo, hi, 'normal'), (hi, 1, 'strong')]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=COLOR[cl], opacity=0.10, line_width=0, row=1, col=1)
    for yb in (lo, hi):
        fig.add_hline(y=yb, line_dash='dot', line_color='gray', opacity=0.6, row=1, col=1)
    op = [1.0 if i < pos else 0.15 for i in range(n)]
    fig.add_trace(go.Scatter(
        x=times, y=posv, mode='lines+markers', line=dict(color='rgba(130,130,130,0.5)', width=1),
        marker=dict(color=[COLOR[ENG[p]] for p in pred],
                    symbol=['circle' if p == true_lab else 'x' for p in pred],
                    size=[9 if p == true_lab else 12 for p in pred], opacity=op,
                    line=dict(width=1, color='#111')),
        customdata=['일치' if p == true_lab else '불일치' for p in pred],
        hovertemplate='%{x:.2f}s · 위치 %{y:.2f} · %{customdata}<extra></extra>'), row=1, col=1)
    cidx = {'약': 0, '보통': 1, '강': 2}
    z = [[(cidx[p] + 0.5) if i < pos else None for i, p in enumerate(pred)]]
    cs = []
    for i, cl in enumerate(['weak', 'normal', 'strong']):
        cs += [[i / 3, COLOR[cl]], [(i + 1) / 3, COLOR[cl]]]
    fig.add_trace(go.Heatmap(z=z, x=times, y=['예측'], zmin=0, zmax=3, colorscale=cs,
                             showscale=False, hoverinfo='skip'), row=2, col=1)
    if 0 < pos < n:
        fig.add_vline(x=float(times[pos - 1]), line_color='red', line_dash='dot')
    ymid = {'약': lo / 2, '보통': (lo + hi) / 2, '강': (hi + 1) / 2}[true_lab]
    fig.add_annotation(x=float(times[0]), y=ymid, text=f'정답 {true_lab}', showarrow=False, xanchor='left',
                       bgcolor='rgba(255,255,255,0.55)', font=dict(color=COLOR[ENG[true_lab]], size=11), row=1, col=1)
    fig.update_yaxes(range=[0, 1], title_text='위치점수', row=1, col=1)
    fig.update_xaxes(title_text='시간(s)', row=2, col=1)
    fig.update_layout(height=380, margin=dict(l=6, r=6, t=10, b=6), showlegend=False)
    return fig


# ── 지도학습 모델 + 시간별(윈도우) 다목적 분류 ───────────────────
MODELS = {
    'RandomForest': lambda: RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42),
    '로지스틱회귀': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
    'SVM': lambda: make_pipeline(StandardScaler(), SVC(class_weight='balanced', random_state=42)),
    '그래디언트부스팅': lambda: GradientBoostingClassifier(random_state=42),
}
TARGET_LABEL = {
    '강도': lambda d: KR[d['intensity']] if d.get('label') == 'SCRATCH' and d.get('intensity') in INT3 else None,
    '부위': lambda d: d.get('body_region'),
    '긁기 감지': lambda d: d.get('label', 'SCRATCH'),
}
REGION_COLOR = {'head_neck': '#6366F1', 'arm': '#10B981', 'torso': '#F59E0B', 'leg': '#EF4444'}


def class_color(target, c):
    if target == '강도':
        return COLOR[ENG.get(c, 'weak')]
    if target == '긁기 감지':
        return {'SCRATCH': '#10B981', 'NOT_SCRATCH': '#94A3B8'}.get(c, '#94A3B8')
    return REGION_COLOR.get(c, '#64748B')


def class_disp(target, c):
    return vk(c) if target == '부위' else c


def optimal_boundaries(scores, ytrue, step=0.025):
    """macro-F1을 최대화하는 (lo, hi) 경계 자동 탐색. 유일 제약 lo ≤ hi."""
    gv = np.round(np.arange(0.05, 0.96, step), 3)
    best = (-1.0, 0.40, 0.60)
    for L in gv:
        for H in gv:
            if H < L:
                continue
            p = np.where(scores < L, 0, np.where(scores > H, 2, 1))
            f1 = f1_score(ytrue, p, average='macro', labels=[0, 1, 2], zero_division=0)
            if f1 > best[0]:
                best = (f1, float(L), float(H))
    return best[1], best[2], best[0]


@st.cache_resource(show_spinner=False)
def ml_windows(_data, idxs, target, model_name, fs, trim, sig):
    """학습셋 샘플들을 윈도우화(det_features 12차원)해 지도학습 모델 학습.
    동작 단위(sample) GroupKFold OOF 예측으로 각 윈도우를 정직하게(자기 샘플 미학습) 판정."""
    labfn = TARGET_LABEL[target]
    X, y, grp, starts = [], [], [], []
    w = int(fs); h = max(1, int(0.25 * fs))
    for i in idxs:
        d = _data[i]; lab = labfn(d)
        if lab is None:
            continue
        a = amag(d, trim); gy = gmag(d, trim); gy = np.zeros_like(a) if gy is None else gy
        for s in range(0, len(a) - w + 1, h):
            X.append(det_features(a[s:s + w], gy[s:s + w], fs)); y.append(lab); grp.append(i); starts.append(s)
    X = np.array(X); y = np.array(y); grp = np.array(grp); starts = np.array(starts)
    classes = sorted(set(y.tolist())) if len(y) else []
    oof = cvacc = model = None
    if len(classes) >= 2 and len(set(grp.tolist())) >= 2:
        k = min(5, len(set(grp.tolist())))
        try:
            oof = cross_val_predict(MODELS[model_name](), X, y, cv=GroupKFold(k), groups=grp)
            cvacc = float((oof == y).mean())
        except Exception:
            oof = None
        model = MODELS[model_name](); model.fit(X, y)
    return dict(y=y, grp=grp, starts=starts, oof=oof, cvacc=cvacc, classes=classes, model=model, w=w, fs=fs)


def class_band_fig(times, pred, gt, target, classes, pos):
    """예측 클래스 타임라인: 위=정답·아래=예측 색 띠 + 불일치 ✕. 어떤 대상(강도/부위/감지)에도 동작."""
    cidx = {c: i for i, c in enumerate(classes)}; ncat = max(1, len(classes)); n = len(times)
    zr = lambda vals: [(cidx[v] + 0.5) if (i < pos and v in cidx) else None for i, v in enumerate(vals)]
    cs = []
    for i, c in enumerate(classes):
        col = class_color(target, c); cs += [[i / ncat, col], [(i + 1) / ncat, col]]
    fig = go.Figure(go.Heatmap(z=[zr(gt), zr(pred)], x=times, y=['정답', '예측'], zmin=0, zmax=ncat,
                               colorscale=cs, showscale=False, ygap=5, hoverinfo='skip'))
    mmx = [float(times[i]) for i in range(min(pos, n)) if pred[i] != gt[i]]
    if mmx:
        fig.add_trace(go.Scatter(x=mmx, y=['예측'] * len(mmx), mode='markers',
                                 marker=dict(symbol='x', color='#111', size=9), hoverinfo='skip', showlegend=False))
    if 0 < pos < n:
        fig.add_vline(x=float(times[pos - 1]), line_color='red', line_dash='dot')
    fig.update_layout(height=170, margin=dict(l=6, r=6, t=8, b=6), xaxis_title="시간(s)")
    return fig


@st.cache_data
def load_sample(path):
    return [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]


def build_frame(data, trim_sec=0.0):
    recs = []
    for i, d in enumerate(data):
        if d.get('label') != 'SCRATCH' or d.get('intensity') not in INT3:
            continue
        r = features(d, trim_sec); r['idx'] = i; r['intensity'] = d['intensity']; r['강도'] = KR[d['intensity']]
        for f in META:
            r[f] = d.get(f, '?')
        recs.append(r)
    return pd.DataFrame(recs)


def cv_scores(frame, feats, mode):
    """LOOCV/LOSO 위치점수. 특징 1개면 (값-약기준)/(강기준-약기준).
    2개 이상이면 표준화 후 '약 중심 → 강 중심' 축에 투영(0~1) — 경계는 그 축에 수직인 초평면.
    학습 통계(표준화·앵커)는 held-out(자신/자기 피험자)을 제외하고 계산해 정직하게."""
    if isinstance(feats, str):
        feats = [feats]
    X = frame[feats].to_numpy(float)
    lab = frame['intensity'].to_numpy(); subj = frame['subject_id'].to_numpy()
    n = len(frame); out = np.full(n, np.nan)
    for i in range(n):
        train = (subj != subj[i]) if mode == 'LOSO' else (np.arange(n) != i)
        Xtr = X[train]
        mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
        Z = (X - mu) / sd
        w = Z[train & (lab == 'weak')]; s = Z[train & (lab == 'strong')]
        cw = w.mean(0) if len(w) else Z[train].mean(0)
        cs = s.mean(0) if len(s) else Z[train].mean(0)
        a = cs - cw; aa = float((a * a).sum()) + 1e-9
        out[i] = np.clip(float(((Z[i] - cw) * a).sum()) / aa, 0, 1)
    return out


def predict(score, lo, hi):
    return np.where(score < lo, '약', np.where(score > hi, '강', '보통'))


def qwk_acc(y, p):
    q = cohen_kappa_score(y, p, weights='quadratic') if len(set(p)) > 1 else 0.0
    return q, float((np.array(p) == np.array(y)).mean())


def train_model(fr, sel, lo, hi):
    """학습 데이터로 분류 모델 생성: 표준화 통계 + 약/강 앵커(중심) + 경계."""
    X = fr[sel].to_numpy(float)
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
    Z = (X - mu) / sd; lab = fr['intensity'].to_numpy()
    cw = Z[lab == 'weak'].mean(0) if (lab == 'weak').any() else Z.mean(0)
    cs = Z[lab == 'strong'].mean(0) if (lab == 'strong').any() else Z.mean(0)
    return {'sel': sel, 'mu': mu, 'sd': sd, 'cw': cw, 'cs': cs, 'lo': lo, 'hi': hi}


def apply_model(model, fr):
    """학습된 모델을 새 데이터에 적용 → (위치점수, 예측). 통계는 학습셋 것을 그대로 사용(정직한 out-of-sample)."""
    X = fr[model['sel']].to_numpy(float)
    Z = (X - model['mu']) / model['sd']
    a = model['cs'] - model['cw']; aa = float((a * a).sum()) + 1e-9
    pos = np.clip((Z - model['cw']) @ a / aa, 0, 1)
    return pos, predict(pos, model['lo'], model['hi'])


def est_fs(d):
    """t_ns 타임스탬프로 실제 샘플링 레이트(Hz) 추정. 없으면 기본 FS."""
    t = d.get('accel', {}).get('t_ns')
    if t and len(t) > 1:
        dur = (float(t[-1]) - float(t[0])) * 1e-9
        if dur > 0:
            return (len(t) - 1) / dur
    return float(FS)


def spectro(m, fs=FS):
    nper = 128 if len(m) >= 128 else 64
    f, t, Sxx = sps.spectrogram(m - m.mean(), fs=fs, nperseg=nper, noverlap=nper // 2)
    return f, t, 10 * np.log10(Sxx + 1e-10)


def spec_fig(m, height=240, zrange=None, fs=FS):
    f, t, Z = spectro(m, fs)
    kw = dict(z=Z, x=t, y=f, colorscale='Viridis', showscale=False)
    if zrange:
        kw['zmin'], kw['zmax'] = zrange
    fig = go.Figure(go.Heatmap(**kw))
    fig.update_layout(height=height, margin=dict(l=6, r=6, t=6, b=6), xaxis_title="시간(s)", yaxis_title="주파수(Hz)")
    return fig


def wave_fig(m, color=ACCENT, height=150, fs=FS):
    t = np.arange(len(m)) / fs
    fig = go.Figure(go.Scatter(x=t, y=m, mode='lines', line=dict(color=color, width=1.2)))
    fig.update_layout(height=height, margin=dict(l=6, r=6, t=6, b=6), showlegend=False,
                      xaxis_title="시간(s)", yaxis_title="가속도 크기")
    return fig


def psd_welch(m, fs=FS):
    f, Pxx = sps.welch(m - m.mean(), fs=fs, nperseg=min(256, len(m)))
    return f, 10 * np.log10(Pxx + 1e-12)


def psd_fig(m, fs=FS, height=220, color=ACCENT):
    f, db = psd_welch(m, fs)
    fig = go.Figure(go.Scatter(x=f, y=db, mode='lines', line=dict(color=color, width=1.4)))
    fig.update_layout(height=height, margin=dict(l=6, r=6, t=6, b=6),
                      xaxis_title="주파수(Hz)", yaxis_title="Power/Freq (dB/Hz)")
    return fig


def psd_compare_fig(mw, mc, fs=FS, height=260):
    fig = go.Figure()
    for m, nm, col in [(mc, '✅ 정분류', COLOR['weak']), (mw, '❌ 오분류', COLOR['strong'])]:
        f, db = psd_welch(m, fs)
        fig.add_trace(go.Scatter(x=f, y=db, mode='lines', name=nm, line=dict(color=col, width=1.6)))
    fig.update_layout(height=height, margin=dict(l=6, r=6, t=6, b=6), xaxis_title="주파수(Hz)",
                      yaxis_title="Power/Freq (dB/Hz)", legend=dict(orientation='h', y=1.15))
    return fig


def confusion_fig(res, height=330):
    cm = pd.crosstab(res['강도'], res['예측']).reindex(index=LABELS, columns=LABELS, fill_value=0)
    fig = go.Figure(go.Heatmap(z=cm.values, x=LABELS, y=LABELS, colorscale='Blues',
                               text=cm.values, texttemplate="%{text}", showscale=False))
    fig.update_layout(height=height, margin=dict(l=6, r=6, t=6, b=6), xaxis_title="예측", yaxis_title="실제")
    return fig


ENG = {'약': 'weak', '보통': 'normal', '강': 'strong'}


def overview_fig(res, lo, hi):
    """오분류 한눈에: 위치점수 스트립. x=위치, y=실제강도 레인, 배경=예측구간, X=오분류(색=예측)."""
    ymap = {'약': 0, '보통': 1, '강': 2}
    fig = go.Figure()
    fig.add_vrect(x0=0, x1=lo, fillcolor=COLOR['weak'], opacity=.08, line_width=0)
    fig.add_vrect(x0=lo, x1=hi, fillcolor=COLOR['normal'], opacity=.08, line_width=0)
    fig.add_vrect(x0=hi, x1=1, fillcolor=COLOR['strong'], opacity=.08, line_width=0)
    fig.add_vline(x=lo, line_dash='dot', line_color='gray'); fig.add_vline(x=hi, line_dash='dot', line_color='gray')
    for lab in LABELS:
        d = res[res['강도'] == lab].sort_values('위치').reset_index(drop=True)
        if d.empty:
            continue
        m = len(d)
        yj = ymap[lab] + (np.linspace(-0.32, 0.32, m) if m > 1 else np.array([0.0]))
        for correct in (True, False):
            dd = d[d['정답'] == correct]
            if dd.empty:
                continue
            pos = dd.index.to_numpy()
            hov = [f"idx {int(r['idx'])} · 실제 {r['강도']} → 예측 {r['예측']} · 위치 {r['위치']:.2f}"
                   for _, r in dd.iterrows()]
            fig.add_trace(go.Scatter(
                x=dd['위치'], y=yj[pos], mode='markers', showlegend=False, text=hov, hoverinfo='text',
                marker=dict(color=[COLOR[ENG[p]] for p in dd['예측']], size=(9 if correct else 15),
                            symbol=('circle' if correct else 'x'), opacity=(.5 if correct else 1),
                            line=dict(width=(0 if correct else 2), color='#111'))))
    fig.update_layout(height=330, margin=dict(l=6, r=6, t=10, b=6),
                      xaxis=dict(title='위치 점수 (0=약 기준 · 1=강 기준)', range=[-0.02, 1.02]),
                      yaxis=dict(tickmode='array', tickvals=[0, 1, 2], ticktext=LABELS,
                                 title='실제 강도', range=[-0.6, 2.6]))
    return fig


def boundary2d_fig(frame, sel, lo, hi, names):
    """특징 2개일 때 결정 경계를 2D로 시각화: 표준화 공간에서 약↔강 축 + 경계선 두 개(띠)."""
    X = frame[sel].to_numpy(float)
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
    Z = (X - mu) / sd; lab = frame['intensity'].to_numpy()
    cw = Z[lab == 'weak'].mean(0); cs = Z[lab == 'strong'].mean(0)
    a = cs - cw; perp = np.array([-a[1], a[0]]); perp = perp / (np.hypot(*perp) + 1e-9)
    fig = go.Figure()
    for g in ['weak', 'normal', 'strong']:
        m = lab == g
        fig.add_trace(go.Scatter(x=Z[m, 0], y=Z[m, 1], mode='markers', name=KR[g],
                                 marker=dict(color=COLOR[g], size=9, opacity=.8, line=dict(width=.5, color='white'))))
    fig.add_trace(go.Scatter(x=[cw[0], cs[0]], y=[cw[1], cs[1]], mode='markers+lines', name='약↔강 축',
                             marker=dict(color='#111', size=13, symbol='x'), line=dict(color='#888', dash='dash')))
    for p, nm in [(lo, '약↔보통 경계'), (hi, '보통↔강 경계')]:
        c0 = cw + p * a; p1 = c0 - 3 * perp; p2 = c0 + 3 * perp
        fig.add_trace(go.Scatter(x=[p1[0], p2[0]], y=[p1[1], p2[1]], mode='lines', name=nm,
                                 line=dict(color='#64748B', dash='dot', width=2)))
    fig.update_layout(height=460, margin=dict(l=6, r=6, t=6, b=6),
                      xaxis_title=f"{names[0]} (표준화)", yaxis_title=f"{names[1]} (표준화)",
                      legend=dict(orientation='h', y=1.08))
    return fig


st.set_page_config(page_title="긁기 강도 오분류 진단", page_icon=":material/troubleshoot:", layout="wide")

# ── 사이드바: 데이터 · 특징 · 검증 · 경계 · 필터 ─────────────────
def read_ndjson(f):
    return [json.loads(l) for l in f.getvalue().decode('utf-8').splitlines() if l.strip()]


with st.sidebar:
    st.markdown("#### :material/database: 데이터")
    up = st.file_uploader("데이터 (NDJSON, 선택)", type=['ndjson', 'jsonl', 'json', 'txt'])
    if up is not None:
        data = read_ndjson(up)
        st.caption(f"업로드: **{up.name}** · {len(data)}줄")
    else:
        data = load_sample(DATA_PATH)
        st.caption("기본: sample_data.ndjson")
    TRIM_SEC = 1.0 if st.toggle("마지막 1초 제거 (전처리)", value=True,
                                help="긁기 종료 구간의 잡음/여운을 제거. 각 신호 끝에서 (1초×샘플링레이트) 샘플을 잘라냅니다.") else 0.0
    full = build_frame(data, TRIM_SEC)

    feats = [k for k in FEATS_ALL if full[k].notna().any() and full[k].std(skipna=True) > 0]
    FEATS = {k: FEATS_ALL[k] for k in feats}
    rho0 = {k: abs(spearmanr(full[k], full['intensity'].map(INT3))[0] or 0) for k in feats}
    best = max(feats, key=lambda k: rho0[k])

    st.markdown("#### :material/tune: 강도 분류 · 검증")
    _default = ['jerk'] if 'jerk' in feats else [best]
    sel = st.multiselect("분류 특징 (여러 개 선택 가능)", feats, default=_default,
                         format_func=lambda k: FEATS[k] + (' ⭐' if k == best else ''))
    sel = sel or _default
    multi = len(sel) > 1
    feat = sel[0]
    if multi:
        st.caption(f"{len(sel)}개 조합 — 표준화 후 약↔강 중심축에 투영해 위치를 계산합니다.")
    nsubj = full['subject_id'].nunique()
    mode = st.radio("검증 방식", (['LOOCV', 'LOSO'] if nsubj > 1 else ['LOOCV']), horizontal=True,
                    help="LOOCV=샘플 1개씩 제외 / LOSO=피험자 1명씩 제외(2명 이상). 학습·평가 미겹침.")

    st.markdown("#### :material/filter_list: 필터")
    varying = {f: sorted(full[f].unique()) for f in META if full[f].nunique() > 1}
    frame = full
    if varying:
        for f, vals in varying.items():
            chosen = st.multiselect(META[f], vals, default=vals, format_func=vk, key=f"flt_{f}")
            frame = frame[frame[f].isin(chosen or vals)]
        st.caption(f"선택 **{len(frame)}** / {len(full)}개")
    else:
        st.caption("값이 여러 개인 축이 아직 없습니다. 데이터가 쌓이면 필터가 늘어납니다.")

    if frame.empty or frame['강도'].nunique() < 2:
        st.warning("분석에 필요한 데이터가 부족합니다(최소 2개 강도). 필터를 넓혀 주세요."); st.stop()
    frame = frame.reset_index(drop=True)

    st.markdown("#### :material/target: 강도 경계")
    pos_cv = cv_scores(frame, sel, mode)
    ytrue_cv = frame['강도'].map(LAB_MAP).values
    opt_lo, opt_hi, opt_f1 = optimal_boundaries(pos_cv, ytrue_cv)
    auto_bound = st.toggle("F1 최대 경계 자동 선택", value=True,
                           help="선택한 특징에서 macro-F1을 최대화하는 경계를 자동 적용. 끄면 수동 슬라이더.")
    if auto_bound:
        lo, hi = opt_lo, opt_hi
        st.caption(f"F1 최적 경계 **[{lo:.2f}, {hi:.2f}]** (macro-F1 {opt_f1:.2f}) 자동 적용")
    else:
        lo, hi = st.slider("강도 경계 · 약 | 보통 | 강", 0.0, 1.0, (opt_lo, opt_hi), 0.01,
                           help="0.5에 얽매이지 않고 어디든 (약 경계 ≤ 강 경계).")

# ── CV 판정 ─────────────────────────────────────────────────────
res = frame.copy()
res['위치'] = np.round(pos_cv, 3)
res['예측'] = predict(res['위치'].values, lo, hi)
res['정답'] = res['예측'] == res['강도']
yt = res['강도'].map(LAB_MAP).values
yp = res['예측'].map(LAB_MAP).values
qwk, acc = qwk_acc(yt, yp)
severe = int(((yt == 0) & (yp == 2)).sum() + ((yt == 2) & (yp == 0)).sum())

fs_data = float(np.median([est_fs(d) for d in data]))

st.title("긁기 강도 오분류 진단")
if multi:
    st.caption(f"분류 특징 **{' + '.join(FEATS[k] for k in sel)}** ({len(sel)}개 조합) · 검증 **{mode}** · "
               f"샘플링 **{fs_data:.0f} Hz** · 표준화 후 약↔강 축 투영 (지표는 교차검증 결과)")
else:
    gw = full[full.intensity == 'weak'][feat].mean(); gs = full[full.intensity == 'strong'][feat].mean()
    st.caption(f"분류 특징 **{FEATS[feat]}** · 검증 **{mode}** · 샘플링 **{fs_data:.0f} Hz** · "
               f"약 기준 {gw:.2f} · 강 기준 {gs:.2f} (지표는 교차검증 결과)")

with st.container(horizontal=True):
    st.metric(f"정확도 ({mode})", f"{acc:.0%}", border=True)
    st.metric("QWK (순서형)", f"{qwk:.2f}", border=True, help="약<보통<강 순서 반영. 1에 가까울수록 좋음.")
    st.metric("심각 오류", severe, border=True, help="약↔강 정반대 혼동. 0이 목표.")
    st.metric("오분류 / 전체", f"{int((~res['정답']).sum())} / {len(res)}", border=True)

screen_all, screen_one = st.tabs([":material/dashboard: 데이터 종합", ":material/biotech: 개별 상세"])

# ═══ 화면 1: 데이터 종합 (전체를 한눈에) ══════════════════════════
with screen_all:
    box = st.container(border=True)
    box.markdown("**오분류 한눈에 보기** · 배경색=예측 구간 · ○ 정분류 · ✕ 오분류(색=예측된 강도)")
    box.plotly_chart(overview_fig(res, lo, hi), width="stretch")
    box.caption(f"오분류 {int((~res['정답']).sum())}건 · 정분류는 약→강 대각선, 대각선을 벗어난 ✕가 오분류. "
                "약 레인의 ✕가 오른쪽 끝(강 구간)이면 심각 오류. 개별 샘플은 '개별 상세' 창에서 뜯어봅니다.")
    t_pat, t_feat, t_opt = st.tabs(["패턴 요약", "특징 비교", "경계 최적화"])

# ═══ 화면 2: 개별 상세 (샘플 하나를 깊게) ═════════════════════════
with screen_one:
    st.markdown("**개별 상세** — 샘플 하나를 고르면 시간축을 따라 윈도우별로 분류합니다. **대상(강도/부위/긁기 감지)과 모델**을 고를 수 있어요.")
    only_wrong = st.checkbox(f"강도 오분류만 보기 ({int((~res['정답']).sum())}건)", value=False)
    pool = (res[~res['정답']] if only_wrong else res).reset_index(drop=True)
    if pool.empty:
        st.success("오분류가 없습니다.", icon=":material/check_circle:")
    else:
        cS, cT, cM = st.columns([2, 1, 1])
        ri = cS.selectbox("샘플 선택", range(len(pool)),
                          format_func=lambda i: f"idx {int(pool.iloc[i]['idx'])} · 실제 {pool.iloc[i]['강도']} "
                                                f"(강도판정 {pool.iloc[i]['예측']})" + ("" if pool.iloc[i]['정답'] else "  ❌"))
        det_target = cT.selectbox("시간별 대상", ['강도', '부위', '긁기 감지'], key="det_target")
        model_opts = (['규칙기반(앵커)'] if det_target == '강도' else []) + list(MODELS)
        det_model = cM.selectbox("분류 모델", model_opts, key="det_model")

        r = pool.iloc[ri]; d = data[int(r['idx'])]; sid = int(r['idx']); fs = est_fs(d)
        m = amag(d, TRIM_SEC); g = gmag(d, TRIM_SEC); dur = len(m) / fs
        L = r['강도']; lc = {'약': 'green', '보통': 'orange', '강': 'red'}[L]; ok = bool(r['정답'])
        rule = (det_target == '강도' and det_model == '규칙기반(앵커)')
        true_cls = TARGET_LABEL[det_target](d)

        posv = None; cvacc = None; honest_note = None
        if rule:
            # 정직한 모델: 종합 화면과 같은 (필터된) frame에서 이 샘플(LOSO면 이 피험자)만 제외 → out-of-sample
            train_fr = frame[frame['subject_id'] != r['subject_id']] if mode == 'LOSO' else frame[frame['idx'] != sid]
            if train_fr['intensity'].nunique() < 2:
                train_fr = frame; honest_note = "이 샘플을 빼면 강도 클래스가 부족해 흐름을 in-sample로 계산 (과대평가 가능)."
            times, posv, pred = sample_strength_series(d, TRIM_SEC, fs, train_model(train_fr, sel, lo, hi))
            classes = LABELS
        else:
            # 감지는 NOT_SCRATCH가 frame에서 제외되므로 전체 data로, 강도/부위는 필터된 frame으로 학습
            idxs = tuple(range(len(data))) if det_target == '긁기 감지' else tuple(sorted(int(x) for x in frame['idx']))
            W = ml_windows(data, idxs, det_target, det_model, int(round(fs_data)), TRIM_SEC, len(data))
            classes = W['classes']; cvacc = W['cvacc']; maskw = (W['grp'] == sid)
            if W['model'] is None or not maskw.any() or true_cls is None:
                times = np.array([]); pred = np.array([])
            else:
                times = (W['starts'][maskw] + W['w'] / 2) / W['fs']
                pred = W['oof'][maskw] if W['oof'] is not None else W['model'].predict(
                    np.array([det_features(m[s:s + W['w']], (g if g is not None else np.zeros_like(m))[s:s + W['w']], fs)
                              for s in W['starts'][maskw]]))
        gt = [true_cls] * len(times)
        n = len(times)
        agree = np.array([pred[i] == gt[i] for i in range(n)], dtype=bool)
        true_disp = class_disp(det_target, true_cls) if true_cls is not None else '—'

        st.markdown(f"### 실제 **{true_disp}** · 시간별 **{det_target}** 판정 ({det_model})")
        if honest_note:
            st.warning(honest_note, icon=":material/warning:")

        if n == 0:
            st.info(f"'{det_target}' × '{det_model}'로는 이 샘플을 분류할 수 없습니다 "
                    "(학습 데이터에 클래스가 하나뿐이거나 이 샘플에 라벨 없음). 아래 원신호는 확인할 수 있어요.",
                    icon=":material/info:")
            pos = 0
        else:
            vc = pd.Series(pred).value_counts()
            dom_frac = vc.iloc[0] / n; hit = agree.mean()
            if rule:
                final_pred = str(predict(np.array([float(np.mean(posv))]), lo, hi)[0])  # 평균 위치(동률 안전·순서형)
            else:
                final_pred = str(vc.index[0])  # 다수결
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("종합 판정", class_disp(det_target, final_pred),
                      help="강도 규칙=평균 위치점수 구간 / ML=윈도우 다수결")
            k2.metric("시간 일관성", f"{dom_frac:.0%}", delta=f"{int(vc.iloc[0])}/{n} 창", delta_color="off",
                      help="가장 많이 나온 예측이 차지하는 비율 — 낮으면 시간에 따라 흔들림")
            k3.metric("정답 일치율", f"{hit:.0%}", delta=f"{int(agree.sum())}/{n} 창", delta_color="off",
                      help="예측==실제 라벨인 윈도우 비율 (홉 75% 겹침으로 창들은 독립 아님)")
            if rule:
                k4.metric("윈도우", f"{n}개", help="1초 창 · 0.25초 홉")
            else:
                k4.metric("모델 CV 정확도", f"{cvacc:.0%}" if cvacc is not None else "—",
                          help="학습셋 동작 단위 GroupKFold 교차검증 정확도 (참고)")

            # 재생 제어 (session_state, det_ 네임스페이스). 기본은 전체 표시.
            dsig = (sid, det_target, det_model, tuple(sel), round(lo, 2), round(hi, 2), TRIM_SEC, mode)
            if st.session_state.get('det_sig') != dsig:
                st.session_state['det_sig'] = dsig; st.session_state['det_pos'] = n; st.session_state['det_playing'] = False
            pos = min(st.session_state.get('det_pos', n), n)
            pc = st.columns([1, 1, 1, 1, 3])
            if pc[0].button("▶ 재생", key="det_play"):
                st.session_state['det_playing'] = True; st.session_state['det_pos'] = 0 if pos >= n else pos
            if pc[1].button("⏸ 일시정지", key="det_pause"):
                st.session_state['det_playing'] = False
            if pc[2].button("⏭ 한 윈도우", key="det_step"):
                st.session_state['det_playing'] = False; st.session_state['det_pos'] = min(n, pos + 1)
            if pc[3].button("⏹ 전체", key="det_stop"):
                st.session_state['det_playing'] = False; st.session_state['det_pos'] = n
            dspeed = pc[4].slider("재생 속도", 1, 20, 8, key="det_speed")
            pos = min(st.session_state.get('det_pos', n), n)

            box = st.container(border=True)
            if rule:
                box.markdown("**강도 흐름** · 선=위치점수(0=약 기준·1=강 기준), 배경=판정 강도 구간, ○=정답 일치·✕=불일치")
                box.plotly_chart(flow_fig(times, posv, pred, L, lo, hi, pos), width="stretch", key="det_flow")
            else:
                box.markdown(f"**{det_target} 예측 타임라인** · 위=정답·아래=예측 · ✕=불일치")
                box.plotly_chart(class_band_fig(times, pred, gt, det_target, classes, pos), width="stretch", key="det_flow")
            leg = " · ".join(f"<span>{class_disp(det_target, c)}</span>" for c in classes)
            box.markdown(f"종합 판정 **{class_disp(det_target, final_pred)}** · 정답 일치 {hit:.0%} · 최빈 예측 {dom_frac:.0%} · 클래스: {leg}",
                         unsafe_allow_html=True)

            # 재생 커서 위치의 현재 윈도우 드릴다운 (전체보기가 아닐 때)
            if 0 < pos < n:
                w = int(fs); s0 = int(round(times[pos - 1] * fs - w / 2)); s0 = max(0, min(len(m) - w, s0))
                seg = m[s0:s0 + w]; pk = pred[pos - 1]; agk = (pk == gt[pos - 1])
                b = st.container(border=True)
                extra = f" · 위치 {posv[pos-1]:.2f}" if posv is not None else ""
                b.markdown(f"**현재 윈도우 #{pos}** ({times[pos-1]:.2f}s){extra} · 예측 **{class_disp(det_target, pk)}** · "
                           + ("✅ 일치" if agk else "❌ 불일치"))
                wc1, wc2 = b.columns(2)
                wc1.plotly_chart(wave_fig(seg, color=class_color(det_target, pk), height=180, fs=fs), width="stretch", key="det_wwave")
                wc2.plotly_chart(spec_fig(seg, height=180, fs=fs), width="stretch", key="det_wspec")

        with st.expander("원신호 (파형 · 스펙트로그램 · PSD · 자이로)", icon=":material/ssid_chart:"):
            trim_note = f" · 마지막 1초 제거됨(-{round(TRIM_SEC*fs)}샘플)" if TRIM_SEC > 0 else ""
            show_axes = st.toggle("가속도 x·y·z 원신호로 보기 (기본: 크기)", value=False, key="det_axes")
            b = st.container(border=True); b.markdown(f"**시간축 가속도**{trim_note}")
            if show_axes:
                ax_x, ax_y, ax_z = _axes(d['accel'], d, TRIM_SEC); t = np.arange(len(ax_x)) / fs
                fig = go.Figure()
                for arr, nm in [(ax_x, 'x'), (ax_y, 'y'), (ax_z, 'z')]:
                    fig.add_trace(go.Scatter(x=t, y=arr, mode='lines', name=nm, line=dict(width=1)))
                fig.update_layout(height=200, margin=dict(l=6, r=6, t=6, b=6), xaxis_title="시간(s)",
                                  yaxis_title="가속도", legend=dict(orientation='h', y=1.2))
                b.plotly_chart(fig, width="stretch")
            else:
                b.plotly_chart(wave_fig(m, height=200, fs=fs), width="stretch")
            c1, c2 = st.columns(2)
            c1.plotly_chart(spec_fig(m, height=280, fs=fs), width="stretch", key="det_spec")
            c2.plotly_chart(psd_fig(m, fs=fs, height=280), width="stretch", key="det_psd")
            if g is not None:
                gc1, gc2 = st.columns(2)
                gc1.plotly_chart(wave_fig(g, color='#8B5CF6', height=200, fs=fs), width="stretch", key="det_gwave")
                gc2.plotly_chart(psd_fig(g, fs=fs, height=200, color='#8B5CF6'), width="stretch", key="det_gpsd")

        with st.expander("이 샘플의 특징값 · 특징의 시간 변화", icon=":material/science:"):
            fc1_, fc2_ = st.columns([2, 3])
            with fc1_:
                b = st.container(border=True); b.markdown("**특징값** · ⭐=현재 분류 사용")
                desc = describe_features(d, TRIM_SEC)
                fpanel = pd.DataFrame([{'도메인': gr, '특징': (('⭐ ' if k in sel else '') + nm),
                                        '값': round(v, 3)} for gr, nm, k, v in desc])
                b.dataframe(fpanel, width="stretch", hide_index=True, height=min(520, 44 + 35 * len(fpanel)))
            with fc2_:
                b = st.container(border=True); b.markdown("**특징의 시간 변화** · 1초 창")
                wkey = b.selectbox("볼 특징", list(WIN_KR), format_func=lambda k: WIN_KR[k], key="det_win")
                wt, wv = windowed(m, fs, wkey)
                if len(wt) >= 1:
                    wf = go.Figure(go.Scatter(x=wt, y=wv, mode='lines+markers', line=dict(color=ACCENT, width=1.5)))
                    wf.update_layout(height=280, margin=dict(l=6, r=6, t=6, b=6), xaxis_title="시간(s)", yaxis_title=WIN_KR[wkey])
                    b.plotly_chart(wf, width="stretch", key="det_winfig")
                else:
                    b.caption("윈도우를 만들기엔 신호가 짧습니다.")

        with st.expander("시간별 누적 일치율 · 윈도우 판정 로그", icon=":material/table_chart:"):
            rc = rt_ = 0; ct, cv = [], []; rec = []
            for i in range(pos):
                rt_ += 1; rc += int(agree[i]); ct.append(float(times[i])); cv.append(rc / rt_ * 100)
                row = {'윈도우': i + 1, '시간(s)': round(float(times[i]), 2),
                       '예측': class_disp(det_target, pred[i]), '정답': class_disp(det_target, gt[i]),
                       '결과': '✅' if agree[i] else '❌'}
                if posv is not None:
                    row['위치점수'] = round(float(posv[i]), 2)
                rec.append(row)
            a1, a2 = st.columns([3, 2])
            with a1:
                b = st.container(border=True); b.markdown("**누적 일치율** (예측==실제, 재생 위치까지)")
                if ct:
                    af = go.Figure(go.Scatter(x=ct, y=cv, mode='lines', line=dict(color=ACCENT, width=2)))
                    af.update_layout(height=240, margin=dict(l=6, r=6, t=6, b=6), xaxis_title="시간(s)",
                                     yaxis_title="누적 일치율(%)", yaxis_range=[0, 100])
                    b.plotly_chart(af, width="stretch", key="det_acc")
                else:
                    b.caption("▶ 재생하거나 ⏭로 진행하면 채워집니다.")
            with a2:
                b = st.container(border=True); b.markdown("**윈도우별 판정** (최근순)")
                b.dataframe(pd.DataFrame(rec).iloc[::-1] if rec else pd.DataFrame(),
                            width="stretch", hide_index=True, height=240)

        if not ok and det_target == '강도':
            with st.expander(f"강도 오분류 진단 — 같은 실제 '{L}' 정분류와 나란히 비교", icon=":material/compare_arrows:", expanded=True):
                same_ok = res[(res['강도'] == L) & (res['정답'])].sort_values('위치').reset_index(drop=True)
                if same_ok.empty:
                    st.warning(f"같은 실제 라벨('{L}')의 정분류 샘플이 없어 비교 대상이 없습니다.")
                else:
                    ci = st.selectbox("비교할 정분류 샘플(같은 실제 라벨)", range(len(same_ok)), index=len(same_ok) // 2,
                                      format_func=lambda j: f"idx {int(same_ok.loc[j,'idx'])} · 위치 {same_ok.loc[j,'위치']:.2f}",
                                      key=f"det_cmp_{int(r['idx'])}")
                    wcol = COLOR[ENG[L]]; mw = m; fw = fs
                    cr = same_ok.loc[ci]; dc = data[int(cr['idx'])]; mc = amag(dc, TRIM_SEC); fcz = est_fs(dc)
                    _, _, Zw = spectro(mw, fw); _, _, Zc = spectro(mc, fcz)
                    zr = (min(Zw.min(), Zc.min()), max(Zw.max(), Zc.max()))
                    cL, cR = st.columns(2)
                    with cL:
                        b = st.container(border=True)
                        b.markdown(f"**:red[❌ 오분류]** · 실제 {L} → 예측 {r['예측']} · 위치 {r['위치']:.2f}")
                        b.plotly_chart(wave_fig(mw, color=wcol, fs=fw), width="stretch", key="cmp_dw_wave")
                        b.plotly_chart(spec_fig(mw, zrange=zr, fs=fw), width="stretch", key="cmp_dw_spec")
                    with cR:
                        b = st.container(border=True)
                        b.markdown(f"**:green[✅ 정분류]** · 실제 {L} → 예측 {L} · 위치 {cr['위치']:.2f}")
                        b.plotly_chart(wave_fig(mc, color=wcol, fs=fcz), width="stretch", key="cmp_dc_wave")
                        b.plotly_chart(spec_fig(mc, zrange=zr, fs=fcz), width="stretch", key="cmp_dc_spec")
                    b = st.container(border=True)
                    b.markdown("**PSD 비교** — 오분류 vs 정분류의 주파수별 파워 차이")
                    b.plotly_chart(psd_compare_fig(mw, mc, fs=fw), width="stretch")
                    cmp = pd.DataFrame({'특징': [FEATS[k] for k in FEATS],
                                        '❌ 오분류': [round(r[k], 3) for k in FEATS],
                                        '✅ 정분류': [round(cr[k], 3) for k in FEATS]})
                    cmp['차이(%)'] = [f"{(r[k]-cr[k])/(abs(cr[k])+1e-9)*100:+.0f}%" for k in FEATS]
                    st.dataframe(cmp, width="stretch", hide_index=True)

        # 자동 재생
        if st.session_state.get('det_playing') and pos < n:
            st.session_state['det_pos'] = min(n, pos + max(1, dspeed)); time.sleep(0.05); st.rerun()
        elif pos >= n and st.session_state.get('det_playing'):
            st.session_state['det_playing'] = False

# ── 탭 2: 패턴 요약 ─────────────────────────────────────────────
with t_pat:
    c1, c2 = st.columns([2, 3])
    with c1:
        box = st.container(border=True); box.markdown(f"**혼동행렬** · 정확도 {acc:.0%}")
        box.plotly_chart(confusion_fig(res), width="stretch")
    with c2:
        box = st.container(border=True); box.markdown("**조건별 오분류율** · 어디서 많이 틀리나")
        if varying:
            dfp = res.copy(); dfp['오분류'] = (~dfp['정답']).astype(int)
            fig = go.Figure()
            for f in varying:
                g = dfp.groupby(f)['오분류'].mean()
                fig.add_trace(go.Bar(x=[f"{META[f]}·{vk(i)}" for i in g.index], y=g.values * 100,
                                     name=META[f]))
            fig.update_layout(height=330, margin=dict(l=6, r=6, t=6, b=6), yaxis_title="오분류율(%)",
                              yaxis_range=[0, 100], legend=dict(orientation='h', y=1.12), showlegend=len(varying) > 1)
            box.plotly_chart(fig, width="stretch")
        else:
            box.caption("조건(부위·사람 등) 변이가 아직 없습니다. 데이터가 쌓이면 채워집니다.")

    primary = 'body_region' if res['body_region'].nunique() > 1 else next(iter(varying), None)
    if primary:
        box = st.container(border=True)
        box.markdown(f"**{META[primary]} × 강도 정확도** · 빨강 칸이 집중 오분류")
        piv = res.pivot_table(index=primary, columns='강도', values='정답', aggfunc='mean').reindex(columns=LABELS)
        cnt = res.pivot_table(index=primary, columns='강도', values='정답', aggfunc='count').reindex(columns=LABELS)
        ylab = [vk(v) for v in piv.index]
        txt = [[("-" if pd.isna(piv.values[r][c]) else f"{piv.values[r][c]*100:.0f}%<br>({int(cnt.values[r][c] or 0)})")
                for c in range(piv.shape[1])] for r in range(piv.shape[0])]
        fig = go.Figure(go.Heatmap(z=piv.values * 100, x=LABELS, y=ylab, zmin=0, zmax=100,
                                   colorscale='RdYlGn', text=txt, texttemplate="%{text}", colorbar=dict(title="정확도%")))
        fig.update_layout(height=110 + 46 * len(ylab), margin=dict(l=6, r=6, t=6, b=6),
                          xaxis_title="실제 강도", yaxis_title=META[primary])
        box.plotly_chart(fig, width="stretch")

# ── 탭 3: 특징 비교 (연구용) ─────────────────────────────────────
with t_feat:
    st.markdown("**특징별 강도 분리력** — Spearman rho(강도 순서와의 상관)와 단일특징 QWK")
    rows = []
    yint = frame['강도'].map(LAB_MAP).values
    for k in FEATS:
        rho = spearmanr(frame[k], frame['intensity'].map(INT3))[0]
        p = predict(cv_scores(frame, k, mode), lo, hi)
        q, a = qwk_acc(yint, pd.Series(p).map(LAB_MAP).values)
        rows.append({'k': k, '특징': FEATS[k], 'rho': rho if rho == rho else 0.0, 'QWK': q, '정확도': a})
    fdf = pd.DataFrame(rows).sort_values('rho')
    c1, c2 = st.columns([3, 2])
    with c1:
        box = st.container(border=True); box.markdown("**|Spearman rho| 순위** · 현재 선택=보라")
        fig = go.Figure(go.Bar(y=fdf['특징'], x=fdf['rho'].abs(), orientation='h',
                               marker_color=[ACCENT if k in sel else GRAY for k in fdf['k']],
                               text=[f"{r:.2f}" for r in fdf['rho']], textposition='auto'))
        fig.update_layout(height=300, margin=dict(l=6, r=6, t=6, b=6), xaxis=dict(range=[0, 1], title="|rho|"))
        box.plotly_chart(fig, width="stretch")
    with c2:
        box = st.container(border=True); box.markdown("**수치**")
        show = fdf.sort_values('rho', key=lambda s: s.abs(), ascending=False)[['특징', 'rho', 'QWK', '정확도']].copy()
        show['rho'] = show['rho'].round(2); show['QWK'] = show['QWK'].round(2)
        show['정확도'] = (show['정확도'] * 100).round(0).astype(int).astype(str) + '%'
        box.dataframe(show, width="stretch", hide_index=True)

    st.info("진폭·저크·평균절대편차·에너지는 서로 상관 0.95 이상 — 하나만 고르면 됩니다. "
            "**진폭(또는 저크) + 스펙트럼 중심** 조합이 QWK 0.89로 가장 높습니다 (크기와 속도라는 서로 다른 축을 재기 때문).",
            icon=":material/lightbulb:")

    best_single_q = fdf['QWK'].max()
    if multi:
        verdict = "조합이 더 낫습니다 ✅" if qwk > best_single_q + 1e-9 else "조합이 단일보다 낫지 않습니다 — 단순한 단일 특징을 권장 ⚠️"
        st.info(f"선택한 {len(sel)}개 조합 QWK **{qwk:.2f}** vs 최고 단일 특징 QWK **{best_single_q:.2f}** → {verdict}",
                icon=":material/balance:")
    else:
        st.caption("여러 특징을 함께 고르면(사이드바) 여기서 '조합 vs 단일' 성능을 비교하고, 정확히 2개면 아래에 2D 결정 경계가 그려집니다.")

    box = st.container(border=True)
    box.markdown("**특징별 분포(박스)** · 약/보통/강이 잘 벌어질수록 좋은 특징")
    fk = list(FEATS); ncol = min(3, len(fk)); nrow = (len(fk) + ncol - 1) // ncol
    grid = make_subplots(rows=nrow, cols=ncol, subplot_titles=[FEATS[k] for k in fk],
                         vertical_spacing=0.16, horizontal_spacing=0.08)
    for i, k in enumerate(fk):
        r, c = i // ncol + 1, i % ncol + 1
        for g in ['weak', 'normal', 'strong']:
            sub = res[res.intensity == g]
            grid.add_trace(go.Box(y=sub[k], name=KR[g], legendgroup=KR[g], showlegend=(i == 0),
                                  marker_color=COLOR[g], boxpoints='all', jitter=0.4, line=dict(width=1)), row=r, col=c)
    for ann, k in zip(grid.layout.annotations, fk):
        ann.font.size = 12
        if k in sel:
            ann.text += " ⭐"; ann.font.color = ACCENT
    grid.update_layout(height=230 * nrow, margin=dict(l=6, r=6, t=30, b=6), boxmode='group',
                       legend=dict(orientation='h', y=1.06, x=0.5, xanchor='center'))
    box.plotly_chart(grid, width="stretch")

    if len(sel) == 2:
        box = st.container(border=True)
        box.markdown(f"**2D 결정 경계** · {FEATS[sel[0]]} × {FEATS[sel[1]]} (표준화 공간)")
        box.plotly_chart(boundary2d_fig(frame, sel, lo, hi, [FEATS[sel[0]], FEATS[sel[1]]]), width="stretch")
        box.caption("×표=약·강 중심(앵커), 점선=두 중심을 잇는 축, 회색 점선 2개=경계. "
                    "특징 1개일 땐 점이던 경계가, 2개가 되면 이렇게 **직선(띠)** 이 됩니다.")
    elif multi:
        st.caption(f"현재 {len(sel)}개 특징 → 경계는 {len(sel)}차원 공간의 초평면입니다(그림은 정확히 2개일 때 제공).")

# ── 탭 4: 경계 최적화 ───────────────────────────────────────────
with t_opt:
    st.markdown("**경계 자동 최적화** — 강도는 순서형(약<보통<강)이라 기준마다 최적 경계가 다릅니다. QWK 권장.")
    st.caption(f"현재 경계 [{lo:.2f}, {hi:.2f}] · 정확도 {acc:.0%} · QWK {qwk:.2f} · 심각오류 {severe}")
    if st.button("최적 경계 찾기", type="primary"):
        sc_all = res['위치'].values

        def cls(L, H):
            return np.where(sc_all < L, 0, np.where(sc_all > H, 2, 1))

        def m_all(p):
            return {'정확도': float((p == yt).mean()),
                    'macro-F1': f1_score(yt, p, average='macro', labels=[0, 1, 2], zero_division=0),
                    'QWK': cohen_kappa_score(yt, p, weights='quadratic') if len(set(p)) > 1 else 0.0,
                    '심각오류': int(np.sum(((yt == 0) & (p == 2)) | ((yt == 2) & (p == 0))))}

        # 전 구간 탐색: 유일한 제약은 약 경계 ≤ 강 경계 (0.5에 얽매이지 않음)
        gv = np.round(np.arange(0.05, 0.96, 0.025), 3)
        grid_lh = [(L, H) for L in gv for H in gv if H >= L]
        summ = []
        for crit in ['정확도', 'macro-F1', 'QWK']:
            bL, bH, bm, bv_ = None, None, None, -1
            for L, H in grid_lh:
                m = m_all(cls(L, H))
                if m[crit] > bv_:
                    bv_, bL, bH, bm = m[crit], L, H, m
            summ.append({'기준': crit, '최적 경계': f"[{bL:.2f}, {bH:.2f}]", '정확도': f"{bm['정확도']:.0%}",
                         'macro-F1': f"{bm['macro-F1']:.2f}", 'QWK': f"{bm['QWK']:.2f}", '심각오류': bm['심각오류']})
        st.dataframe(pd.DataFrame(summ), width="stretch", hide_index=True)

        qg = np.full((len(gv), len(gv)), np.nan)
        for a_, L in enumerate(gv):
            for b_, H in enumerate(gv):
                if H >= L:
                    qg[a_, b_] = m_all(cls(L, H))['QWK']
        box = st.container(border=True); box.markdown("**경계별 QWK** · 밝을수록 좋음 (대각선 위 삼각형만 유효)")
        fig = go.Figure(go.Heatmap(z=qg, x=[f"{h:.2f}" for h in gv], y=[f"{l:.2f}" for l in gv],
                                   colorscale='Viridis', colorbar=dict(title="QWK")))
        fig.update_layout(height=420, margin=dict(l=6, r=6, t=6, b=6),
                          xaxis_title="보통↔강 경계", yaxis_title="약↔보통 경계")
        box.plotly_chart(fig, width="stretch")
        st.caption("원하는 경계를 사이드바 range 슬라이더에 넣으면 전체 진단이 그 경계로 갱신됩니다. "
                   "이제 두 경계가 둘 다 0.5 미만/초과여도 탐색합니다.")
