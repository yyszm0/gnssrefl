"""
wang_cutoff_swh.py

- /etc/gnssrefl/refl_code/yyyy/snr/station/ 以下の .snr66(.gz) をすべて読み込む
- 衛星ごとのトラックを切り出す
- Wang et al. 流の wavelet 解析から
    * h_peak_index(e)  （高さインデックス）
    * e_cutoff_lower / e_cutoff_upper
    * ridge_level      （高仰角側の代表インデックス）
  を推定
- e_cutoff_upper → SWH の関数（ユーザが用意）を用いて SWH を推定
- 結果を DataFrame で返し、CSV にも保存
"""

import gzip
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt


# ---------- SNR66 読み込み（単一/ディレクトリ共通） ----------
def _read_snr_file(path: Path, snr_column: str = "L1") -> pd.DataFrame:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            data = np.loadtxt(f, comments="#")
    else:
        data = np.loadtxt(path, comments="#")

    sat = data[:, 0].astype(int)
    elev = data[:, 1]
    azim = data[:, 2]
    sod  = data[:, 3]
    snr_L1 = data[:, 6]
    snr_L2 = data[:, 7]
    snr_L5 = data[:, 8]

    if snr_column == "L1":
        snr = snr_L1
    elif snr_column == "L2":
        snr = snr_L2
    elif snr_column == "L5":
        snr = snr_L5
    else:
        raise ValueError("snr_column must be 'L1', 'L2', or 'L5'")

    df = pd.DataFrame(
        {
            "sat": sat,
            "elev_deg": elev,
            "az_deg": azim,
            "sod": sod,
            "snr_db": snr,
        }
    )
    return df


def load_snr(
    path_or_dir,
    snr_column: str = "L1",
    snr_type: str | None = None,   # "66", "88", "99" など
) -> pd.DataFrame:
    path = Path(path_or_dir)

    if path.is_file():
        df = _read_snr_file(path, snr_column=snr_column)
        df["day_index"] = 0
        return df

    if path.is_dir():
        # ★ ここが超重要
        if snr_type is None:
            pattern = "*.gz"
        else:
            pattern = f"*.snr{snr_type}.gz"

        print(f"pattern = {pattern}")
        files = sorted(path.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matching {pattern} in {path}")

        print("Reading snr files:")
        df_list = []
        for day_index, f in enumerate(files):
            print(f"  {f}")
            df_i = _read_snr_file(f, snr_column=snr_column)
            df_i["day_index"] = day_index
            df_list.append(df_i)
        df_all = pd.concat(df_list, ignore_index=True)
        return df_all

    raise FileNotFoundError(f"{path} is neither a file nor a directory")


def load_snr_station(
    base_dir,
    year,
    station,
    snr_column: str = "L1",
    snr_type: str | None = None,
) -> pd.DataFrame:
    base_dir = Path(base_dir)
    snr_dir = base_dir / str(year) / "snr" / station
    return load_snr(snr_dir, snr_column=snr_column, snr_type=snr_type)

# ---------- トラック分割 ----------

def split_into_passes(
    df,
    e_min=5.0,
    e_max=40.0,
    min_points=40,
    max_gap_sec=600.0,
):
    """
    SNR66 DataFrame を衛星ごとのトラック(arc)に分ける。

    - sat, day_index ごとに sod でソート
    - time gap > max_gap_sec で別トラック
    - 各トラックから e_min〜e_max のデータだけを使う
    """
    passes = []

    # day_index がなければ 0 とみなす
    if "day_index" not in df.columns:
        df = df.copy()
        df["day_index"] = 0

    for (sat_id, day_index), sdf in df.groupby(["sat", "day_index"]):
        sdf = sdf.sort_values("sod").reset_index(drop=True)

        start_idx = 0
        for i in range(1, len(sdf)):
            gap = sdf.loc[i, "sod"] - sdf.loc[i - 1, "sod"]
            if gap > max_gap_sec:
                segment = sdf.iloc[start_idx:i]
                _collect_pass_from_segment(
                    segment,
                    sat_id,
                    day_index,
                    e_min,
                    e_max,
                    min_points,
                    passes,
                )
                start_idx = i

        segment = sdf.iloc[start_idx:]
        _collect_pass_from_segment(
            segment,
            sat_id,
            day_index,
            e_min,
            e_max,
            min_points,
            passes,
        )

    return passes


def _collect_pass_from_segment(segment, sat_id, day_index, e_min, e_max, min_points, passes):
    if len(segment) < min_points:
        return

    mask = (segment["elev_deg"] >= e_min) & (segment["elev_deg"] <= e_max)
    seg2 = segment[mask]
    if len(seg2) < min_points:
        return

    passes.append(
        {
            "sat": sat_id,
            "day_index": day_index,
            "az_deg": float(seg2["az_deg"].mean()),
            "time_hour": float(seg2["sod"].mean() / 3600.0),
            "e_deg": seg2["elev_deg"].to_numpy(),
            "snr_db": seg2["snr_db"].to_numpy(),
        }
    )


# ---------- Wang の wavelet 部分（インデックス空間） ----------


def preprocess_snr(e_deg, snr_db, poly_order=2):
    """
    1衛星パス分の SNR を x = sin(e) に並べ替え & トレンド除去。
    """
    e_deg = np.asarray(e_deg)
    snr_db = np.asarray(snr_db)

    x = np.sin(np.deg2rad(e_deg))
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    snr_sorted = snr_db[sort_idx]
    e_sorted_deg = e_deg[sort_idx]

    coeffs = np.polyfit(x_sorted, snr_sorted, poly_order)
    trend = np.polyval(coeffs, x_sorted)
    snr_detr = snr_sorted - trend

    return x_sorted, snr_detr, e_sorted_deg


# ---------- Wavelet ヘルパー ----------


def build_cmor_name(
    wavelet_name: str | None = None,
    bandwidth: float | None = None,
    center_freq: float | None = None,
    default_name: str = "cmor1.5-1.0",
) -> str:
    """
    wavelet_name が指定されていればそれをそのまま返す。
    None の場合は bandwidth, center_freq から "cmorB-C" を作る。
    それも無ければ default_name を使う。
    """
    if wavelet_name is not None:
        return wavelet_name

    if bandwidth is not None and center_freq is not None:
        # PyWavelets の cmor 命名規則に合わせる
        return f"cmor{bandwidth}-{center_freq}"

    return default_name


def compute_wavelet_power_h_e(
    x,
    snr_detr,
    wavelength=0.19029367,
    h_min=0.5,
    h_max=20.0,
    num_h=80,
    wavelet_name: str | None = None,
    cmor_bandwidth: float | None = None,
    cmor_center: float | None = None,
):
    """
    SNR(x) に CWT をかけて、(h, e) パワーマップ P(h, e) を作る（h は[m]）。

    wavelet_name を直接指定するか、
    cmor_bandwidth, cmor_center から "cmorB-C" を構成する。
    """
    x = np.asarray(x)
    snr_detr = np.asarray(snr_detr)

    dx = np.mean(np.diff(x))  # x = sin(e) の刻み
    N = len(x)

    # スケールを計算しやすい範囲に制限
    s_min = 2.0
    s_max = min(N // 2, 256)
    if s_max <= s_min:
        s_max = s_min + 1.0

    scales = np.logspace(np.log10(s_min), np.log10(s_max), num_h)

    # wavelet 名を決定
    wavelet_name = build_cmor_name(
        wavelet_name=wavelet_name,
        bandwidth=cmor_bandwidth,
        center_freq=cmor_center,
        default_name="cmor1.5-1.0",
    )

    wavelet = pywt.ContinuousWavelet(wavelet_name)
    center_freq = pywt.central_frequency(wavelet)

    # sampling_period=dx として CWT
    coeffs, _ = pywt.cwt(
        snr_detr,
        scales=scales,
        wavelet=wavelet,
        sampling_period=dx,
    )
    power_all = np.abs(coeffs) ** 2  # shape = (num_scales, N)

    # 各スケールに対応する f_x [cycles per unit x], そこから h[m]
    f_x = center_freq / (scales * dx)
    h_all = 0.5 * wavelength * f_x

    # 欲しい h 範囲で切り取る
    mask = (h_all >= h_min) & (h_all <= h_max)
    if not np.any(mask):
        h_grid = h_all
        power = power_all
    else:
        h_grid = h_all[mask]
        power = power_all[mask, :]

    return h_grid, power


def extract_h_peak_per_e(h_grid, power):
    """
    各 e（列）ごとに最大パワーの h[m] を返す。
    """
    peak_idx = np.argmax(power, axis=0)  # shape (num_e,)
    h_peak = h_grid[peak_idx]
    return h_peak


def compute_coh_inco_power_per_e(
    h_grid,
    power,
    rh_ref,
    coh_half_width=2.0,  # RH_ref ± 2 m を coherent 帯
    inco_gap=4.0,  # RH_ref - 4 m 以下を incoherent 帯
):
    """
    各仰角 e ごとに、
        - coherent 帯 (rh_ref ± coh_half_width) のパワー
        - incoherent 帯 (h <= rh_ref - inco_gap) のパワー
    を求める。
    """
    h = np.asarray(h_grid)

    # coherent 帯
    coh_min = rh_ref - coh_half_width
    coh_max = rh_ref + coh_half_width
    coh_mask = (h >= coh_min) & (h <= coh_max)

    # incoherent 帯（低い方）
    inco_max = rh_ref - inco_gap
    inco_mask = h <= inco_max

    if not np.any(coh_mask):
        # 保険：coherent 帯が空なら、全体 max を coherent とみなす
        power_coh = power.max(axis=0)
    else:
        power_coh = power[coh_mask, :].sum(axis=0)

    if not np.any(inco_mask):
        power_inco = np.zeros(power.shape[1])
    else:
        power_inco = power[inco_mask, :].sum(axis=0)

    return power_coh, power_inco


def estimate_cutoff_from_h_peak_step(
    e_deg_sorted,
    h_peak,
    h_high=9.0,  # 「RHが9m以上」の閾値
    h_low=5.0,  # 「5m未満」の閾値
):
    """
    h_peak(e) から e_cutoff_lower / e_cutoff_upper を決める単純なロジック。
    """
    e = np.asarray(e_deg_sorted)
    h = np.asarray(h_peak)

    if len(e) < 2:
        return None, None

    state = np.full(len(e), "M", dtype="<U1")
    state[h >= h_high] = "H"
    state[h <= h_low] = "L"

    cut_idxs = []
    for i in range(1, len(e)):
        s0, s1 = state[i - 1], state[i]
        if (s0 == "H" and s1 == "L") or (s0 == "L" and s1 == "H"):
            cut_idxs.append(i)

    if not cut_idxs:
        return None, None

    i_min = min(cut_idxs)
    i_max = max(cut_idxs)
    e_cutoff_lower = float(e[i_min])
    e_cutoff_upper = float(e[i_max])

    return e_cutoff_lower, e_cutoff_upper


def estimate_cutoff_from_power(
    e_deg_sorted,
    h_grid,
    power,
    rh_input=None,
    high_frac=0.3,
    coh_half_width=2.0,
    inco_gap=4.0,
    ratio_thresh=1.0,
    min_run_len=2,
):
    """
    2次元パワーから e_cutoff_lower, e_cutoff_upper を判定する。
    """
    e = np.asarray(e_deg_sorted)

    # まず h_peak を出して RH_ref 推定に使う
    h_peak = extract_h_peak_per_e(h_grid, power)

    if rh_input is None:
        # 高仰角側から RH_ref を推定
        q = np.percentile(e, 100 * (1.0 - high_frac))
        mask_high = e >= q
        if np.any(mask_high):
            rh_ref = float(np.median(h_peak[mask_high]))
        else:
            rh_ref = float(np.median(h_peak))
    else:
        rh_ref = float(rh_input)

    # coherent / incoherent パワー
    Pc, Pi = compute_coh_inco_power_per_e(
        h_grid,
        power,
        rh_ref,
        coh_half_width=coh_half_width,
        inco_gap=inco_gap,
    )

    # パワー比
    eps = 1e-6
    ratio = Pi / (Pc + eps)

    # 「incoherent が優勢な領域」をフラグ
    bad = ratio >= ratio_thresh
    if not np.any(bad):
        return None, None, rh_ref, Pc, Pi, ratio

    idx = np.where(bad)[0]

    # 連続区間を列挙
    runs = []
    start = idx[0]
    prev = idx[0]
    for k in idx[1:]:
        if k == prev + 1:
            prev = k
        else:
            runs.append((start, prev))
            start = k
            prev = k
    runs.append((start, prev))

    # 一番長い区間だけを cut-off 領域として採用
    runs = [r for r in runs if (r[1] - r[0] + 1) >= min_run_len]
    if not runs:
        return None, None, rh_ref, Pc, Pi, ratio

    i0, i1 = max(runs, key=lambda ab: ab[1] - ab[0] + 1)
    e_segment = e[i0 : i1 + 1]

    e_cutoff_lower = float(e_segment.min())
    e_cutoff_upper = float(e_segment.max())

    return e_cutoff_lower, e_cutoff_upper, rh_ref, Pc, Pi, ratio


def bin_h_peak_in_elevation(e_deg_sorted, h_peak, bin_width_deg=1.0, min_count=3):
    """
    仰角 e_deg_sorted に対する h_peak を、bin_width_deg ごとにビン平均する。
    """
    e = np.asarray(e_deg_sorted)
    h = np.asarray(h_peak)

    e_min = e.min()
    e_max = e.max()

    # ビン境界（右端を少し余裕を持たせて作る）
    bins = np.arange(e_min, e_max + bin_width_deg, bin_width_deg)

    # それぞれのサンプルがどのビンに入るか
    idx = np.digitize(e, bins)

    e_coarse = []
    h_coarse = []

    for b in range(1, len(bins) + 1):
        mask = idx == b
        if np.sum(mask) >= min_count:
            e_coarse.append(np.mean(e[mask]))
            h_coarse.append(np.mean(h[mask]))

    return np.array(e_coarse), np.array(h_coarse)


def estimate_cutoff_coh_inco_adaptive(
    e_deg_sorted,
    h_peak,
    high_frac=0.3,  # 高仰角側の割合
    coh_half_width=2.0,  # coherent 帯の ±幅[m]
    inco_gap=4.0,  # RH_ref からこれだけ低いところを incoherent 帯に
    min_run_len=2,
):
    e = np.asarray(e_deg_sorted)
    h = np.asarray(h_peak)
    if len(e) == 0:
        return None, None, None

    # 高仰角側から RH_ref を求める
    q = np.percentile(e, 100 * (1.0 - high_frac))
    mask_high = e >= q
    if np.any(mask_high):
        rh_ref = float(np.median(h[mask_high]))
    else:
        rh_ref = float(np.median(h))

    # 自動決定された帯
    h_coh_min = rh_ref - coh_half_width
    h_coh_max = rh_ref + coh_half_width
    h_inco_max = rh_ref - inco_gap

    coh_mask = (h >= h_coh_min) & (h <= h_coh_max)
    inco_mask = h <= h_inco_max

    state = np.full(len(e), "N", dtype="<U1")
    state[coh_mask] = "C"
    state[inco_mask] = "I"

    def find_transition(prev_state, next_state):
        idxs = []
        for i in range(1, len(e)):
            if state[i - 1] == prev_state and state[i] == next_state:
                idxs.append(i)
        if not idxs:
            return None
        # 連続区間チェック
        runs = []
        start = idxs[0]
        prev = idxs[0]
        for k in idxs[1:]:
            if k == prev + 1:
                prev = k
            else:
                runs.append((start, prev))
                start = k
                prev = k
        runs.append((start, prev))
        for s, t in runs:
            if (t - s + 1) >= min_run_len:
                return s
        return None

    idx_lower = find_transition("C", "I")
    idx_upper = find_transition("I", "C")

    e_cutoff_lower = float(e[idx_lower]) if idx_lower is not None else None
    e_cutoff_upper = float(e[idx_upper]) if idx_upper is not None else None

    return e_cutoff_lower, e_cutoff_upper, rh_ref


def estimate_cutoff_angles_from_h_peak_index(
    e_deg_sorted,
    h_peak_index,
    high_frac=0.3,
    jump_thresh_index=3,
    min_run_length=3,
):
    """
    h_peak_index(e) から e_cutoff_lower / e_cutoff_upper を推定（改訂版）。
    """
    e = np.asarray(e_deg_sorted)
    h_idx = np.asarray(h_peak_index)

    if len(e) == 0:
        return None, None, None

    # 高仰角側の plateau を RH レベルとみなす
    q = np.percentile(e, 100 * (1.0 - high_frac))  # 例: 上位 30%
    mask_high = e >= q
    if not np.any(mask_high):
        rh_level = float(np.median(h_idx))
    else:
        rh_level = float(np.median(h_idx[mask_high]))

    # 「RH より十分低く落ちた点」だけを見る
    diff_down = rh_level - h_idx  # 正なら RH より低い
    bad = diff_down >= jump_thresh_index

    if not np.any(bad):
        # 落ち込みが検出できない場合は cut-off なし
        return None, None, rh_level

    # bad が連続している区間を検出
    idx_bad = np.where(bad)[0]

    runs = []
    start = idx_bad[0]
    prev = idx_bad[0]
    for k in idx_bad[1:]:
        if k == prev + 1:
            prev = k
        else:
            runs.append((start, prev))
            start = k
            prev = k
    runs.append((start, prev))

    # 一番長い区間を採用
    best_run = max(runs, key=lambda ab: ab[1] - ab[0] + 1)

    if best_run[1] - best_run[0] + 1 < min_run_length:
        # 区間が短すぎるなら cut-off なし扱い
        return None, None, rh_level

    i0, i1 = best_run
    e_segment = e[i0 : i1 + 1]

    e_cutoff_lower = float(np.min(e_segment))
    e_cutoff_upper = float(np.max(e_segment))

    return e_cutoff_lower, e_cutoff_upper, rh_level


def estimate_cutoff_angles_from_h_peak(
    e_deg_sorted,
    h_peak,
    rh_input=None,  # 入力RHがあればここに[m]で渡せる
    high_frac=0.3,  # 自動推定する場合: 上位30%の仰角からRHを推定
    threshold_m=6.0,  # 「RH±3 m」ルール
):
    """
    h_peak(e)[m] から e_cutoff_lower / e_cutoff_upper を推定。
    """
    e = np.asarray(e_deg_sorted)
    h = np.asarray(h_peak)

    if len(e) == 0:
        return None, None, None

    # RH の決め方
    if rh_input is None:
        q = np.percentile(e, 100 * (1.0 - high_frac))  # 例: 上位30%仰角
        mask_high = e >= q
        if np.any(mask_high):
            rh_ref = float(np.median(h[mask_high]))
        else:
            rh_ref = float(np.median(h))
    else:
        rh_ref = float(rh_input)

    diff = np.abs(h - rh_ref)
    outside = diff > threshold_m

    if not np.any(outside):
        # 全部 RH±3m に収まっている → cut-off なし
        return None, None, rh_ref

    e_bad = e[outside]
    e_cutoff_lower = float(np.min(e_bad))
    e_cutoff_upper = float(np.max(e_bad))

    return e_cutoff_lower, e_cutoff_upper, rh_ref


def process_single_pass(
    e_deg,
    snr_db,
    wavelength=0.19029367,
    h_min=0.5,
    h_max=20.0,
    num_h=80,
    wavelet_name: str | None = "cmor9.0-1.0",
    cmor_bandwidth: float | None = None,
    cmor_center: float | None = None,
    plot=False,
    rh_input=None,  # 使うなら別途
    e_plot_range=None,   # 追加 (例: (5.0, 40.0))
):
    """
    1パス分のデータに対して wavelet 解析を実行し、
    h_peak(e) と cut-off などを求める。
    """
    # 1) 前処理
    x, snr_detr, e_sorted = preprocess_snr(e_deg, snr_db)

    # 2) CWT → (h[m], e) パワーマップ
    h_grid, power = compute_wavelet_power_h_e(
        x,
        snr_detr,
        wavelength=wavelength,
        h_min=h_min,
        h_max=h_max,
        num_h=num_h,
        wavelet_name=wavelet_name,
        cmor_bandwidth=cmor_bandwidth,
        cmor_center=cmor_center,
    )

    # 3) h_peak(e)
    h_peak = extract_h_peak_per_e(h_grid, power)

    # 4) h_peak の「9m↔5m」遷移から cut-off を決める
    e_cut_low, e_cut_up = estimate_cutoff_from_h_peak_step(
        e_sorted,
        h_peak,
        h_high=9.0,
        h_low=5.0,
    )

    # RH は「高仰角側の median」で推定、あるいは rh_input を使う
    if rh_input is None:
        q = np.percentile(e_sorted, 70)  # 上位30% 仰角より大きい領域
        mask_high = e_sorted >= q
        if np.any(mask_high):
            rh_ref = float(np.median(h_peak[mask_high]))
        else:
            rh_ref = float(np.median(h_peak))
    else:
        rh_ref = float(rh_input)

    if plot:
        # wavelet power
        plt.figure(figsize=(6, 4))
        plt.pcolormesh(e_sorted, h_grid, power, shading="auto")
        plt.xlabel("Elevation [deg]")
        plt.ylabel("Reflector height h [m]")
        plt.title(f"Wavelet power |W|^2 ({wavelet_name})")
        plt.colorbar(label="Power")
        plt.axhline(rh_ref, color="w", linestyle="--", label="RH")
        plt.xlim(5.0, 40.0)          # ★ ここで Elevation のレンジを固定
        plt.legend()
        plt.show()

        # h_peak と cut-off
        plt.figure()
        plt.plot(e_sorted, h_peak, ".", label="h_peak(e)")
        plt.axhline(9.0, color="k", linestyle="--", label="9 m")
        plt.axhline(5.0, color="k", linestyle=":", label="5 m")
        if e_cut_low is not None:
            plt.axvline(e_cut_low, color="r", linestyle="--", label="e_cutoff_lower")
        if e_cut_up is not None:
            plt.axvline(e_cut_up, color="g", linestyle="--", label="e_cutoff_upper")
        plt.xlabel("Elevation [deg]")
        plt.ylabel("h_peak [m]")
        plt.xlim(5.0, 40.0)          # ★ こっちの図にも同じように
        plt.legend()
        plt.title("h_peak(e) and cutoff (9m ↔ 5m rule)")
        plt.show()

    return {
        "e_deg_sorted": e_sorted,
        "h_peak": h_peak,
        "h_grid": h_grid,
        "power": power,
        "e_cutoff_lower": e_cut_low,
        "e_cutoff_upper": e_cut_up,
        "rh_ref": rh_ref,
    }


# ---------- SWH 推定 & ステーション全体実行 ----------


def estimate_swh_for_pass(
    e_cutoff_upper_deg,
    azimuth_deg,
    T_rep,
    func_e_to_swh,
):
    """
    e_cutoff,upper から SWH を求めるラッパー。

    func_e_to_swh はユーザが用意する:
        swh = func_e_to_swh(e_cutoff_upper_deg, azimuth_deg, T_rep)
    """
    if e_cutoff_upper_deg is None or np.isnan(e_cutoff_upper_deg):
        return np.nan
    return float(func_e_to_swh(e_cutoff_upper_deg, azimuth_deg, T_rep))


def process_station_for_swh(
    year,
    station,
    func_e_to_swh,
    base_dir="/etc/gnssrefl/refl_code",
    wavelength=0.19029367,
    T_rep=50.0,
    snr_column="L1",
    e_min=5.0,
    e_max=40.0,
    min_points=40,
    max_gap_sec=600.0,
    h_min=0.5,
    h_max=20.0,
    num_h=80,
    rh_level=None,
    output_csv="swh_estimates.csv",
    debug=False,
    use_binning=False,  # まだ未使用だがインターフェース維持
    # wavelet 関連
    wavelet_name: str | None = None,
    cmor_bandwidth: float | None = None,
    cmor_center: float | None = None,
    snr_type: str | None = None,   # 例: "66", "88", "99"
):
    """
    /etc/gnssrefl/refl_code/yyyy/snr/station/ 以下の snr66(.gz) をすべて読み込み、
    cut-off & ridge_level & SWH を推定して CSV に保存する。
    """
    t_all0 = time.perf_counter()

    df_snr = load_snr_station(
        base_dir=base_dir,
        year=year,
        station=station,
        snr_column=snr_column,
        snr_type=snr_type,
    )
    t1 = time.perf_counter()
    if debug:
        print(f"load_snr66_station: {len(df_snr)} rows, took {t1 - t_all0:.3f} s")

    passes = split_into_passes(
        df_snr,
        e_min=e_min,
        e_max=e_max,
        min_points=min_points,
        max_gap_sec=max_gap_sec,
    )
    t2 = time.perf_counter()
    if debug:
        print(f"split_into_passes: {len(passes)} passes, took {t2 - t1:.3f} s")

    records = []
    t_pass_total = 0.0
    for idx, p in enumerate(passes):
        t0 = time.perf_counter()
        res = process_single_pass(
            p["e_deg"],
            p["snr_db"],
            wavelength=wavelength,
            h_min=h_min,
            h_max=h_max,
            num_h=num_h,
            wavelet_name=wavelet_name,
            cmor_bandwidth=cmor_bandwidth,
            cmor_center=cmor_center,
            plot=False,
            rh_input=rh_level,
        )
        t1 = time.perf_counter()
        t_pass = t1 - t0
        t_pass_total += t_pass
        if debug and (idx < 5 or idx % 10 == 0):
            print(f"  pass {idx+1}/{len(passes)} took {t_pass:.3f} s")

        e_cut_low = res["e_cutoff_lower"]
        e_cut_up = res["e_cutoff_upper"]
        rh_ref = res["rh_ref"]

        swh = estimate_swh_for_pass(
            e_cutoff_upper_deg=e_cut_up,
            azimuth_deg=p["az_deg"],
            T_rep=T_rep,
            func_e_to_swh=func_e_to_swh,
        )

        records.append(
            {
                "year": year,
                "station": station,
                "day_index": p["day_index"],
                "sat": p["sat"],
                "time_hour": p["time_hour"],
                "azimuth_deg": p["az_deg"],
                "e_cutoff_lower_deg": e_cut_low,
                "e_cutoff_upper_deg": e_cut_up,
                "rh_ref_m": rh_ref,
                "swh_m": swh,
            }
        )

    df_out = pd.DataFrame(records)
    df_out.to_csv(output_csv, index=False)

    t_all1 = time.perf_counter()
    if debug:
        print(f"total pass processing time: {t_pass_total:.3f} s")
        print(f"TOTAL process_station_for_swh: {t_all1 - t_all0:.3f} s")

    return df_out
