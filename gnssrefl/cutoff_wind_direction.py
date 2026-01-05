import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def estimate_wind_direction_from_cutoff_lower(df_swh):
    df_valid = df_swh.dropna(subset=["e_cutoff_lower_deg", "azimuth_deg"])
    if df_valid.empty:
        return None
    idx_max = df_valid["e_cutoff_lower_deg"].idxmax()
    return float(df_valid.loc[idx_max, "azimuth_deg"])


def fit_ellipse_like_curve(df, n_points=360):
    """
    df は既に azimuth マスク済みのものを渡す想定。
    """
    df_valid = df.dropna(subset=["e_cutoff_lower_deg", "azimuth_deg"])
    if df_valid.empty:
        return None, None

    theta = np.deg2rad(df_valid["azimuth_deg"].to_numpy())
    r = df_valid["e_cutoff_lower_deg"].to_numpy()

    A = np.column_stack(
        [
            np.ones_like(theta),
            np.cos(theta),
            np.sin(theta),
            np.cos(2 * theta),
            np.sin(2 * theta),
        ]
    )
    coeff, *_ = np.linalg.lstsq(A, r, rcond=None)

    theta_fit = np.linspace(0, 2 * np.pi, n_points)
    A_fit = np.column_stack(
        [
            np.ones_like(theta_fit),
            np.cos(theta_fit),
            np.sin(theta_fit),
            np.cos(2 * theta_fit),
            np.sin(2 * theta_fit),
        ]
    )
    r_fit = A_fit @ coeff
    r_fit[r_fit < 0] = 0.0

    return theta_fit, r_fit


def _apply_azimuth_mask(df, az_mask):
    """
    az_mask: None または [(min_deg, max_deg), ...] のリスト。
    min<=max のときは通常区間、
    min>max のときは 360°またぎ (例: (300, 60) で [300,360]∪[0,60])。
    """
    if az_mask is None:
        return df

    az = df["azimuth_deg"].to_numpy()
    mask_all = np.zeros_like(az, dtype=bool)

    for a_min, a_max in az_mask:
        if a_min <= a_max:
            mask = (az >= a_min) & (az <= a_max)
        else:
            # wrap around 360°
            mask = (az >= a_min) | (az <= a_max)
        mask_all |= mask

    return df[mask_all]


def plot_wind_polar(
    df_swh,
    wind_speed=None,
    fig_path=None,          # None なら保存しない
    title_prefix="",
    show=True,              # True なら plt.show()
    az_mask=None,           # 例: [(0,180)] とか [(200,260),(280,320)]
    r_max_fixed=45.0,       # 衛星仰角のlimを固定したいのでデフォルト45°
    enable_ellipse=True,    # ★ 楕円フィット描画をするか
    enable_direction=True,  # ★ 風向の推定＋矢印を出すか
):
    """
    Wang 図っぽい極座標図を描く。

    - az_mask が指定されていれば、その azimuth 範囲内のパスだけ使用。
    - 半径（r）はデフォルトで 0–45° に固定。
    """
    # 有効データのみ
    df = df_swh.dropna(subset=["e_cutoff_lower_deg", "azimuth_deg"]).copy()
    if df.empty:
        raise ValueError("No valid e_cutoff_lower data")

    # azimuth マスクを適用
    df = _apply_azimuth_mask(df, az_mask)
    if df.empty:
        raise ValueError("No data left after azimuth mask")

    # 風向（mask 後のデータで推定）
    if enable_direction:
        wind_dir_deg = estimate_wind_direction_from_cutoff_lower(df)
    else:
        wind_dir_deg = None
    theta_wind = np.deg2rad(wind_dir_deg) if wind_dir_deg is not None else None

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    theta = np.deg2rad(df["azimuth_deg"].to_numpy())
    r = df["e_cutoff_lower_deg"].to_numpy()

    ax.scatter(theta, r, marker="s", color="k", label="Cutoff lower")

    # 楕円っぽい曲線（mask 後の df を使う）
    if enable_ellipse:
        theta_fit, r_fit = fit_ellipse_like_curve(df)
    else:
        theta_fit, r_fit = None, None

    if theta_fit is not None:
        ax.plot(theta_fit, r_fit, color="red", linewidth=2.0, label="Fitted curve")
        idx_max = np.argmax(r_fit)
        theta_max = theta_fit[idx_max]
        ax.plot(
            [theta_max, theta_max + np.pi],
            [0, np.max(r_fit)],
            color="red",
            linestyle="--",
            linewidth=1.0,
        )

    # 半径の上限を 0–r_max_fixed で固定
    if r_max_fixed is not None:
        r_lim_max = r_max_fixed
    else:
        r_lim_max = max(
            np.max(r),
            np.max(r_fit) if r_fit is not None else 0
        ) * 1.1

    # 風向ベクトル
    if theta_wind is not None:
        ax.plot(
            [theta_wind, theta_wind],
            [0, r_lim_max],
            color="blue",
            linewidth=2.5,
        )
        txt = f"Direction:{wind_dir_deg:.0f}°"
        if wind_speed is not None:
            txt += f"\nSpeed:{wind_speed:.1f} m/s"
        ax.text(
            theta_wind,
            r_lim_max * 1.05,
            txt,
            color="blue",
            fontsize=10,
            ha="left",
            va="bottom",
        )

    ax.set_rlabel_position(225)
    ax.set_ylim(0, r_lim_max)
    ax.set_title(f"{title_prefix}Elevation cutoff", va="bottom")
    ax.grid(True)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()

    if fig_path is not None:
        plt.savefig(fig_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return wind_dir_deg
