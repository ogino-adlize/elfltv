# -*- coding: utf-8 -*-
"""
ecforce 受注CSVから「回数別継続率」「累積LTV推移」「回数別許容CPA」を
可視化・シミュレーションするStreamlitアプリ（拡張版）

起動方法:
    streamlit run app.py

パスワード設定:
    .streamlit/secrets.toml に以下を記載してください
        APP_PASSWORD = "任意のパスワード"
"""

import io
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# =========================================================
# 定数（ecforce 受注一覧CSVの列名）
# =========================================================
COL_ORDER_ID = "受注ID"
COL_CUSTOMER_ID = "顧客ID"
COL_SUB_ID = "定期受注番号"
COL_COUNT = "定期回数"
COL_PURCHASE_URL = "購入URL"
COL_AD_GROUP = "広告URLグループ名"
COL_STATUS = "対応状況"
COL_DATE = "作成日"
COL_SUBTOTAL = "小計"

# オプション列（あれば自動検出して機能を有効化する）
COL_PRODUCT_RAW = "購入商品（商品名）"       # カンマ区切りで同梱物込みの生データ
COL_PRODUCT_MAIN = "商品(主力)"             # ↑から主力商品だけを抽出した派生列
COL_PAYMENT_STATUS = "決済状況"
COL_SUB_STATUS = "定期ステータス"
COL_PAYMENT_METHOD = "支払い方法"

REQUIRED_COLUMNS = [
    COL_ORDER_ID, COL_CUSTOMER_ID, COL_SUB_ID, COL_COUNT,
    COL_PURCHASE_URL, COL_AD_GROUP, COL_STATUS, COL_DATE, COL_SUBTOTAL,
]

# 商品名らしき列の候補（存在すれば自動検出してフィルタに使う。派生の主力商品列を最優先）
PRODUCT_COL_CANDIDATES = [COL_PRODUCT_MAIN, "商品名", "商品ID", "品番", "商品コード", "商品グループ名"]

# 明らかに「無効な受注」とみなすステータス（デフォルトで除外）
DEFAULT_EXCLUDE_STATUS = ["キャンセル", "決済NG", "返送", "重複注文", "確認中（注文後）", "不明なステータス"]
# 「不良受注」とみなすステータス（品質分析タブ用）
DEFECT_STATUS = ["決済NG", "キャンセル", "返送", "重複注文"]
# 決済状況のうち、明らかに「無効」とみなす値（デフォルトで除外）
DEFAULT_EXCLUDE_PAYMENT_STATUS = [
    "取消完了", "取引修正失敗", "与信審査エラー", "仮売上失敗", "取引登録失敗", "取引キャンセル失敗", "決済エラー", "入金待ち",
]
# 同梱物・販促物など「主力商品ではない」ことを示すブラケットタグのキーワード
FREEBIE_BRACKET_KEYWORDS = ["共通"]

st.set_page_config(
    page_title="定期通販 継続率 / LTV / 許容CPA シミュレーター",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 認証
# =========================================================
def check_password() -> bool:
    def password_entered():
        correct = st.secrets.get("APP_PASSWORD", None)
        if correct is None:
            correct = "changeme"
        if st.session_state.get("password_input") == correct:
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.title("🔒 ログイン")
    st.text_input(
        "パスワードを入力してください",
        type="password",
        key="password_input",
        on_change=password_entered,
    )
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("パスワードが違います。")

    if "APP_PASSWORD" not in st.secrets:
        st.caption(
            "⚠️ st.secrets に APP_PASSWORD が設定されていません。"
            "現在は仮パスワード「changeme」でログインできます。"
            "本番公開前に .streamlit/secrets.toml を設定してください。"
        )
    return False


if not check_password():
    st.stop()


# =========================================================
# データ読み込み
# =========================================================
def extract_main_product(name_str):
    """
    「【定期】NECK LESS,ブランドブック,【共通】初回挨拶状」のようなカンマ区切りの中から、
    同梱物・販促物（【共通】〜など）を除いた主力商品名を1つ抽出する。
    該当なしの場合は先頭要素をそのまま返す。
    """
    if pd.isna(name_str):
        return None
    tokens = [t.strip() for t in str(name_str).split(",") if t.strip()]
    for t in tokens:
        m = re.match(r"【([^】]+)】", t)
        if m and not any(k in m.group(1) for k in FREEBIE_BRACKET_KEYWORDS):
            return t
    return tokens[0] if tokens else None


@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    last_err = None
    df = None
    for enc in ("cp932", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            df = None
    if df is None:
        raise ValueError(f"CSVの読み込みに失敗しました: {last_err}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"必要な列が見つかりません: {missing}")

    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    df[COL_COUNT] = pd.to_numeric(df[COL_COUNT], errors="coerce")
    df[COL_SUBTOTAL] = pd.to_numeric(df[COL_SUBTOTAL], errors="coerce").fillna(0)
    df = df.dropna(subset=[COL_COUNT, COL_DATE])
    df[COL_COUNT] = df[COL_COUNT].astype(int)

    if COL_PRODUCT_RAW in df.columns:
        df[COL_PRODUCT_MAIN] = df[COL_PRODUCT_RAW].apply(extract_main_product)

    return df


def detect_product_col(df: pd.DataFrame):
    for c in PRODUCT_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


# =========================================================
# コア集計ロジック
# =========================================================
@st.cache_data(show_spinner=False)
def compute_cohort(df_valid: pd.DataFrame, sub_ids: tuple, max_round: int):
    """denom件数を分母に、回数別の継続率・累積LTV・累積受注回数を算出"""
    d = df_valid[df_valid[COL_SUB_ID].isin(sub_ids)]
    denom = len(sub_ids)

    pivot = d.pivot_table(index=COL_SUB_ID, columns=COL_COUNT, values=COL_SUBTOTAL, aggfunc="sum")
    pivot = pivot.reindex(index=list(sub_ids), columns=range(1, max_round + 1))

    retention = pivot.notna().sum(axis=0) / denom * 100
    order_flag = pivot.notna()
    filled = pivot.fillna(0)

    avg_cum_ltv = filled.cumsum(axis=1).mean(axis=0)
    avg_cum_orders = order_flag.cumsum(axis=1).mean(axis=0)

    return retention, avg_cum_ltv, avg_cum_orders, denom


def simulate_profit(
    avg_cum_ltv, avg_cum_orders, cost_mode, cost_value, ship_cost, payment_fee_rate, target_profit_rate
):
    """
    cost_mode: "rate"  -> cost_value は原価率(0〜1)。売上（累積LTV）に比例した原価を計算。
               "fixed" -> cost_value は1回あたりの原価(円)。受注回数に比例した原価を計算（配送料と同じ考え方）。
    """
    if cost_mode == "fixed":
        cum_cost = avg_cum_orders * cost_value
    else:
        cum_cost = avg_cum_ltv * cost_value
    cum_ship = avg_cum_orders * ship_cost
    cum_payment_fee = avg_cum_ltv * payment_fee_rate
    cum_gross_profit = avg_cum_ltv - cum_cost - cum_ship - cum_payment_fee
    target_profit = avg_cum_ltv * target_profit_rate
    allowable_cpa = cum_gross_profit - target_profit
    return cum_cost, cum_ship, cum_payment_fee, cum_gross_profit, target_profit, allowable_cpa


def build_first_orders(valid_df, date_from, date_to, product_col, product_selected):
    fo = valid_df[valid_df[COL_COUNT] == 1].copy()
    fo = fo[(fo[COL_DATE].dt.date >= date_from) & (fo[COL_DATE].dt.date <= date_to)]
    if product_col and product_selected is not None:
        fo = fo[fo[product_col].isin(product_selected)]
    return fo


@st.cache_data(show_spinner=False)
def compute_multi_group_metrics(valid_df: pd.DataFrame, fo: pd.DataFrame, group_col: str, max_round: int):
    """グループ列（広告URLグループ名 / 購入URL など）ごとに継続率・LTV・受注回数を算出"""
    results = {}
    for g, sub in fo.groupby(group_col):
        ids = tuple(sub[COL_SUB_ID].unique())
        if len(ids) == 0:
            continue
        retention, avg_cum_ltv, avg_cum_orders, denom = compute_cohort(valid_df, ids, max_round)
        results[g] = {
            "retention": retention,
            "avg_cum_ltv": avg_cum_ltv,
            "avg_cum_orders": avg_cum_orders,
            "denom": denom,
        }
    return results


def compute_quality_by_group(fo_all_status: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """ステータス問わず全初回受注を対象に、広告グループ別の不良受注率を算出"""
    total = fo_all_status.groupby(group_col).size().rename("総初回受注数")
    defect_mask = fo_all_status[COL_STATUS].isin(DEFECT_STATUS)
    defect = fo_all_status[defect_mask].groupby(group_col).size().rename("不良受注数")
    q = pd.concat([total, defect], axis=1).fillna(0)
    q["不良受注数"] = q["不良受注数"].astype(int)
    q["不良率(%)"] = (q["不良受注数"] / q["総初回受注数"] * 100).round(1)
    for s in DEFECT_STATUS:
        cnt = fo_all_status[fo_all_status[COL_STATUS] == s].groupby(group_col).size()
        q[f"{s}率(%)"] = (cnt / q["総初回受注数"] * 100).round(1)
    q[[f"{s}率(%)" for s in DEFECT_STATUS]] = q[[f"{s}率(%)" for s in DEFECT_STATUS]].fillna(0)
    return q.sort_values("不良率(%)", ascending=False)


# =========================================================
# メイン画面
# =========================================================
st.title("📊 定期通販 継続率 / 累積LTV / 許容CPA シミュレーター")
st.caption("ecforceの「受注一覧CSV」をアップロードして分析します。")

uploaded_file = st.file_uploader("受注一覧CSVをアップロード", type=["csv"])

if uploaded_file is None:
    st.info("CSVファイルをアップロードすると分析が始まります。")
    st.stop()

try:
    df = load_csv(uploaded_file.getvalue())
except ValueError as e:
    st.error(str(e))
    st.stop()

product_col = detect_product_col(df)
st.success(
    f"読み込み完了：{len(df):,} 行"
    + (f"　／　商品列を検出：「{product_col}」" if product_col else "　／　商品名などの列は検出されませんでした")
)

# ---------------------------------------------------------
# サイドバー：期間・広告URL・商品・有効ステータスの絞り込み
# ---------------------------------------------------------
st.sidebar.header("🔍 絞り込み条件")

first_orders_all = df[df[COL_COUNT] == 1]
min_date = first_orders_all[COL_DATE].min().date()
max_date = first_orders_all[COL_DATE].max().date()

st.sidebar.subheader("初回受注日時（期間）")
date_from, date_to = st.sidebar.date_input(
    "From - To", value=(min_date, max_date), min_value=min_date, max_value=max_date,
)

st.sidebar.subheader("広告URLの絞り込み")
ad_col_choice = st.sidebar.radio("絞り込み・比較に使う列", [COL_AD_GROUP, COL_PURCHASE_URL], horizontal=True)
ad_options = sorted(df[ad_col_choice].dropna().unique().tolist())
select_all_ad = st.sidebar.checkbox("すべて選択", value=True, key="select_all_ad")
ad_selected = st.sidebar.multiselect(ad_col_choice, options=ad_options, default=ad_options if select_all_ad else [])

product_selected = None
if product_col:
    st.sidebar.subheader(f"商品の絞り込み（{product_col}）")
    product_options = sorted(df[product_col].dropna().unique().tolist())
    select_all_product = st.sidebar.checkbox("すべて選択", value=True, key="select_all_product")
    product_selected = st.sidebar.multiselect(
        product_col, options=product_options, default=product_options if select_all_product else []
    )
else:
    st.sidebar.info("商品名・商品ID等の列がCSVにないため、商品絞り込みは利用できません。")

st.sidebar.subheader("有効受注とみなす対応状況")
status_options = sorted(df[COL_STATUS].dropna().unique().tolist())
default_status = [s for s in status_options if s not in DEFAULT_EXCLUDE_STATUS]
valid_status_selected = st.sidebar.multiselect(
    "対応状況（売上確定系のみチェック推奨）", options=status_options, default=default_status,
)

payment_status_selected = None
if COL_PAYMENT_STATUS in df.columns:
    st.sidebar.subheader("有効受注とみなす決済状況")
    payment_status_options = sorted(df[COL_PAYMENT_STATUS].dropna().unique().tolist())
    default_payment_status = [s for s in payment_status_options if s not in DEFAULT_EXCLUDE_PAYMENT_STATUS]
    payment_status_selected = st.sidebar.multiselect(
        "決済状況（対応状況とAND条件で有効受注を判定）", options=payment_status_options, default=default_payment_status,
    )

sub_status_mode = "すべて含める（進行中も含む速報値）"
if COL_SUB_STATUS in df.columns:
    st.sidebar.subheader("継続率の集計対象")
    sub_status_mode = st.sidebar.radio(
        "対象とする定期受注",
        ["すべて含める（進行中も含む速報値）", "終了した定期のみ（有効中は除外し確定値）"],
        help="「有効」＝まだ解約されていない進行中の定期購入。含めると直近コホートの継続率が実態より低く出る場合があります。",
    )

# ---------------------------------------------------------
# シミュレーター入力
# ---------------------------------------------------------
st.subheader("⚙️ 原価・利益シミュレーター")

cost_mode_label = st.radio(
    "原価の入力方法",
    ["原価率 (%) で計算（売上に比例）", "1回あたりの原価 (円) で計算（配送料と同じ固定額方式）"],
    horizontal=True,
)
cost_mode = "fixed" if cost_mode_label.startswith("1回あたり") else "rate"

c1, c2, c3, c4 = st.columns(4)
with c1:
    if cost_mode == "rate":
        cost_rate_pct = st.slider("商品原価率 (%)", 0.0, 100.0, 30.0, 0.5)
        cost_value = cost_rate_pct / 100
    else:
        cost_yen = st.number_input("1回あたりの原価 (円)", min_value=0, value=500, step=10)
        cost_value = cost_yen
with c2:
    ship_cost_yen = st.number_input("1回あたりの配送料・資材費 (円)", min_value=0, value=0, step=10, help="小計に配送料が含まれておらず加味したくない場合は0円のままでOKです。")
with c3:
    payment_fee_pct = st.slider("決済手数料 (%)", 0.0, 20.0, 0.0, 0.1, help="加味したくない場合は0%のままでOKです。")
with c4:
    target_profit_pct = st.slider("目標利益率 (%)", 0.0, 100.0, 20.0, 0.5)

payment_fee_rate = payment_fee_pct / 100
target_profit_rate = target_profit_pct / 100

st.divider()

# ---------------------------------------------------------
# 共通データ準備
# ---------------------------------------------------------
valid_df = df[df[COL_STATUS].isin(valid_status_selected)]
if COL_PAYMENT_STATUS in df.columns and payment_status_selected is not None:
    valid_df = valid_df[valid_df[COL_PAYMENT_STATUS].isin(payment_status_selected)]

fo_base = build_first_orders(valid_df, date_from, date_to, product_col, product_selected)  # 広告絞り込み前（比較用）

# 「終了した定期のみ」モードの場合、現在も「有効」（進行中）な定期は分母から除外する
sub_status_map = None
if COL_SUB_STATUS in df.columns:
    sub_status_map = df.drop_duplicates(subset=COL_SUB_ID).set_index(COL_SUB_ID)[COL_SUB_STATUS]
    if sub_status_mode == "終了した定期のみ（有効中は除外し確定値）":
        fo_base = fo_base[fo_base[COL_SUB_ID].map(sub_status_map) != "有効"]

fo_main = fo_base[fo_base[ad_col_choice].isin(ad_selected)]  # サイドバー広告フィルタ適用後（メイン分析用）

# 品質分析用：ステータス問わず全初回受注（valid_dfではなくdf全体から）
fo_raw_all_status = df[df[COL_COUNT] == 1].copy()
fo_raw_all_status = fo_raw_all_status[
    (fo_raw_all_status[COL_DATE].dt.date >= date_from) & (fo_raw_all_status[COL_DATE].dt.date <= date_to)
]
if product_col and product_selected is not None:
    fo_raw_all_status = fo_raw_all_status[fo_raw_all_status[product_col].isin(product_selected)]
fo_raw_all_status = fo_raw_all_status[fo_raw_all_status[ad_col_choice].isin(ad_selected)]

if len(fo_main) == 0:
    st.warning("条件に合致する初回受注が見つかりませんでした。絞り込み条件を確認してください。")
    st.stop()

sub_ids = tuple(fo_main[COL_SUB_ID].unique())
max_round_global = int(valid_df[valid_df[COL_SUB_ID].isin(sub_ids)][COL_COUNT].max())

retention, avg_cum_ltv, avg_cum_orders, denom = compute_cohort(valid_df, sub_ids, max_round_global)
cum_cost, cum_ship, cum_payment_fee, cum_gross_profit, target_profit, allowable_cpa = simulate_profit(
    avg_cum_ltv, avg_cum_orders, cost_mode, cost_value, ship_cost_yen, payment_fee_rate, target_profit_rate
)

# ---------------------------------------------------------
# タブ構成
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 基本分析", "🏆 広告グループ別ランキング", "🗺️ 継続率ヒートマップ",
    "🚨 受注品質分析", "📅 月次コホート", "📉 離脱率分析", "💳 決済手段別分析",
])

# ============ Tab1: 基本分析 ============
with tab1:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("対象顧客数（分母）", f"{denom:,} 人")
    m2.metric("期間", f"{date_from} 〜 {date_to}")
    m3.metric("最大定期回数", f"{max_round_global} 回")
    m4.metric("2回目継続率", f"{retention.get(2, 0):.1f} %" if max_round_global >= 2 else "N/A")

    if sub_status_map is not None:
        status_counts = sub_status_map.reindex(sub_ids).value_counts()
        active_n = int(status_counts.get("有効", 0))
        ended_n = int(denom - active_n)
        if sub_status_mode == "すべて含める（進行中も含む速報値）" and active_n > 0:
            st.caption(
                f"ℹ️ 分母{denom:,}人のうち、現在も**「有効」（進行中）**な定期が{active_n:,}人含まれています。"
                "この人たちは今後さらに回数を重ねる可能性があるため、直近の回数の継続率は実態より低めに出ることがあります。"
                "確定値だけで見たい場合はサイドバーの「継続率の集計対象」を切り替えてください。"
            )
        elif sub_status_mode.startswith("終了"):
            st.caption(f"ℹ️「終了した定期のみ」モードで集計中（進行中の定期は分母から除外、対象{denom:,}人）。")

    st.subheader("📈 回数別 継続率")
    fig_retention = go.Figure()
    fig_retention.add_trace(go.Scatter(
        x=retention.index, y=retention.values, mode="lines+markers+text",
        text=[f"{v:.1f}%" for v in retention.values], textposition="top center", line=dict(width=3), name="継続率",
    ))
    fig_retention.update_layout(xaxis_title="定期回数", yaxis_title="継続率 (%)", yaxis_range=[0, 105], height=420, xaxis=dict(dtick=1))
    st.plotly_chart(fig_retention, use_container_width=True)

    st.subheader("💰 累積LTV・累積粗利・許容CPAの推移")
    fig_ltv = go.Figure()
    fig_ltv.add_trace(go.Scatter(x=avg_cum_ltv.index, y=avg_cum_ltv.values, mode="lines+markers", name="累積LTV（売上）"))
    fig_ltv.add_trace(go.Scatter(x=cum_gross_profit.index, y=cum_gross_profit.values, mode="lines+markers", name="累積粗利"))
    fig_ltv.add_trace(go.Scatter(x=allowable_cpa.index, y=allowable_cpa.values, mode="lines+markers", name="許容CPA", line=dict(dash="dash")))
    fig_ltv.add_hline(y=0, line_color="gray", line_width=1)
    fig_ltv.update_layout(xaxis_title="定期回数", yaxis_title="金額 (円)", height=460, xaxis=dict(dtick=1),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_ltv, use_container_width=True)

    st.subheader("📋 集計データ")
    result_table = pd.DataFrame({
        "定期回数": retention.index, "継続率(%)": retention.values.round(1),
        "累積LTV(円)": avg_cum_ltv.values.round(0), "累積原価(円)": cum_cost.values.round(0),
        "累積配送費(円)": cum_ship.values.round(0), "累積決済手数料(円)": cum_payment_fee.values.round(0),
        "累積粗利(円)": cum_gross_profit.values.round(0), "目標利益(円)": target_profit.values.round(0),
        "許容CPA(円)": allowable_cpa.values.round(0),
    })
    st.dataframe(result_table, use_container_width=True, hide_index=True)
    csv_download = result_table.to_csv(index=False).encode("cp932", errors="ignore")
    st.download_button("この集計結果をCSVでダウンロード", data=csv_download, file_name="ltv_cpa_result.csv", mime="text/csv")

# ============ Tab2: 広告グループ別ランキング ============
with tab2:
    st.subheader(f"🏆 {ad_col_choice} 別 許容CPAランキング")
    st.caption("サイドバーで選択中の広告URL（グループ）を対象に、指定した回数時点の許容CPAで比較します。")

    rank_round = st.slider("何回目時点で比較するか", 1, max_round_global, min(3, max_round_global), key="rank_round")

    group_metrics = compute_multi_group_metrics(valid_df, fo_main, ad_col_choice, max_round_global)
    rows = []
    for g, m in group_metrics.items():
        if rank_round not in m["avg_cum_ltv"].index:
            continue
        gc, gs, gf, gp, gt, gcpa = simulate_profit(
            m["avg_cum_ltv"], m["avg_cum_orders"], cost_mode, cost_value, ship_cost_yen, payment_fee_rate, target_profit_rate
        )
        rows.append({
            ad_col_choice: g, "対象人数": m["denom"],
            f"{rank_round}回目継続率(%)": round(m["retention"].get(rank_round, np.nan), 1),
            f"{rank_round}回目累積LTV(円)": round(m["avg_cum_ltv"].get(rank_round, np.nan)),
            f"{rank_round}回目許容CPA(円)": round(gcpa.get(rank_round, np.nan)),
        })
    if rows:
        rank_df = pd.DataFrame(rows).sort_values(f"{rank_round}回目許容CPA(円)", ascending=False).reset_index(drop=True)
        rank_df.index += 1
        st.dataframe(rank_df, use_container_width=True)

        fig_rank = px.bar(
            rank_df, x=ad_col_choice, y=f"{rank_round}回目許容CPA(円)",
            color=f"{rank_round}回目許容CPA(円)", color_continuous_scale="RdYlGn",
        )
        fig_rank.update_layout(height=460, xaxis_title="", xaxis_tickangle=-30)
        st.plotly_chart(fig_rank, use_container_width=True)
    else:
        st.info("この回数まで到達したデータがまだありません。")

# ============ Tab3: 継続率ヒートマップ ============
with tab3:
    st.subheader(f"🗺️ {ad_col_choice} × 回数 継続率ヒートマップ")
    heat_max_round = st.slider("表示する最大回数", 1, max_round_global, min(10, max_round_global), key="heat_round")

    group_metrics_h = compute_multi_group_metrics(valid_df, fo_main, ad_col_choice, max_round_global)
    heat_rows = {g: m["retention"].reindex(range(1, heat_max_round + 1)) for g, m in group_metrics_h.items()}
    if heat_rows:
        heat_df = pd.DataFrame(heat_rows).T
        heat_df = heat_df.loc[heat_df[1].sort_values(ascending=False).index] if 1 in heat_df.columns else heat_df
        fig_heat = px.imshow(
            heat_df, text_auto=".0f", color_continuous_scale="Blues", aspect="auto",
            labels=dict(x="定期回数", y=ad_col_choice, color="継続率(%)"),
        )
        fig_heat.update_layout(height=max(400, 30 * len(heat_df)))
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("数値が空欄（NaN）の場合、その回数まで到達した顧客がまだいないことを示します。")
    else:
        st.info("表示できるデータがありません。")

# ============ Tab4: 受注品質分析 ============
with tab4:
    st.subheader(f"🚨 {ad_col_choice} 別 不良受注率（キャンセル・決済NG・返送・重複注文）")
    st.caption("有効ステータス絞り込みの影響を受けず、選択期間・広告・商品条件内の「全初回受注」を対象に算出します。")

    if len(fo_raw_all_status) == 0:
        st.info("対象データがありません。")
    else:
        quality_df = compute_quality_by_group(fo_raw_all_status, ad_col_choice).reset_index()
        st.dataframe(quality_df, use_container_width=True, hide_index=True)

        fig_q = px.bar(
            quality_df.sort_values("不良率(%)", ascending=False),
            x=ad_col_choice, y="不良率(%)", color="不良率(%)", color_continuous_scale="Reds",
        )
        fig_q.update_layout(height=460, xaxis_title="", xaxis_tickangle=-30)
        st.plotly_chart(fig_q, use_container_width=True)

# ============ Tab5: 月次コホート ============
with tab5:
    st.subheader("📅 獲得月別 継続率コホート")
    cohort_max_round = st.slider("表示する最大回数", 1, max_round_global, min(10, max_round_global), key="cohort_round")

    fo_month = fo_main.copy()
    fo_month["cohort_month"] = fo_month[COL_DATE].dt.strftime("%Y-%m")
    month_metrics = compute_multi_group_metrics(valid_df, fo_month, "cohort_month", max_round_global)
    month_rows = {g: m["retention"].reindex(range(1, cohort_max_round + 1)) for g, m in month_metrics.items()}
    denom_rows = {g: m["denom"] for g, m in month_metrics.items()}

    if month_rows:
        month_df = pd.DataFrame(month_rows).T.sort_index()
        fig_month = px.imshow(
            month_df, text_auto=".0f", color_continuous_scale="Greens", aspect="auto",
            labels=dict(x="定期回数", y="獲得月", color="継続率(%)"),
        )
        fig_month.update_layout(height=max(400, 30 * len(month_df)))
        st.plotly_chart(fig_month, use_container_width=True)

        denom_df = pd.DataFrame({"獲得月": list(denom_rows.keys()), "初回顧客数": list(denom_rows.values())}).sort_values("獲得月")
        st.caption("各月の分母（初回顧客数）")
        st.dataframe(denom_df, use_container_width=True, hide_index=True)
    else:
        st.info("表示できるデータがありません。")

# ============ Tab6: 離脱率分析 ============
with tab6:
    st.subheader("📉 回数間の離脱率（ドロップオフ）")
    st.caption("「n回目 → n+1回目」で、n回目継続者のうち何%が離脱したかを示します。")

    dropoff_vals = []
    for n in range(2, max_round_global + 1):
        prev = retention.get(n - 1, np.nan)
        curr = retention.get(n, np.nan)
        rate = (prev - curr) / prev * 100 if prev and prev > 0 else np.nan
        dropoff_vals.append({"区間": f"{n-1}回目→{n}回目", "離脱率(%)": round(rate, 1) if not np.isnan(rate) else None})

    dropoff_df = pd.DataFrame(dropoff_vals)
    if len(dropoff_df) > 0:
        fig_drop = px.bar(dropoff_df, x="区間", y="離脱率(%)", color="離脱率(%)", color_continuous_scale="OrRd")
        fig_drop.update_layout(height=440, xaxis_tickangle=-30)
        st.plotly_chart(fig_drop, use_container_width=True)
        st.dataframe(dropoff_df, use_container_width=True, hide_index=True)

        worst = dropoff_df.loc[dropoff_df["離脱率(%)"].idxmax()] if dropoff_df["離脱率(%)"].notna().any() else None
        if worst is not None:
            st.warning(f"最も離脱率が高い区間：**{worst['区間']}**（{worst['離脱率(%)']}%）。この回数のタイミングでのフォロー施策を優先検討してください。")
    else:
        st.info("2回目以降のデータがまだありません。")

# ============ Tab7: 決済手段別分析 ============
with tab7:
    st.subheader("💳 支払い方法別 継続率・LTV比較")

    if COL_PAYMENT_METHOD not in df.columns:
        st.info("このCSVには「支払い方法」列が含まれていないため、この分析は利用できません。")
    else:
        pm_round = st.slider("何回目時点で比較するか", 1, max_round_global, min(3, max_round_global), key="pm_round")

        pm_metrics = compute_multi_group_metrics(valid_df, fo_main, COL_PAYMENT_METHOD, max_round_global)
        pm_rows = []
        for g, m in pm_metrics.items():
            if pm_round not in m["avg_cum_ltv"].index:
                continue
            gc, gs, gf, gp, gt, gcpa = simulate_profit(
                m["avg_cum_ltv"], m["avg_cum_orders"], cost_mode, cost_value, ship_cost_yen, payment_fee_rate, target_profit_rate
            )
            pm_rows.append({
                COL_PAYMENT_METHOD: g, "対象人数": m["denom"],
                f"{pm_round}回目継続率(%)": round(m["retention"].get(pm_round, np.nan), 1),
                f"{pm_round}回目累積LTV(円)": round(m["avg_cum_ltv"].get(pm_round, np.nan)),
                f"{pm_round}回目許容CPA(円)": round(gcpa.get(pm_round, np.nan)),
            })

        if pm_rows:
            pm_df = pd.DataFrame(pm_rows).sort_values(f"{pm_round}回目許容CPA(円)", ascending=False).reset_index(drop=True)
            pm_df.index += 1
            st.dataframe(pm_df, use_container_width=True)

            st.markdown("**継続率カーブの比較**")
            fig_pm = go.Figure()
            for g, m in pm_metrics.items():
                fig_pm.add_trace(go.Scatter(
                    x=m["retention"].index, y=m["retention"].values, mode="lines+markers", name=f"{g}（{m['denom']}人）",
                ))
            fig_pm.update_layout(xaxis_title="定期回数", yaxis_title="継続率 (%)", yaxis_range=[0, 105], height=440, xaxis=dict(dtick=1))
            st.plotly_chart(fig_pm, use_container_width=True)
        else:
            st.info("この回数まで到達したデータがまだありません。")
