import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import joblib
from sklearn.preprocessing import PolynomialFeatures
import warnings
import requests
import urllib3
import os
import gdown

# --- 1. 경고 및 보안 설정 ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

# --- 2. 페이지 설정 (가장 먼저 실행되어야 함) ---
st.set_page_config(page_title="공정 품질 KPI 대시보드", layout="wide")

# --- 3. 상수 및 경로 설정 ---
MONTHLY_CONFIG = {
    "2023-09": {"shot": "1ed7eRVk25aTBN1aVsvzEZZQrbGPN0lbY", "pre": "1rw9Kv055wD1w_HLO1Pdizbk4CEfHAeiF"},
    "2023-10": {"shot": "1gYE2xh6TcrtlNd6rAYe0MrjOmkTvz9vL", "pre": "12ZKLaEUw09JGno05isB7QQW_zMRicl-l"},
    "2023-11": {"shot": "1QPNxQffDP3F22KjyXhcORSxgY_jYha96", "pre": "1xaObr2lANjOsHENGkslUqNiHOuHEaVFb"}
}

# [추가] 설비별 모델 파일 ID 매핑
MACHINE_MODEL_CONFIG = {
    "D01": "1E7PwqwTu64FJjqWOIYWrodfC6migkKbR",
    "D10": "1GYsZXdCLc7izqYag6gQ40q81CiQN7WBR",
    "F09": "1yPlNAUEatopQbHqv1BInnD4qP3sRRzBZ",
    "F11": "1uTdb1Z9L04sJYq93KO0zXuk1VJWQWZ9Y",
    "G01": "1zHx1w2GPF902iz3eKMziy34NuLMu1adG",
    "G02": "1_xQvj0l2hcCcuDrisQxJZ5Mss6Fep96k",
    "G03": "1sDcIxxQ1MwXIyXfDq-3t6tkDwMePY46t",
    "G05": "17jPNOz-SeLwh9bRd8kDrsPcsGmBCXFAS",
    "G06": "1dF_ZCpJsXTFK16JXFKx5PbFYNqHF1p1o",
    "G10": "1LsTOoSUSUPqqx_088wv8YvHl9RUDbvR9"
}

# --- 4. 파일 다운로드 함수 ---
def download_from_gdrive(file_id, output_path):
    if not os.path.exists(output_path):
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            gdown.download(url, output_path, quiet=True)
        except Exception as e:
            st.error(f"다운로드 실패: {e}")
            return False
    return True

# --- [수정] 6. 데이터 로드 및 캐싱 함수 ---

# --- 6. [수정] 설비별 모델 로드 및 캐싱 함수 ---
# 메모리 절약을 위해 max_entries를 설정하여 필요한 모델만 메모리에 올립니다.
@st.cache_resource(max_entries=5) 
def load_single_machine_model(mach_no):
    file_id = MACHINE_MODEL_CONFIG.get(mach_no)
    if not file_id:
        return None
    
    model_path = f"model_{mach_no}.pkl"
    
    # 1. 파일 다운로드 로직 (중복 다운로드 방지)
    if not os.path.exists(model_path):
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            # gdown 실행 시 오류가 나면 None을 반환하여 캐시에 저장되지 않게 함
            output = gdown.download(url, model_path, quiet=True)
            if output is None:
                return None
        except Exception as e:
            st.error(f"모델 파일 다운로드 중 오류 발생 ({mach_no}): {e}")
            return None
    
    # 2. 모델 로드 및 CPU 강제 설정
    try:
        # mmap_mode='r'을 사용하면 메모리 사용량을 줄이고 로딩 속도를 높일 수 있습니다.
        model_info = joblib.load(model_path)
        
        # XGBoost 또는 다른 모델이 GPU 설정을 가지고 있을 경우 CPU로 강제 전환
        # 이 과정이 로드 직후에 수행되어야 무한 로딩을 방지할 수 있습니다.
        if isinstance(model_info, dict) and 'model' in model_info:
            target_model = model_info['model']
            if hasattr(target_model, 'set_params'):
                try:
                    target_model.set_params(device="cpu", n_jobs=1)
                except:
                    pass
        return model_info
        
    except Exception as e:
        st.error(f"모델 로드 실패 ({mach_no}): {e}")
        # 로드 실패 시 손상된 파일일 수 있으므로 삭제 (다음 실행 시 재다운로드 유도)
        if os.path.exists(model_path):
            os.remove(model_path)
        return None


def get_all_models_for_monitoring(mach_list):
    """실시간 관제 센터(Tab 1)에서 여러 설비를 한꺼번에 분석할 때 사용"""
    models = {}
    for m in mach_list:
        mdl = load_single_machine_model(m)
        if mdl:
            models[m] = mdl
    return models

@st.cache_data
def load_monthly_data(year_month):
    config = MONTHLY_CONFIG.get(year_month)
    if not config: return pd.DataFrame(), pd.DataFrame()

    shot_file = f"shot_{year_month}.parquet"
    pre_file = f"pre_{year_month}.parquet"

    download_from_gdrive(config['shot'], shot_file)
    download_from_gdrive(config['pre'], pre_file)

    try:
        df_shot = pd.read_parquet(shot_file)
        df_pre = pd.read_parquet(pre_file)
        
        # 전처리
        for df in [df_shot, df_pre]:
            df.columns = [c.strip() for c in df.columns]
            if 'Timestamp_사출' in df.columns:
                df['Timestamp_사출'] = pd.to_datetime(df['Timestamp_사출'], errors='coerce')
        
        if 'Result' not in df_shot.columns and 'NG' in df_shot.columns:
            df_shot['Result'] = df_shot['NG'].map({0: '정상(OK)', 1: '불량(NG)'})
            
        return df_shot, df_pre
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return pd.DataFrame(), pd.DataFrame()

# --- [수정] 관제 센터용 데이터 로직 (Tail 적용) ---
def get_recent_7days_status(df_pre):
    if df_pre.empty:
        return pd.DataFrame()
    
    # 1. 일단 최신 7일 데이터를 가져오되
    latest_date = df_pre['Timestamp_사출'].max()
    start_date = latest_date - pd.Timedelta(days=6)
    week_df = df_pre[df_pre['Timestamp_사출'] >= start_date].copy()
    
    # 2. 성능 테스트를 위해 각 설비별로 가장 최근 '100건'만 남깁니다 (Tail 추출)
    # 데이터가 너무 많아 연산이 밀리는 것을 방지합니다.
    week_df = week_df.groupby('MACHNO').tail(100).reset_index(drop=True)
    
    return week_df

# --- [수정] 7. 실시간 관제용 로직 (모델 딕셔너리 구조 반영) ---

def get_machine_status(mach, m_week_data, models_dict):
    """
    설비별 AI 모델(Scaler + Model + Features)을 사용하여 
    실제 데이터 기반 위험도를 판정합니다.
    """
    if m_week_data.empty:
        return "empty"
    
    try:
        # 1. 설비에 해당하는 모델 정보(dict) 가져오기
        mach_key = str(mach)
        if not models_dict or mach_key not in models_dict:
            return "error (No Model)"
        
        info = models_dict[mach_key]
        trained_features = info['features']
        
        # 2. 최신 데이터 1건 추출 및 피처 구성
        latest_row = m_week_data.sort_values('Timestamp_사출').iloc[-1]
        
        # 학습 당시 사용한 피처 리스트와 동일하게 구성
        X_input = pd.DataFrame([latest_row])
        X = pd.DataFrame(index=[0])
        for col in trained_features:
            X[col] = X_input[col] if col in X_input.columns else 0
        X = X.fillna(0)

        # 3. 잔차(Residual) 계산 (최근 데이터가 1건이므로 이전 데이터 필요)
        # 관제용 로직을 위해 m_week_data 전체를 활용해 잔차를 구함
        X_week = m_week_data[trained_features].copy()
        X_res_full = X_week - X_week.rolling(window=10, min_periods=1).mean()
        X_res_latest = X_res_full.iloc[-1:] # 마지막 행 잔차
        X_res_latest.columns = [f"{c}_resid" for c in X_res_latest.columns]
        
        # 데이터 결합 (원본 + 잔차)
        X_combined = pd.concat([X.reset_index(drop=True), X_res_latest.reset_index(drop=True)], axis=1)

        # 4. Polynomial Features (G06 및 난조군 특수 로직)
        if mach_key == 'G06':
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            top_cols = [c for c in X.columns if any(s in c for s in ['금형온도', '최소쿠션', '충진시간', '압력'])][:6]
            X_poly = poly.fit_transform(X_combined[top_cols])
            X_final = np.hstack([X_combined.values, X_poly])
        elif info.get('group') == '난조군':
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            top_cols = [c for c in X.columns if any(s in c for s in ['금형온도', '최소쿠션', '충진시간'])][:5]
            X_poly = poly.fit_transform(X_combined[top_cols])
            X_final = np.hstack([X_combined.values, X_poly])
        else:
            X_final = X_combined.values

        # 5. 스케일링 및 예측
        X_scaled = info['scaler'].transform(X_final)
        model_obj = info['model']
        if hasattr(model_obj, 'set_params'):
            try:
                # 학습 환경과 상관없이 현재 실행 환경(CPU)에 맞춤
                model_obj.set_params(device="cpu", updater="grow_quantile_histmaker")
            except:
                pass
                
        prob = model_obj.predict_proba(X_scaled)[0, 1]
        
        # 6. 이상치 점수(LOF) 반영 (선택 사항)
        d_score = info['lof'].predict(X_scaled)[0]
        if d_score == 1: # 이상치인 경우 페널티 부여
            prob *= info.get('lof_penalty', 1.0)
            prob = min(prob, 1.0) # 100% 초과 방지

        # 7. 상태 및 색상 결정
        threshold = info.get('threshold', 0.5)
        status_label = "정상"
        if prob >= threshold:
            status_label = "위험(발생)"
        elif prob >= threshold * 0.7:
            status_label = "주의"

        return {
            "판정": status_label,
            "위험도": prob,
            "시간": latest_row['Timestamp_사출'].strftime('%m/%d %H:%M')
        }
        
    except Exception as e:
        return f"error ({str(e)})"
    
# ==========================================
# --- 8. 메인 실행부 (UI 구성) ---
st.sidebar.title("🛠️ 공정 필터링")

# 1. 월 선택 (기존 코드 유지)
raw_month_list = sorted(list(MONTHLY_CONFIG.keys()), reverse=True)
latest_month_key = raw_month_list[0] # 가장 최신월 (2023-11)

# 내부 키와 표시용 이름을 매핑 (2023-XX를 2025년 XX월로 치환)
display_month_map = {
    m: m.replace("2023-", "2025년 ").replace("-", "") + "월" 
    for m in raw_month_list
}

# 사용자에게는 '2025년 11월'로 보여줌
selected_display_month = st.sidebar.selectbox(
    "📅 분석 월 선택", 
    options=list(display_month_map.values())
)

# 실제 데이터 로드에 사용할 키 추출 (예: 2023-11)
selected_month = [k for k, v in display_month_map.items() if v == selected_display_month][0]

# 2. 분석용 데이터 로드 (사이드바 선택에 따라 바뀜)
df_shot, df_pre = load_monthly_data(selected_month)

# [추가] 설비 리스트 및 선택 로직 (필요함)
machine_list = sorted(df_shot['MACHNO'].unique().tolist()) if not df_shot.empty else []
selected_machine = st.sidebar.selectbox("🏗️ 분석 대상 설비", options=machine_list) if machine_list else None

# [수정] 실시간 관제 전용 데이터 로드
if selected_month == latest_month_key:
    df_pre_for_monitor = df_pre
else:
    _, df_pre_for_monitor = load_monthly_data(latest_month_key)

# 3. 관제 센터용 데이터 (df_pre_for_monitor를 명시적으로 사용)
week_df = get_recent_7days_status(df_pre_for_monitor)

if not week_df.empty:
    current_machs = sorted(week_df['MACHNO'].unique().tolist())
    models_dict = get_all_models_for_monitoring(current_machs)
else:
    models_dict = {}

# 5. 상세 분석용 모델 로드 (선택된 1개 설비)
# --- 데이터 변수명 통일 및 전월 데이터 로드 ---
df_filtered_month = df_shot  # KPI 계산에 사용되는 메인 변수

# 전월 데이터 로드 (KPI 비교용)
current_idx = raw_month_list.index(selected_month)
if current_idx + 1 < len(raw_month_list):
    last_month_key = raw_month_list[current_idx + 1]
    df_last_month = load_monthly_data(last_month_key)[0]
else:
    df_last_month = pd.DataFrame()

df_final = df_shot # 하단 품질 진단 로직용

# 탭 구성
tab_kpi, tab_detail, tab_analysis = st.tabs(["🚀 공장 전체 KPI", "🔍 설비 상세 리포트", "🚨 불량 원인 분석"])

# ==============================================================================
# TAB 1: 공장 전체 KPI
# ==============================================================================
with tab_kpi:
    st.title("🚀 공정 품질 핵심 성과 지표 (KPI)")
    st.info(f"📍 **{selected_month}** 공장 전체 설비 가동 현황 요약")
    
    # [1] 핵심 메트릭
    total_qty = len(df_filtered_month)
    total_ng = len(df_filtered_month[df_filtered_month['Result'] == '불량(NG)'])
    total_ok = total_qty - total_ng
    avg_defect_rate = (total_ng / total_qty * 100) if total_qty > 0 else 0

    # 전월 비교
    has_history = False
    if not df_last_month.empty:
        l_total = len(df_last_month)
        l_ng = len(df_last_month[df_last_month['Result'] == '불량(NG)'])
        l_ok = l_total - l_ng
        l_rate = (l_ng / l_total * 100) if l_total > 0 else 0
        
        qty_delta = f"{total_qty - l_total:,}건"
        ok_delta = f"{total_ok - l_ok:,}건"
        ng_delta = f"{total_ng - l_ng:,}건"
        rate_delta = f"{avg_defect_rate - l_rate:.2f}%"
        has_history = True

    k1, k2, k3, k4 = st.columns(4)
    if has_history:
        k1.metric("총 생산량 (All)", f"{total_qty:,}건", delta=qty_delta)
        k2.metric("양품 수량", f"{total_ok:,}건", delta=ok_delta)
        k3.metric("불량 수량", f"{total_ng:,}건", delta=ng_delta, delta_color="inverse")
        k4.metric("공장 평균 불량률", f"{avg_defect_rate:.2f}%", delta=rate_delta, delta_color="inverse")
    else:
        k1.metric("총 생산량 (All)", f"{total_qty:,}건")
        k2.metric("양품 수량", f"{total_ok:,}건")
        k3.metric("불량 수량", f"{total_ng:,}건")
        k4.metric("공장 평균 불량률", f"{avg_defect_rate:.2f}%")

    st.divider()

    # [2] 설비별 순위 및 추이 차트
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("📊 설비별 불량률 순위")
        m_stats = df_filtered_month.groupby('MACHNO')['Result'].value_counts().unstack(fill_value=0)
        for col in ['불량(NG)', '정상(OK)']:
            if col not in m_stats.columns: m_stats[col] = 0
        
        m_stats['Rate'] = (m_stats['불량(NG)'] / (m_stats['불량(NG)'] + m_stats['정상(OK)']) * 100)
        m_stats = m_stats.sort_values('Rate', ascending=False).reset_index()

        fig_rank = px.bar(m_stats, x='MACHNO', y='Rate', 
                          text=m_stats['Rate'].apply(lambda x: f'{x:.1f}%'),
                          color='Rate', color_continuous_scale='Reds',
                          labels={'Rate': '불량률(%)', 'MACHNO': '설비'})
        fig_rank.update_traces(textposition='outside', cliponaxis=False)
        fig_rank.update_layout(height=400, margin=dict(t=30), yaxis=dict(range=[0, m_stats['Rate'].max()*1.2]))
        st.plotly_chart(fig_rank, width='stretch')

    with col_b:
        st.subheader("📉 설비별 생산량 추이 비교")
        compare_machines = st.multiselect("비교 대상 설비", options=machine_list, default=[selected_machine])
        
        if compare_machines:
            df_compare = df_filtered_month[df_filtered_month['MACHNO'].isin(compare_machines)].copy()
            if 'Timestamp_사출' in df_compare.columns:
                df_compare['Date'] = df_compare['Timestamp_사출'].dt.date
            
                trend_compare = df_compare.groupby(['Date', 'MACHNO']).size().reset_index(name='Count')
                fig_compare = px.line(trend_compare, x='Date', y='Count', color='MACHNO', markers=True)
                fig_compare.update_layout(
                    height=400, 
                    margin=dict(t=30), 
                    xaxis=dict(
                        tickformat="%d일",           # 표시 형식: 01일, 05일...
                        dtick=86400000.0 * 5,       # 1일을 밀리초로 환산한 값에 5를 곱함 (5일 간격)
                        tickangle=0                 # 글자가 겹치지 않도록 수평 유지
                    ),
                    legend=dict(orientation="h", y=1.1)
                )
                st.plotly_chart(fig_compare, width='stretch')
    # [3] 실시간 관제 센터
    st.divider()
    st.subheader("🏭 실시간 설비 이상 징후 관제 센터 (최근 7일)")
    st.info(f"💡 현재 관제 센터는 **실시간 상태**를 유지하기 위해 선택 월과 무관하게 **최신 데이터({latest_month_key})**를 분석 중입니다.")
    week_df = get_recent_7days_status(df_pre)
    if not week_df.empty:
        all_mach_list = sorted(week_df['MACHNO'].unique())
        
        st.write("🔍 분석할 설비를 선택하면 AI 모델이 로드됩니다.")
        selected_monitor_mach = st.radio(
            "관제 대상 설비 선택", 
            all_mach_list, 
            horizontal=True,
            key="monitor_mach_selector"
        )

        if selected_monitor_mach:
            with st.status(f"🤖 {selected_monitor_mach} 설비 모델 로드 및 분석 중...", expanded=False) as status:
                single_model_info = load_single_machine_model(selected_monitor_mach)
                
                if single_model_info:
                    # 함수 요구 형식에 맞춰 래핑
                    temp_models_dict = {selected_monitor_mach: single_model_info}
                    m_week_data = week_df[week_df['MACHNO'] == selected_monitor_mach]
                    
                    res = get_machine_status(selected_monitor_mach, m_week_data, temp_models_dict)
                    status.update(label=f"✅ {selected_monitor_mach} 분석 완료", state="complete")
                    
                    # 결과 출력 UI
                    col_res1, col_res2 = st.columns([1, 2])
                    with col_res1:
                        with st.container(border=True):
                            if isinstance(res, dict):
                                color = "#e74c3c" if "위험" in res['판정'] else "#2ecc71"
                                st.markdown(f"### {selected_monitor_mach}")
                                st.markdown(f"<h1 style='text-align: center; color: {color};'>{res['판정']}</h1>", unsafe_allow_html=True)
                                st.metric("현재 위험도", f"{res['위험도']:.1%}")
                                st.caption(f"🕒 최종 데이터: {res['시간']}")
                            else:
                                st.error(f"분석 오류: {res}")
                    
                    with col_res2:
                        st.write(f"📊 {selected_monitor_mach} 주요 판단 근거 (Top 5)")
                        try:
                            # 모델 및 피처 정보 추출
                            model_obj = single_model_info['model']
                            trained_feats = single_model_info['features']
                            
                            if hasattr(model_obj, 'feature_importances_'):
                                importances = model_obj.feature_importances_
                                # 피처와 중요도 매핑 (Polynomial 등으로 늘어난 경우 대비)
                                feat_imp_df = pd.DataFrame({
                                    'Feature': trained_feats[:len(importances)],
                                    'Importance': importances[:len(trained_feats)]
                                }).sort_values('Importance', ascending=True).tail(5)

                                fig_imp = px.bar(
                                    feat_imp_df, x='Importance', y='Feature', 
                                    orientation='h', color='Importance',
                                    color_continuous_scale='Reds' if "위험" in res['판정'] else 'Blues',
                                    text_auto='.3f'
                                )
                                fig_imp.update_layout(
                                    height=250, margin=dict(t=0, b=0, l=0, r=0),
                                    showlegend=False, coloraxis_showscale=False,
                                    xaxis_title="기여도 (Weight)", yaxis_title=None
                                )
                                st.plotly_chart(fig_imp, width='stretch')
                                st.caption("💡 AI가 현재 시점에서 불량 가능성을 판단할 때 가장 중요하게 평가한 공정 변수입니다.")
                            else:
                                st.info("현재 모델은 세부 판단 근거(Feature Importance)를 지원하지 않습니다.")
                        except Exception as e:
                            st.info(f"판단 근거 시각화 중 알 수 없는 오류가 발생했습니다.")
                else:
                    st.error(f"❌ {selected_monitor_mach} 설비의 모델 파일을 찾을 수 없습니다.")
    else:
        st.error("최근 7일간의 데이터를 찾을 수 없습니다.")

# ==============================================================================
# TAB 2: 설비 상세 리포트
# ==============================================================================
with tab_detail:
    st.markdown(f"### 🔍 {selected_machine} 설비 정밀 분석 리포트")
    
    # 해당 설비 데이터 중 가장 최근 200건만 가져와서 시각화/분석 진행
    m_df_full = df_filtered_month[df_filtered_month['MACHNO'] == selected_machine].sort_values('Timestamp_사출')
    m_df = m_df_full.tail(200) # 전체 데이터 대신 Tail(200건)만 사용하여 메모리 보호
    
    if 'Timestamp_사출' in m_df.columns:
        m_df['Date'] = m_df['Timestamp_사출'].dt.date

    # 설비 상세 메트릭
    m_t = len(m_df)
    m_ng = len(m_df[m_df['Result'] == '불량(NG)'])
    m_ok = m_t - m_ng
    m_rate = (m_ng / m_t * 100) if m_t > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("생산량", f"{m_t:,}건")
    m2.metric("양품 수량", f"{m_ok:,}건")
    m3.metric("불량 수량", f"{m_ng:,}건")
    m4.metric("불량률", f"{m_rate:.2f}%")
    
    st.write("")

    c1, c2 = st.columns([1, 2.5])
    with c1:
        st.write(f"##### 🍩 품질 비율")
        fig_pie = px.pie(m_df, names='Result', hole=0.6, color='Result',
                          color_discrete_map={'정상(OK)': '#2ecc71', '불량(NG)': '#e74c3c'})
        fig_pie.update_layout(height=300, showlegend=True, 
                              legend=dict(orientation="h", y=-0.1), margin=dict(t=20, b=20))
        fig_pie.add_annotation(text=f"{m_rate:.1f}%", x=0.5, y=0.5, font_size=20, showarrow=False)
        st.plotly_chart(fig_pie, width='stretch')

    with c2:
        st.write(f"##### 📈 일별 생산 추이")
        daily_prod = m_df.groupby(['Date', 'Result']).size().reset_index(name='Count')
        fig_line_prod = px.line(daily_prod, x='Date', y='Count', color='Result',
                                color_discrete_map={'정상(OK)': '#2ecc71', '불량(NG)': '#e74c3c'},
                                markers=True)
        fig_line_prod.update_layout(height=300, margin=dict(t=20, b=20), 
                                    xaxis=dict(tickformat="%Y-%m-%d"),
                                    legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_line_prod, width='stretch')

    st.write("---")
    st.write(f"💡 {selected_machine} 품질 진단")
    
    p_avg = 0.05
    if not df_final.empty:
        p_avg = (df_final['Result'] == '불량(NG)').mean()
    
    target_warn = p_avg * 1.5 * 100
    
    if m_rate <= p_avg * 100:
        st.success(f"✅ **안정**: 불량률이 공장 평균({p_avg*100:.2f}%) 이하입니다.")
    elif m_rate <= target_warn:
        st.warning(f"⚠️ **주의**: 불량률이 공장 평균을 상회합니다.")
    else:
        st.error(f"🚨 **위험**: 불량률이 관리 한계를 초과했습니다.")

    # 일별 불량률 추이
    st.subheader("📈 일별 불량률 추이 (비가동 구간 포함)")
    
    if not m_df.empty:
        start_date = m_df['Date'].min()
        end_date = m_df['Date'].max()
        all_days = pd.date_range(start=start_date, end=end_date).date
        
        daily_stats = m_df.groupby('Date')['Result'].value_counts().unstack(fill_value=0)
        if '불량(NG)' not in daily_stats.columns: daily_stats['불량(NG)'] = 0
        if '정상(OK)' not in daily_stats.columns: daily_stats['정상(OK)'] = 0
        
        daily_stats['Rate'] = (daily_stats['불량(NG)'] / (daily_stats['불량(NG)'] + daily_stats['정상(OK)'])) * 100
        daily_stats = daily_stats.reset_index()

        existing_days = daily_stats['Date'].tolist()
        missing_days = [d for d in all_days if d not in existing_days]

        fig_line = px.line(daily_stats, x='Date', y='Rate', markers=True, text=daily_stats['Rate'].apply(lambda x: f'{x:.1f}%'))
        
        for m_day in missing_days:
            fig_line.add_vrect(x0=pd.to_datetime(m_day)-pd.Timedelta(hours=12),
                               x1=pd.to_datetime(m_day)+pd.Timedelta(hours=12),
                               fillcolor="Gray", opacity=0.15, layer="below", line_width=0)

        fig_line.update_traces(line_color='#e74c3c', textposition="top center")
        fig_line.update_layout(height=400, xaxis=dict(tickformat="%d일"), yaxis_title="불량률(%)")
        st.plotly_chart(fig_line, width='stretch')

# ==============================================================================
# TAB 3: 불량 원인 분석
# ==============================================================================
with tab_analysis:
    st.subheader("🚨 불량(NG) 데이터 특성 상세 분석")
    
    available_dates = ["전체(해당 월)"] + sorted(m_df['Date'].unique().astype(str).tolist(), reverse=True)
    selected_date_analysis = st.selectbox("📅 분석 기간 선택", available_dates, key="analysis_date")

    if selected_date_analysis == "전체(해당 월)":
        target_df = m_df
        label = "이번 달 전체"
    else:
        target_df = m_df[m_df['Date'].astype(str) == selected_date_analysis]
        label = selected_date_analysis

    m_ng_df = target_df[target_df['Result'] == '불량(NG)']
    m_ok_df = target_df[target_df['Result'] == '정상(OK)']

    if not m_ng_df.empty:
        st.markdown(f"**{label}** 기준, 불량 데이터 **{len(m_ng_df)}건** 분석")
        
        analyze_cols = ['Cycle Time', '사출 시간', '충진 시간', '최소 쿠션', '피크압_주성분', '보압 완료 위치']
        valid_cols = [c for c in analyze_cols if c in m_df.columns]
        
        if valid_cols:
            ng_mean = m_ng_df[valid_cols].mean()
            ok_mean = m_ok_df[valid_cols].mean() if not m_ok_df.empty else m_df[valid_cols].mean()

            cols = st.columns(len(valid_cols))
            for i, col in enumerate(valid_cols):
                diff = ng_mean[col] - ok_mean[col]
                cols[i].metric(col, f"{ng_mean[col]:.2f}", f"{diff:+.2f}", delta_color="inverse")

            st.write("#### 🕸️ 정상 대비 변동 비율 (%)")
            ratios = [(ng_mean[c] / ok_mean[c] * 100) if ok_mean[c] != 0 else 0 for c in valid_cols]
            
            r_data = ratios + [ratios[0]]
            theta_data = valid_cols + [valid_cols[0]]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=r_data, theta=theta_data, fill='toself', name='불량 특성'))
            fig_radar.add_trace(go.Scatterpolar(r=[100]*len(theta_data), theta=theta_data, 
                                                line=dict(dash='dash', color='green'), name='정상 기준'))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), height=450)
            st.plotly_chart(fig_radar, width='stretch')
        else:
            st.warning("분석할 공정 데이터 컬럼을 찾을 수 없습니다.")
    else:
        st.success(f"✅ {label}에는 불량 데이터가 없습니다.")







