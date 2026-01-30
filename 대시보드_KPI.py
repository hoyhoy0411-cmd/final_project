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
PREPROCESSED_PATH = "전처리데이터.parquet"
SHOT_DATA_PATH = "대시보드_샷별.parquet"
MODEL_PATH = "injection_models.pkl"

FILE_IDS = {
    "models": "1ozTrBdUE-4fq-wghLSCInzzuXHtVcmTc", 
    "shot_data": "19Lh5DFCkl-RO0myqYTJWsBffJSxPwVR3", 
    "preprocessed": "1_s_tXukPRANC7wsfk0nv5k8HQB1Exby6"
}

# --- 4. 파일 다운로드 함수 ---
def download_file(file_id, output_name, display_text):
    """구글 드라이브에서 파일을 다운로드하여 로컬에 저장하는 함수"""
    url = f'https://drive.google.com/uc?id={file_id}'
    
    # 파일이 이미 존재하면 다운로드 건너뜀 (속도 향상)
    if not os.path.exists(output_name):
        try:
            with st.spinner(f'📥 데이터 로드 중: {display_text}...'):
                gdown.download(url, output_name, quiet=False)
        except Exception as e:
            st.error(f"❌ 다운로드 실패 ({display_text}): {e}")
            return None
    return output_name

# --- 5. 앱 시작 시 데이터 다운로드 실행 ---
# 파일이 없으면 다운로드하고, 있으면 넘어갑니다.
download_file(FILE_IDS['models'], MODEL_PATH, "AI 분석 모델")
download_file(FILE_IDS['shot_data'], SHOT_DATA_PATH, "메인 공정 데이터")
download_file(FILE_IDS['preprocessed'], PREPROCESSED_PATH, "실시간 전처리 데이터")


# --- 6. 데이터 로드 및 캐싱 함수 ---

@st.cache_data
def get_base_data():
    """메인 공정 데이터를 읽어오는 함수"""
    if not os.path.exists(SHOT_DATA_PATH):
        st.error("데이터 파일이 로컬에 존재하지 않습니다.")
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(SHOT_DATA_PATH, engine='pyarrow')
        df.columns = [c.strip() for c in df.columns]
        df['Timestamp_사출'] = pd.to_datetime(df['Timestamp_사출'], errors='coerce')
        df = df.dropna(subset=['Timestamp_사출'])
        return df
    except Exception as e:
        st.error(f"파일 읽기 오류: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_all_models():
    """저장된 파일로부터 모델 객체를 로드 (다운로드는 이미 완료됨)"""
    try:
        if os.path.exists(MODEL_PATH):
            return joblib.load(MODEL_PATH)
        else:
            st.warning("⚠️ 모델 파일이 없습니다.")
            return {}
    except Exception as e:
        st.error(f"⚠️ 모델 파일 로드 실패: {e}")
        return {}

@st.cache_data
def load_latest_week_data():
    """실시간 관제용 최근 1주일 데이터 로드"""
    try:
        if not os.path.exists(PREPROCESSED_PATH):
            return pd.DataFrame(), None, None
            
        df = pd.read_parquet(PREPROCESSED_PATH, engine='pyarrow')
        df.columns = [c.strip() for c in df.columns]
        
        if 'Timestamp_사출' not in df.columns:
            return pd.DataFrame(), None, None

        df['Timestamp_사출'] = pd.to_datetime(df['Timestamp_사출'], errors='coerce')
        df = df.dropna(subset=['Timestamp_사출'])
        df['Date'] = df['Timestamp_사출'].dt.date
        
        latest_date = df['Timestamp_사출'].max()
        if pd.isna(latest_date):
            return pd.DataFrame(), None, None
            
        start_date = (latest_date - pd.Timedelta(days=6)).replace(hour=0, minute=0, second=0)
        week_df = df[(df['Timestamp_사출'] >= start_date) & (df['Timestamp_사출'] <= latest_date)].copy()
        
        return week_df, start_date.date(), latest_date.date()
    except Exception as e:
        st.error(f"주간 데이터 로드 실패: {e}")
        return pd.DataFrame(), None, None

@st.cache_data
def load_recent_process_data(n_per_machine=50):
    """설비별 최신 데이터 로드 (실시간 시뮬레이션용)"""
    try:
        if not os.path.exists(PREPROCESSED_PATH):
            return pd.DataFrame()
            
        df = pd.read_parquet(PREPROCESSED_PATH, engine='pyarrow')
        df.columns = [c.strip() for c in df.columns]
        
        if 'Timestamp_사출' not in df.columns:
            return pd.DataFrame()
            
        df['Timestamp_사출'] = pd.to_datetime(df['Timestamp_사출'], errors='coerce')
        # 설비별 최신 n건만 필터링
        recent_df = df.groupby('MACHNO').tail(n_per_machine).copy()
        return recent_df
    except Exception as e:
        st.error(f"🚨 실시간 전처리 데이터 로드 실패: {e}")
        return pd.DataFrame()

@st.cache_data
def get_available_months():
    """사용 가능한 월 목록 추출"""
    try:
        if not os.path.exists(SHOT_DATA_PATH):
            return []
            
        df_full = pd.read_parquet(SHOT_DATA_PATH, columns=['Timestamp_사출'], engine='pyarrow')
        df_full['Timestamp_사출'] = pd.to_datetime(df_full['Timestamp_사출'], errors='coerce')
        df_full = df_full.dropna(subset=['Timestamp_사출'])
        return sorted(df_full['Timestamp_사출'].dt.strftime('%Y-%m').unique(), reverse=True)
    except Exception as e:
        st.error(f"📅 월 목록 로드 실패: {e}")
        return []

@st.cache_data
def load_data_by_month(selected_month):
    """선택된 월의 데이터 필터링"""
    df = get_base_data()
    if df.empty: return pd.DataFrame(), pd.DataFrame()
    
    df['YearMonth'] = df['Timestamp_사출'].dt.strftime('%Y-%m')
    
    # 전월 계산
    current_date = pd.to_datetime(selected_month + "-01")
    last_month_str = (current_date - pd.offsets.MonthBegin(1)).strftime('%Y-%m')
    
    # Result 컬럼 생성
    if 'NG' in df.columns and 'Result' not in df.columns:
        df['Result'] = df['NG'].map({0: '정상(OK)', 1: '불량(NG)'})
        
    curr_df = df[df['YearMonth'] == selected_month].copy()
    last_df = df[df['YearMonth'] == last_month_str].copy()
    return curr_df, last_df

# --- 7. 상태 판정 로직 (AI 모델 활용) ---
def get_machine_status(mach, day_df, _models_dict): 
    if _models_dict is None or len(_models_dict) == 0:
        return "no_model"
    if day_df.empty or mach not in _models_dict:
        return "empty"
    
    m_data = day_df[day_df['MACHNO'] == mach].sort_values('Timestamp_사출')
    if m_data.empty: 
        return "empty"
    
    info = _models_dict[mach]
    try:
        # 특징 추출 및 전처리
        X_input = m_data[info['features']].tail(1).values.astype(np.float32)
        
        expected_features = info['scaler'].n_features_in_
        current_features = X_input.shape[1]
        
        if current_features < expected_features:
            padding_size = expected_features - current_features
            padding = np.zeros((1, padding_size), dtype=np.float32)
            X_final = np.hstack([X_input, padding])
        else:
            X_final = X_input[:, :expected_features]

        X_scaled = info['scaler'].transform(X_final)
        
        model = info['model']
        # XGBoost CPU 설정 (호환성)
        if hasattr(model, 'get_booster'):
            try:
                model.get_booster().set_param({'predictor': 'cpu_predictor', 'device': 'cpu'})
            except:
                pass
        
        probs = model.predict_proba(X_scaled)[:, 1][0]
        
        # LOF(이상탐지) 반영
        d_score = info['lof'].predict(X_scaled)[0]
        if d_score == 1: 
            probs *= info.get('lof_penalty', 1.0)
            
        is_anomaly = probs >= info.get('threshold', 0.5)
        last_time = m_data['Timestamp_사출'].iloc[-1].strftime('%H:%M:%S')
        
        return {
            "판정": "🚨 위험" if is_anomaly else "🟢 정상",
            "위험도": probs,
            "시간": last_time
        }
    except Exception as e:
        return f"error: {str(e)}"

# ==========================================
# 8. 메인 실행부 (UI 구성)
# ==========================================
st.sidebar.title("🛠️ 공정 필터링")

# 월 목록 로드 및 선택
month_list = get_available_months()
if not month_list:
    st.error("데이터 파일에 유효한 날짜 데이터가 없습니다.")
    st.stop()

selected_month = st.sidebar.selectbox("📅 분석 월 선택", month_list, key="sb_month")

# 데이터 로드
df_filtered_month, df_last_month = load_data_by_month(selected_month)
models_dict = load_all_models()

# 세션 스테이트 관리
if 'df' not in st.session_state or st.session_state.get('current_month') != selected_month:
    st.session_state['df'] = df_filtered_month
    st.session_state['current_month'] = selected_month

df_final = st.session_state['df']

# 설비 선택 사이드바
if not df_final.empty:
    machine_list = sorted(df_final['MACHNO'].unique().tolist())
    selected_machine = st.sidebar.selectbox("🏭 상세 분석 설비 (기준)", machine_list, key="sb_machine")
else:
    st.sidebar.warning("선택한 월에 데이터가 없습니다.")
    st.stop()

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
                fig_compare.update_layout(height=400, margin=dict(t=30), 
                                          xaxis=dict(tickformat="%d일", dtick=86400000.0),
                                          legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_compare, width='stretch')

    # [3] 실시간 관제 센터
    st.divider()
    st.subheader("🏭 실시간 설비 이상 징후 관제 센터 (최근 7일)")

    week_df, start_day, end_day = load_latest_week_data()

    if not week_df.empty:
        st.write(f"📅 **관제 기간:** {start_day} ~ {end_day} (최근 7일간)")
        
        all_mach_list = sorted(week_df['MACHNO'].unique())
        
        # 5개씩 나누어 표시
        cols = st.columns(5)
        cols2 = st.columns(5)
        all_cols = cols + cols2
        
        for i, mach in enumerate(all_mach_list):
            if i >= 10: break # 최대 10개까지만 표시 (UI 보호)
            
            with all_cols[i]:
                with st.container(border=True):
                    m_week_data = week_df[week_df['MACHNO'] == mach]
                    
                    st.markdown(f"### {mach}")
                    
                    status = get_machine_status(mach, m_week_data, models_dict)
                    
                    if isinstance(status, dict):
                        color = "red" if "위험" in status['판정'] else "green"
                        st.markdown(f"<h2 style='text-align: center; color: {color};'>{status['판정']}</h2>", unsafe_allow_html=True)
                        st.metric("현재 위험도", f"{status['위험도']:.1%}")
                        st.caption(f"🕒 최종: {status['시간']}")
                    elif status == "empty":
                        st.warning("데이터 없음")
                    else:
                        st.error("분석 불가")
    else:
        st.error("최근 7일간의 데이터를 찾을 수 없습니다.")

# ==============================================================================
# TAB 2: 설비 상세 리포트
# ==============================================================================
with tab_detail:
    st.markdown(f"### 🔍 {selected_machine} 설비 정밀 분석 리포트")
    m_df = df_filtered_month[df_filtered_month['MACHNO'] == selected_machine].sort_values('Timestamp_사출')
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


