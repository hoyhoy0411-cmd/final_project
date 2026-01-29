import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import math
import joblib
from sklearn.preprocessing import PolynomialFeatures
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from io import BytesIO
import requests

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# --- 구글 드라이브 파일 ID 설정 ---
FILE_IDS = {
    "models": "1ozTrBdUE-4fq-wghLSCInzzuXHtVcmTc",
    "shot_data": "1f_hVACgK9030zpHjiw4huE2ZV4xRcbY_",
    "preprocessed": "1_eROsCEx2FR6cbOOBi-d07xb2wWuXcsy",
    "classification": "11nuPSVeJSFEk5E3wMkFn-lnTaY9p7Vlt"
}

def get_drive_url(file_id):
    return f'https://drive.google.com/uc?id={file_id}'


# --- 라이브러리 없이 직접 분석하는 함수 (대체 로직) ---
@st.cache_resource
def load_all_models():
    try:
        url = get_drive_url(FILE_IDS["models"])
        response = requests.get(url)
        # joblib은 파일 객체나 바이트 스트림을 직접 읽을 수 있습니다.
        data = joblib.load(BytesIO(response.content))
        if not data:
            st.error("모델 파일이 비어 있습니다.")
        return data
    except Exception as e:
        st.error(f"모델 파일 로드 실패: {e}")
        return {}
    
def get_realtime_status_with_ai(full_df, models_dict):
    """
    분석 월 필터와 관계없이, 로드된 전체 데이터(full_df)에서 
    각 설비별 가장 마지막(최신) 데이터를 찾아 AI 판정을 수행합니다.
    """
    results = []
    if full_df.empty or not models_dict:
        return pd.DataFrame()
        
    # 모델이 존재하는 모든 설비 또는 데이터에 존재하는 모든 설비 대상
    all_machines = sorted(full_df['MACHNO'].unique())
    
    for mach in all_machines:
        if mach in models_dict:
            # 해당 설비의 전체 기록 중 가장 최신 10건 추출 (잔차 계산용)
            m_data = full_df[full_df['MACHNO'] == mach].sort_values('Timestamp_사출').tail(10)
            
            if len(m_data) < 1: 
                continue
            
            info = models_dict[mach]
            features = info['features']
            
            try:
                # 데이터 전처리
                X = m_data[features].copy()
                X_res = X - X.rolling(window=10, min_periods=1).mean()
                X_res.columns = [f"{c}_resid" for c in X_res.columns]
                X_combined = pd.concat([X, X_res], axis=1)
                
                # Polynomial 적용 로직
                if mach == 'G06':
                    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                    top_cols = [c for c in X.columns if any(s in c for s in ['금형온도', '최소쿠션', '충진시간', '압력'])][:6]
                    X_poly = poly.fit_transform(X_combined[top_cols])
                    X_final = np.hstack([X_combined.values, X_poly])
                elif info['group'] == '난조군':
                    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                    top_cols = [c for c in X.columns if any(s in c for s in ['금형온도', '최소쿠션', '충진시간'])][:5]
                    X_poly = poly.fit_transform(X_combined[top_cols])
                    X_final = np.hstack([X_combined.values, X_poly])
                else:
                    X_final = X_combined.values

                # 모델 예측 (가장 최신 샷 1건)
                X_scaled = info['scaler'].transform(X_final[[-1]])
                
                # 만약 모델이 XGBoost라면 안전하게 CPU에서 실행되도록 보장
                model = info['model']
                
                # 예측 수행
                probs = model.predict_proba(X_scaled)[:, 1][0]
                d_score = info['lof'].predict(X_scaled)[0]
                
                refined_prob = probs
                if d_score == 1:
                    refined_prob *= info.get('lof_penalty', 1.0)
                
                is_anomaly = refined_prob >= info['threshold']
                latest_res = X_res.iloc[-1].abs()
                top_var = latest_res.idxmax()
                
                # 최신 데이터의 시간 정보 추출
                last_update = m_data['Timestamp_사출'].iloc[-1].strftime('%m/%d %H:%M')
                
                results.append({
                    "설비": mach,
                    "판정": "🚨 위험" if is_anomaly else "🟢 정상",
                    "위험도": refined_prob,
                    "주요요인": str(top_var).replace('_resid', '').split('_')[0],
                    "변동폭": round(latest_res.max(), 3),
                    "업데이트": last_update
                })
            except Exception as e:
                continue
                
    return pd.DataFrame(results)


# --- 2. 페이지 설정 ---
st.set_page_config(page_title="공정 품질 KPI 대시보드", layout="wide")

# --- 3. 데이터 로드 함수 (캐싱 적용) ---
@st.cache_data
def load_latest_week_data():
    try:
        url = get_drive_url(FILE_IDS["preprocessed"])
        df = pd.read_csv(url)
        df['Timestamp_사출'] = pd.to_datetime(df['Timestamp_사출'])
        latest_date = df['Timestamp_사출'].max()
        start_date = (latest_date - pd.Timedelta(days=6)).replace(hour=0, minute=0, second=0)
        week_df = df[(df['Timestamp_사출'] >= start_date) & (df['Timestamp_사출'] <= latest_date)].copy()
        return week_df, start_date.date(), latest_date.date()
    except Exception as e:
        st.error(f"주간 데이터 로드 실패: {e}")
        return pd.DataFrame(), None, None

@st.cache_resource
# --- 2. 상태 판정 함수 (단 하나만 유지) ---
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
        # 1. 원본 특징(features) 추출
        X_input = m_data[info['features']].tail(1).values.astype(np.float32)
        
        # 2. 모델이 기대하는 피처 수 확인 (StandardScaler 기준)
        expected_features = info['scaler'].n_features_in_
        current_features = X_input.shape[1]
        
        # 3. 부족한 피처 수만큼 0으로 채우기 (Padding)
        if current_features < expected_features:
            padding_size = expected_features - current_features
            padding = np.zeros((1, padding_size), dtype=np.float32)
            X_final = np.hstack([X_input, padding])
        else:
            X_final = X_input[:, :expected_features] # 혹시 더 많으면 자름

        # 4. 스케일링 및 예측
        X_scaled = info['scaler'].transform(X_final)
        
        model = info['model']
        if hasattr(model, 'get_booster'):
            model.get_booster().set_param({'predictor': 'cpu_predictor', 'device': 'cpu'})
        
        probs = model.predict_proba(X_scaled)[:, 1][0]
        
        # LOF 점수 반영
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

@st.cache_data
def load_process_data():
    try:
        df = pd.read_csv('전처리데이터.csv')
        df['Timestamp_사출'] = pd.to_datetime(df['Timestamp_사출'])
        return df
    except:
        return pd.DataFrame()

# 데이터 로드 실행
# [수정 후]
# 1. 월 목록만 가져오는 가벼운 함수를 따로 만듭니다.
@st.cache_data
def get_available_months():
    """월 목록만 가볍게 가져오기"""
    try:
        url = get_drive_url(FILE_IDS["shot_data"])
        # 전체를 다 읽지 않고 필요한 컬럼만 읽기
        df_temp = pd.read_csv(url, usecols=['Timestamp_사출'])
        return sorted(pd.to_datetime(df_temp['Timestamp_사출']).dt.strftime('%Y-%m').unique(), reverse=True)
    except Exception as e:
        st.error(f"월 목록 로드 실패: {e}")
        return []

@st.cache_data
def load_data_by_month(selected_month):
    """선택된 월의 데이터만 읽어오기"""
    try:
        url = get_drive_url(FILE_IDS["shot_data"])
        df = pd.read_csv(url) 
        df['Timestamp_사출'] = pd.to_datetime(df['Timestamp_사출'])
        df['YearMonth'] = df['Timestamp_사출'].dt.strftime('%Y-%m')
        
        df = df[df['YearMonth'] == selected_month].copy()
        df['Date'] = df['Timestamp_사출'].dt.date 
        df['NG'] = df['NG'].astype(str)
        df['Result'] = df['NG'].apply(lambda x: '불량(NG)' if x in ['1', '1.0', 'NG'] else '정상(OK)')
        return df
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return pd.DataFrame()

def load_recent_process_data(n_per_machine=50):
    try:
        url = get_drive_url(FILE_IDS["preprocessed"])
        df = pd.read_csv(url)
        df['Timestamp_사출'] = pd.to_datetime(df['Timestamp_사출'])
        recent_df = df.sort_values('Timestamp_사출').groupby('MACHNO').tail(n_per_machine)
        return recent_df
    except Exception as e:
        st.error(f"실시간 데이터 로드 실패: {e}")
        return pd.DataFrame()

# ==========================================
# 데이터 로드 실행 (사이드바 연동)
# ==========================================
st.sidebar.title("🛠️ 공정 필터링")

# 1. 사용 가능한 월 목록 가져오기
month_list = get_available_months()
# 2. 첫 번째 selectbox: 분석 월 선택
selected_month = st.sidebar.selectbox("📅 분석 월 선택", month_list, key="sb_month")

# 데이터 로드 실행 (선택된 월에 따라)
df_filtered_month = load_data_by_month(selected_month)
models_dict = load_all_models()
# 실시간 관제를 위해 설비별 최신 데이터 n건 로드
full_df = load_recent_process_data(50) 

# 세션 스테이트 관리 (데이터 업데이트 확인용)
if 'df' not in st.session_state or st.session_state.get('current_month') != selected_month:
    st.session_state['df'] = df_filtered_month
    st.session_state['current_month'] = selected_month

df_final = st.session_state['df']

# 데이터가 비어있지 않을 때만 하단 필터(설비 선택) 표시
if not df_final.empty:
    machine_list = sorted(df_final['MACHNO'].unique().tolist())
    # 두 번째 selectbox: 상세 분석용 설비 선택
    selected_machine = st.sidebar.selectbox("🏭 상세 분석 설비 (기준)", machine_list, key="sb_machine")
else:
    st.sidebar.warning("선택한 월에 데이터가 없습니다.")
    st.stop()

# 탭 구성
tab_kpi, tab_detail, tab_analysis = st.tabs(["🚀 공장 전체 KPI", "🔍 설비 상세 리포트", "🚨 불량 원인 분석"])

# ==============================================================================
# TAB 1: 공장 전체 KPI (순서 변경: 지표 -> 차트 -> 관제센터)
# ==============================================================================
with tab_kpi:
    st.title("🚀 공정 품질 핵심 성과 지표 (KPI)")
    st.info(f"📍 **{selected_month}** 공장 전체 설비 가동 현황 요약")

    # [1] 핵심 메트릭 (전월 대비 증감)
    try:
        current_date = pd.to_datetime(selected_month + "-01")
        last_month_str = (current_date - pd.offsets.MonthBegin(1)).strftime('%Y-%m')
        df_last_month = df_final[df_final['YearMonth'] == last_month_str]
        
        # 현재 월
        total_qty = len(df_filtered_month)
        total_ng = len(df_filtered_month[df_filtered_month['Result'] == '불량(NG)'])
        total_ok = total_qty - total_ng
        avg_defect_rate = (total_ng / total_qty * 100) if total_qty > 0 else 0

        # 전월 비교
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
        else:
            has_history = False
    except:
        has_history = False
        total_qty, total_ok, total_ng, avg_defect_rate = 0, 0, 0, 0

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
            df_compare = df_filtered_month[df_filtered_month['MACHNO'].isin(compare_machines)]
            trend_compare = df_compare.groupby(['Date', 'MACHNO']).size().reset_index(name='Count')
            
            fig_compare = px.line(trend_compare, x='Date', y='Count', color='MACHNO', markers=True)
            fig_compare.update_layout(height=400, margin=dict(t=30), 
                                      xaxis=dict(tickformat="%d일", dtick=86400000.0),
                                      legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_compare, width='stretch')

    # [3] 실시간 관제 센터 (하단 배치)
    st.divider()

    st.subheader("🏭 실시간 설비 이상 징후 관제 센터 (최근 7일)")

    # 1. 일주일치 데이터 로드
    week_df, start_day, end_day = load_latest_week_data()

    if not week_df.empty:
        st.write(f"📅 **관제 기간:** {start_day} ~ {end_day} (최근 7일간)")
        
        all_mach_list = [f"G{str(i).zfill(2)}" for i in range(1, 11)] 
        cols = st.columns(5)
        cols2 = st.columns(5)
        all_cols = cols + cols2
        
        for i, mach in enumerate(all_mach_list):
            with all_cols[i]:
                # 설비별 테두리 컨테이너
                with st.container(border=True):
                    # 해당 설비의 7일치 데이터
                    m_week_data = week_df[week_df['MACHNO'] == mach]
                    
                    # 최신 상태 분석 (기존 함수 활용)
                    status = get_machine_status(mach, week_df, models_dict)
                    
                    st.markdown(f"### {mach}")
                    
                    if isinstance(status, dict):
                        color = "red" if "위험" in status['판정'] else "green"
                        st.markdown(f"<h2 style='text-align: center; color: {color};'>{status['판정']}</h2>", unsafe_allow_html=True)
                        
                        # 추가 정보: 주간 가동 샷 수
                        weekly_shots = len(m_week_data)
                        st.metric("현재 위험도", f"{status['위험도']:.1%}")
                        st.caption(f"📊 주간 생산량: {weekly_shots:,} 샷")
                        st.caption(f"🕒 최종 업데이트: {status['시간']}")
                    
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

    # 도넛 차트 & 라인 차트
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

    # 품질 상태 진단
    st.write("---")
    st.write(f"💡 {selected_machine} 품질 진단")
    
    # 간단한 통계적 임계값 (전체 평균 기준)
    p_avg = 0.05 # 임의 기준값 (데이터에 따라 동적 조정 가능)
    if not df_final.empty:
        p_avg = (df_final['Result'] == '불량(NG)').mean()
    
    target_warn = p_avg * 1.5 * 100
    
    if m_rate <= p_avg * 100:
        st.success(f"✅ **안정**: 불량률이 공장 평균({p_avg*100:.2f}%) 이하입니다.")
    elif m_rate <= target_warn:
        st.warning(f"⚠️ **주의**: 불량률이 공장 평균을 상회합니다.")
    else:
        st.error(f"🚨 **위험**: 불량률이 관리 한계를 초과했습니다.")

    # 일별 불량률 추이 (비가동 구간 표시)
    st.subheader("📈 일별 불량률 추이 (비가동 구간 포함)")
    
    start_date = m_df['Date'].min()
    end_date = m_df['Date'].max()
    all_days = pd.date_range(start=start_date, end=end_date).date
    
    daily_stats = m_df.groupby('Date')['Result'].value_counts().unstack(fill_value=0)
    if '불량(NG)' not in daily_stats.columns: daily_stats['불량(NG)'] = 0
    if '정상(OK)' not in daily_stats.columns: daily_stats['정상(OK)'] = 0
    
    daily_stats['Rate'] = (daily_stats['불량(NG)'] / (daily_stats['불량(NG)'] + daily_stats['정상(OK)'])) * 100
    daily_stats = daily_stats.reset_index()

    # 비가동일 식별
    existing_days = daily_stats['Date'].tolist()
    missing_days = [d for d in all_days if d not in existing_days]

    fig_line = px.line(daily_stats, x='Date', y='Rate', markers=True, text=daily_stats['Rate'].apply(lambda x: f'{x:.1f}%'))
    
    # 비가동 구간 음영 처리
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
    
    # 날짜 필터
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
        
        # 분석 대상 컬럼 (예시)
        analyze_cols = ['Cycle Time', '사출 시간', '충진 시간', '최소 쿠션', '피크압_주성분', '보압 완료 위치']
        # 실제 데이터프레임에 있는 컬럼만 필터링
        valid_cols = [c for c in analyze_cols if c in m_df.columns]
        
        if valid_cols:
            # 평균 비교
            ng_mean = m_ng_df[valid_cols].mean()
            ok_mean = m_ok_df[valid_cols].mean() if not m_ok_df.empty else m_df[valid_cols].mean()

            # 메트릭 표시
            cols = st.columns(len(valid_cols))
            for i, col in enumerate(valid_cols):
                diff = ng_mean[col] - ok_mean[col]
                cols[i].metric(col, f"{ng_mean[col]:.2f}", f"{diff:+.2f}", delta_color="inverse")

            # 레이더 차트
            st.write("#### 🕸️ 정상 대비 변동 비율 (%)")
            ratios = [(ng_mean[c] / ok_mean[c] * 100) if ok_mean[c] != 0 else 0 for c in valid_cols]
            
            # 차트용 데이터 (닫힌 도형을 위해 첫 데이터 반복)
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