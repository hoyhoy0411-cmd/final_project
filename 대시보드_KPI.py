import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import warnings
import urllib3
import os
import gdown

# --- 1. 경고 및 보안 설정 ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

# --- 2. 페이지 설정 ---
st.set_page_config(page_title="공정 품질 KPI 대시보드", layout="wide")

# --- 3. 상수 및 경로 설정 ---
# 관제 로직 제거로 'pre' 데이터는 필요 없어져서 로드하지 않도록 최적화했습니다.
MONTHLY_CONFIG = {
    "2023-09": "1ed7eRVk25aTBN1aVsvzEZZQrbGPN0lbY",
    "2023-10": "1gYE2xh6TcrtlNd6rAYe0MrjOmkTvz9vL",
    "2023-11": "1QPNxQffDP3F22KjyXhcORSxgY_jYha96"
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

# --- 5. 데이터 로드 및 캐싱 함수 (최적화 적용) ---
@st.cache_data
def load_monthly_data(year_month):
    file_id = MONTHLY_CONFIG.get(year_month)
    if not file_id: 
        return pd.DataFrame()

    shot_file = f"shot_{year_month}.parquet"
    
    # 다운로드
    download_from_gdrive(file_id, shot_file)

    try:
        # 데이터 로드
        df = pd.read_parquet(shot_file)
        
        # [메모리 최적화 1] 컬럼명 공백 제거 및 날짜 변환
        df.columns = [c.strip() for c in df.columns]
        if 'Timestamp_사출' in df.columns:
            df['Timestamp_사출'] = pd.to_datetime(df['Timestamp_사출'], errors='coerce')
        
        # [메모리 최적화 2] 수치형 데이터 다운캐스팅 (64bit -> 32bit)
        fcols = df.select_dtypes('float64').columns
        df[fcols] = df[fcols].astype('float32')
        
        icols = df.select_dtypes('int64').columns
        df[icols] = df[icols].astype('int32')

        # [메모리 최적화 3] 범주형 데이터 변환 (Object -> Category)
        # 반복되는 문자열(설비명, 결과 등)을 카테고리로 변환하면 메모리가 대폭 감소합니다.
        for col in ['MACHNO', 'Result', 'NG']:
            if col in df.columns:
                df[col] = df[col].astype('category')

        # Result 컬럼 생성 (없을 경우)
        if 'Result' not in df.columns and 'NG' in df.columns:
            # map 사용 시 category 구조 유지를 위해 주의 필요하나, 여기선 단순 처리
            temp_result = df['NG'].map({0: '정상(OK)', 1: '불량(NG)'})
            df['Result'] = temp_result.astype('category')
            
        return df
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return pd.DataFrame()

# ==========================================
# --- 6. 메인 실행부 (UI 구성) ---
st.sidebar.title(" 공정 필터링")

# 1. 월 선택
raw_month_list = sorted(list(MONTHLY_CONFIG.keys()), reverse=True)
display_month_map = {
    m: m.replace("2023-", "2025년 ").replace("-", "") + "월" 
    for m in raw_month_list
}

selected_display_month = st.sidebar.selectbox(
    " 분석 월 선택", 
    options=list(display_month_map.values())
)
selected_month = [k for k, v in display_month_map.items() if v == selected_display_month][0]

# 2. 데이터 로드 (shot 데이터만 로드)
df_shot = load_monthly_data(selected_month)

if df_shot.empty:
    st.error("데이터를 불러올 수 없습니다.")
    st.stop()

# 3. 설비 선택
machine_list = sorted(df_shot['MACHNO'].unique().tolist())
selected_machine = st.sidebar.selectbox("🏗️ 분석 대상 설비", options=machine_list)

# 4. 전월 데이터 로드 (KPI 비교용, 선택사항)
current_idx = raw_month_list.index(selected_month)
if current_idx + 1 < len(raw_month_list):
    last_month_key = raw_month_list[current_idx + 1]
    df_last_month = load_monthly_data(last_month_key)
else:
    df_last_month = pd.DataFrame()

# --- 데이터 준비 ---
df_filtered_month = df_shot
df_final = df_shot

# 탭 구성 (관제 센터 탭 제거됨)
tab_kpi, tab_detail, tab_analysis = st.tabs([" 공장 전체 KPI", " 설비 상세 리포트", " 불량 원인 분석"])

# ==============================================================================
# TAB 1: 공장 전체 KPI
# ==============================================================================
with tab_kpi:
    st.title(" 공정 품질 핵심 성과 지표 (KPI)")
    st.info(f" 공장 전체 설비 가동 현황 요약")
    
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
        st.subheader(" 설비별 불량률 순위")
        # value_counts는 category 타입에서 매우 빠름
        m_stats = df_filtered_month.groupby('MACHNO', observed=True)['Result'].value_counts().unstack(fill_value=0)
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
        st.subheader(" 설비별 생산량 추이 비교")
        compare_machines = st.multiselect("비교 대상 설비", options=machine_list, default=[selected_machine])
        
        if compare_machines:
            df_compare = df_filtered_month[df_filtered_month['MACHNO'].isin(compare_machines)].copy()
            if 'Timestamp_사출' in df_compare.columns:
                df_compare['Date'] = df_compare['Timestamp_사출'].dt.date
            
                trend_compare = df_compare.groupby(['Date', 'MACHNO'], observed=True).size().reset_index(name='Count')
                fig_compare = px.line(trend_compare, x='Date', y='Count', color='MACHNO', markers=True)
                fig_compare.update_layout(
                    height=400, 
                    margin=dict(t=30), 
                    xaxis=dict(tickformat="%d일", dtick=86400000.0 * 5, tickangle=0),
                    legend=dict(orientation="h", y=1.1)
                )
                st.plotly_chart(fig_compare, width='stretch')

# ==============================================================================
# TAB 2: 설비 상세 리포트
# ==============================================================================
with tab_detail:
    st.markdown(f"###  {selected_machine} 설비 정밀 분석 리포트")
    m_df = df_filtered_month[df_filtered_month['MACHNO'] == selected_machine].sort_values('Timestamp_사출')
    if 'Timestamp_사출' in m_df.columns:
        m_df['Date'] = m_df['Timestamp_사출'].dt.date

    # --- [수정 시작] 설비 상세 메트릭 (전월 대비 증감 추가) ---
    
    # 1. 이번 달 지표 계산
    m_t = len(m_df)
    m_ng = len(m_df[m_df['Result'] == '불량(NG)'])
    m_ok = m_t - m_ng
    m_rate = (m_ng / m_t * 100) if m_t > 0 else 0

    # 2. 전월 데이터 계산 (비교용)
    has_m_history = False
    lm_t, lm_ok, lm_ng, lm_rate = 0, 0, 0, 0

    if not df_last_month.empty:
        # 전월 데이터에서 현재 선택된 설비만 필터링
        m_last_df = df_last_month[df_last_month['MACHNO'] == selected_machine]
        
        if not m_last_df.empty:
            lm_t = len(m_last_df)
            lm_ng = len(m_last_df[m_last_df['Result'] == '불량(NG)'])
            lm_ok = lm_t - lm_ng
            lm_rate = (lm_ng / lm_t * 100) if lm_t > 0 else 0
            has_m_history = True

    # 3. 메트릭 표시 (Delta 적용)
    m1, m2, m3, m4 = st.columns(4)

    if has_m_history:
        m1.metric("생산량", f"{m_t:,}건", delta=f"{m_t - lm_t:,}건")
        m2.metric("양품 수량", f"{m_ok:,}건", delta=f"{m_ok - lm_ok:,}건")
        # 불량 수량과 불량률은 낮을수록 좋으므로 delta_color="inverse" (증가하면 빨강, 감소하면 초록)
        m3.metric("불량 수량", f"{m_ng:,}건", delta=f"{m_ng - lm_ng:,}건", delta_color="inverse")
        m4.metric("불량률", f"{m_rate:.2f}%", delta=f"{m_rate - lm_rate:.2f}%", delta_color="inverse")
    else:
        # 전월 데이터가 없을 경우 기존 방식대로 표시
        m1.metric("생산량", f"{m_t:,}건")
        m2.metric("양품 수량", f"{m_ok:,}건")
        m3.metric("불량 수량", f"{m_ng:,}건")
        m4.metric("불량률", f"{m_rate:.2f}%")
    
    st.write("")

    c1, c2 = st.columns([1, 2.5])
    with c1:
        st.write(f"#####  품질 비율")
        fig_pie = px.pie(m_df, names='Result', hole=0.6, color='Result',
                          color_discrete_map={'정상(OK)': '#2ecc71', '불량(NG)': '#e74c3c'})
        fig_pie.update_layout(height=300, showlegend=True, 
                              legend=dict(orientation="h", y=-0.1), margin=dict(t=20, b=20))
        fig_pie.add_annotation(text=f"{m_rate:.1f}%", x=0.5, y=0.5, font_size=20, showarrow=False)
        st.plotly_chart(fig_pie, width='stretch')

    with c2:
        st.write(f"#####  일별 생산 추이")
        daily_prod = m_df.groupby(['Date', 'Result'], observed=True).size().reset_index(name='Count')
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
        st.success(f" **안정**: 불량률이 공장 평균({p_avg*100:.2f}%) 이하입니다.")
    elif m_rate <= target_warn:
        st.warning(f" **주의**: 불량률이 공장 평균을 상회합니다.")
    else:
        st.error(f" **위험**: 불량률이 관리 한계를 초과했습니다.")

    # 일별 불량률 추이
    st.subheader(" 일별 불량률 추이 (비가동 구간 포함)")
    
    if not m_df.empty:
        start_date = m_df['Date'].min()
        end_date = m_df['Date'].max()
        all_days = pd.date_range(start=start_date, end=end_date).date
        
        daily_stats = m_df.groupby('Date', observed=True)['Result'].value_counts().unstack(fill_value=0)
        if '불량(NG)' not in daily_stats.columns: daily_stats['불량(NG)'] = 0
        if '정상(OK)' not in daily_stats.columns: daily_stats['정상(OK)'] = 0
        
        daily_stats['Rate'] = (daily_stats['불량(NG)'] / (daily_stats['불량(NG)'] + daily_stats['정상(OK)'])) * 100
        daily_stats = daily_stats.reset_index()

        existing_days = daily_stats['Date'].tolist()
        missing_days = [d for d in all_days if d not in existing_days]

        fig_line = px.line(daily_stats, x='Date', y='Rate', markers=True, text=daily_stats['Rate'].apply(lambda x: f'{x:.1f}'))
        
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
    selected_date_analysis = st.selectbox(" 분석 기간 선택", available_dates, key="analysis_date")

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
        st.success(f" {label}에는 불량 데이터가 없습니다.")











