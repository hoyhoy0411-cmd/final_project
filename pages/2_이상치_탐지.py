import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import PolynomialFeatures
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import gdown
from io import BytesIO

BASE_DIR = os.getcwd()

# --- [페이지 설정] ---
st.set_page_config(page_title="사출 공정 통합 모니터링 시스템", layout="wide")

# ==========================================
# [설정 및 ID 매핑]
# ==========================================

# 1. 모델 파일 ID 매핑
MODEL_IDS = {
    'D01': '1E7PwqwTu64FJjqWOIYWrodfC6migkKbR',
    'D10': '1GYsZXdCLc7izqYag6gQ40q81CiQN7WBR',
    'F09': '1yPlNAUEatopQbHqv1BInnD4qP3sRRzBZ',
    'F11': '1uTdb1Z9L04sJYq93KO0zXuk1VJWQWZ9Y',
    'G01': '1zHx1w2GPF902iz3eKMziy34NuLMu1adG',
    'G02': '1_xQvj0l2hcCcuDrisQxJZ5Mss6Fep96k',
    'G03': '1sDcIxxQ1MwXIyXfDq-3t6tkDwMePY46t',
    'G05': '17jPNOz-SeLwh9bRd8kDrsPcsGmBCXFAS',
    'G06': '1dF_ZCpJsXTFK16JXFKx5PbFYNqHF1p1o',
    'G10': '1LsTOoSUSUPqqx_088wv8YvHl9RUDbvR9'
}

# 2. 데이터 파일 ID 매핑
DATA_IDS = {
    '9월': {
        'prep': '1rw9Kv055wD1w_HLO1Pdizbk4CEfHAeiF',
        'shot': '1ed7eRVk25aTBN1aVsvzEZZQrbGPN0lbY'
    },
    '10월': {
        'prep': '12ZKLaEUw09JGno05isB7QQW_zMRicl-l',
        'shot': '1gYE2xh6TcrtlNd6rAYe0MrjOmkTvz9vL'
    },
    '11월': {
        'prep': '1xaObr2lANjOsHENGkslUqNiHOuHEaVFb',
        'shot': '1QPNxQffDP3F22KjyXhcORSxgY_jYha96'
    }
}

# 3. 그룹 정의
GROUPS = {
    '고성능군': ['D01', 'F09', 'G03'],
    '안정군': ['D10', 'F11', 'G05', 'G10'],
    '난조군': ['G01', 'G02', 'G06'] 
}

# ==========================================
# [공통 데이터 및 함수 정의]
# ==========================================

# 파일 다운로드 헬퍼 함수
def download_file_from_google_drive(file_id, output_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(output_path):
        with st.spinner(f'파일 다운로드 중: {output_path}...'):
            gdown.download(url, output_path, quiet=True)
    return output_path

# 1. 모델 로드 함수
@st.cache_resource
def load_single_model(mach_no):
    if mach_no not in MODEL_IDS:
        return None
        
    file_id = MODEL_IDS[mach_no]
    file_name = f"model_{mach_no}.pkl"
    file_path = os.path.join(BASE_DIR, file_name)
    
    if not os.path.exists(file_path):
        with st.spinner(f'{mach_no} 모델 다운로드 중...'):
            download_file_from_google_drive(file_id, file_path)
            
    try:
        return joblib.load(file_path)
    except Exception as e:
        st.error(f"{mach_no} 모델 로드 실패: {e}")
        return None
    
# 2. 데이터 로드 함수
@st.cache_data
def load_local_data(month_key):
    if month_key not in DATA_IDS:
        return pd.DataFrame()
    
    file_id = DATA_IDS[month_key]['prep']
    file_name = f"prep_data_{month_key}.parquet"
    file_path = os.path.join(BASE_DIR, file_name)

    if not os.path.exists(file_path):
        with st.spinner(f'{month_key} 데이터 다운로드 중...'):
            download_file_from_google_drive(file_id, file_path)

    try:
        df = pd.read_parquet(file_path)
        df.columns = [col.strip() for col in df.columns]
        
        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')
        
        target_col = 'Timestamp_사출'
        if target_col in df.columns:
            df[target_col] = pd.to_datetime(df[target_col], errors='coerce')
            df = df.dropna(subset=[target_col])
        else:
            st.error(f"'{target_col}' 컬럼을 찾을 수 없습니다.")
            
        return df
    except Exception as e:
        st.error(f"데이터 로드 실패 ({month_key}): {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return pd.DataFrame()

# ==========================================
# [메인 실행 함수]
# ==========================================
def main():
    # 1. 사이드바: 데이터 로드 및 설비 선택
    with st.sidebar:
        st.header("⚙️ 분석 설정")
        month_options = list(DATA_IDS.keys()) 
        selected_month = st.selectbox("1. 분석 월 선택", month_options)
        
        full_df = load_local_data(selected_month)        

        data_ready = False
        if not full_df.empty:
            available_machines = sorted(full_df['MACHNO'].unique())
            target_mach = st.selectbox("2. 분석 설비 선택", available_machines)
            
            model_info = load_single_model(target_mach)
            if model_info:
                st.success(f" {target_mach} 모델 로드 완료")
                data_ready = True
            else:
                st.warning(f" {target_mach} 모델 없음")
        else:
            st.error("데이터를 불러올 수 없습니다.")

        # [요구사항 4] 사이드바 주의사항 추가
        st.markdown("---")
        st.error(
            """
            **주의사항**
            
            **반드시 데이터가 모두 로드된 후에 분석 월과 분석 설비를 선택해주세요.** 
            스트림릿의 램 용량 부족으로 인해 서버에 과부하가 걸릴 수 있습니다.
            """
        )

    # --- [메인 화면] ---
    st.title(" 사출 공정 스마트 상세 분석 대시보드 (이상치 탐지)")

    if data_ready and not full_df.empty:
        st.subheader(f" {target_mach} 설비 상세 분석 ({selected_month})")
        st.info(f" **{selected_month} {target_mach} 설비**의 전체 생산 Shot 중에서 공정 과정에 이상이 생겼다고 판단되는 불량입니다.\n\n(설비의 특정 센서값이 이전 패턴과 달라졌음을 의미합니다.)")

        # 1. 데이터 필터링
        m_df = full_df[full_df['MACHNO'] == target_mach].copy().reset_index(drop=True)
        
        # 2. 분석 수행
        info = model_info 
        
        # --- 전처리 및 Feature Engineering ---
        trained_features = info['features']
        X = pd.DataFrame(index=m_df.index)
        
        for col in trained_features:
            X[col] = m_df[col] if col in m_df.columns else 0
        X = X.fillna(0)
        
        # 잔차(Residual) 생성 (전체 기간)
        X_res = X - X.rolling(window=10, min_periods=1).mean()
        X_res.columns = [f"{c}_resid" for c in X_res.columns]
        X_combined = pd.concat([X, X_res], axis=1)

        # Polynomial Features
        if target_mach == 'G06':
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

        # --- AI 모델 예측 ---
        X_scaled = info['scaler'].transform(X_final)
        
        try:
            info['model'].set_params(device="cpu")
        except:
            pass 

        probs = info['model'].predict_proba(X_scaled)[:, 1]
        
        # Anomaly Detection (LOF)
        d_scores = info['lof'].predict(X_scaled) 
        refined_probs = probs.copy()
        refined_probs[d_scores == 1] *= info['lof_penalty']
        
        # 결과 병합
        m_df['predict_prob'] = refined_probs
        m_df['y_pred'] = (refined_probs >= info['threshold']).astype(int)
        
        # --- 대시보드 UI ---
        col1, col2, col3 = st.columns(3)
        col1.metric("설비 그룹", info.get('group', 'Unknown'))
        col2.metric("전체 사이클", f"{len(m_df):,} Shot")
        col3.metric("평균 이상 발생률", f"{(m_df['y_pred'].sum() / len(m_df) * 100):.2f}%", delta_color="inverse")
        st.divider()
        
        col_sub1, col_sub2 = st.columns([1, 2])

        # ==============================================================================
        # 좌측: 월간 종합 상태 요약 (정상 vs 이상 개수)
        # ==============================================================================
        with col_sub1:
            st.subheader(f" {selected_month} 종합 판정 결과")

            total_ng = int(m_df['y_pred'].sum())
            total_ok = int(len(m_df) - total_ng)
            
            # 카드의 색상은 불량률이 높으면 빨간색, 아니면 초록색/파란색 계열
            ng_rate = total_ng / len(m_df)
            card_color = "#dc3545" if ng_rate > 0.2 else "#28a745" # 20% 기준
            
            st.markdown(f"""
                <div style="background-color:{card_color}; padding:20px; border-radius:15px; text-align:center; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
                    <h3 style="color:white; margin-bottom: 20px;">{target_mach} 설비 판정 현황</h3>
                    <div style="display: flex; justify-content: space-around; align-items: center; color: white;">
                        <div>
                            <div style="font-size: 1.2rem; font-weight: bold;">🟢 정상(OK)</div>
                            <div style="font-size: 2.5rem; font-weight: bold;">{total_ok:,}</div>
                            <div style="font-size: 0.9rem;">Shot</div>
                        </div>
                        <div style="width: 2px; height: 60px; background-color: rgba(255,255,255,0.5);"></div>
                        <div>
                            <div style="font-size: 1.2rem; font-weight: bold;">🔴 이상(NG)</div>
                            <div style="font-size: 2.5rem; font-weight: bold;">{total_ng:,}</div>
                            <div style="font-size: 0.9rem;">Shot</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # ==============================================================================
        # 우측: 월간 변수 기여도 분석 (Top 5)
        # ==============================================================================
        with col_sub2:
            st.subheader(f" {selected_month} 주요 변동 요인 (Top 5)")
            
            # [요구사항 2] 주요 변동 요인 설명 추가
            st.caption(" 공정이 바뀌었다고 판단하는 지표 중 가장 많은 영향력을 가진 주요 5가지 변수입니다.")

            # 전체 기간 잔차의 절대값 평균을 구함 (변동성이 컸던 변수 찾기)
            # 컬럼명에서 _resid 제거하여 표시
            monthly_importance = X_res.abs().mean().sort_values(ascending=False).head(5)
            monthly_importance.index = [c.replace('_resid', '') for c in monthly_importance.index]
            
            fig_imp = go.Figure(go.Bar(
                x=monthly_importance.values,
                y=monthly_importance.index,
                orientation='h',
                marker=dict(color=monthly_importance.values, colorscale='Blues'),
                text=[f"{v:.4f}" for v in monthly_importance.values],
                textposition='auto'
            ))
            
            fig_imp.update_layout(
                xaxis_title="평균 변동량 (Mean Abs Residual)",
                yaxis=dict(autorange="reversed"), # 상위 항목이 위로 오게
                height=300,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        st.divider()
        
        # ==============================================================================
        # 하단: 일별 정상/불량 그래프 (날짜 슬라이더 제거)
        # ==============================================================================
        st.subheader(" 일별 생산 및 품질 현황")
        
        # [요구사항 3] 일별 현황 설명 추가
        st.info(" 이상치 탐지 모델의 하루 **정상 판단 Shot** 개수와 **이상 판단 Shot** 개수를 비교해볼 수 있습니다.\n\n단순 제품 불량이 아닌, **설비의 이상(공정 변화)을 판단하는 지표**입니다.")

        # 일별, 판정별 집계
        daily_df = m_df.copy()
        daily_df['Date'] = daily_df['Timestamp_사출'].dt.date
        
        # 0: 정상, 1: 불량 매핑
        daily_stats = daily_df.groupby(['Date', 'y_pred']).size().reset_index(name='Count')
        daily_stats['Status'] = daily_stats['y_pred'].map({0: '정상(OK)', 1: '이상(NG)'})
        
        if not daily_stats.empty:
            # 막대 차트 (Stacked)
            fig_daily = px.bar(
                daily_stats, 
                x='Date', 
                y='Count', 
                color='Status',
                color_discrete_map={'정상(OK)': '#28a745', '이상(NG)': '#dc3545'},
                text='Count',
                title=f"{selected_month} 일별 판정 결과 추이"
            )
            
            fig_daily.update_traces(textposition='inside', textfont_color='white')
            fig_daily.update_xaxes(
                type='category',  # 날짜를 카테고리처럼 표시하여 모든 날짜 보이게
                tickangle=-45,
                title="날짜"
            )
            fig_daily.update_yaxes(title="생산 수량 (Shot)")
            fig_daily.update_layout(
                height=400,
                legend=dict(orientation="h", y=1.1, title=None)
            )
            
            st.plotly_chart(fig_daily, use_container_width=True)
        else:
            st.info("표시할 일별 데이터가 없습니다.")

    else:
        st.info(" 사이드바에서 월(Month)과 설비를 선택하고 데이터 로드를 기다려주세요.")

if __name__ == "__main__":
    main()





