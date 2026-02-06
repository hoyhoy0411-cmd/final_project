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
import shap

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

# 3-1. 그룹별 특성 설명 (이미지 기반 추가)
GROUP_DESCRIPTIONS = {
    '고성능군': "불량 패턴이 명확하고 정상 데이터와 구분이 쉬운 설비들입니다.",
    '안정군': "공정이 비교적 일정하게 유지되나 간헐적인 튀는 값이 존재하는 설비들입니다.",
    '난조군': "데이터 노이즈가 심하고 정상/불량 경계가 모호한 고난도 설비들입니다."
}

# 4. 설비별 대응 전략 정의
STRATEGIES = {
    "D01": {
        "priority": "압력 과상승 억제 및 기저치 유지",
        "actions": [
            "**안정적인 하한 유지**: 충진 압력/금형 온도가 하한 구간 내 형성되도록 관리",
            "**과도한 상승 억제**: 압력이 기저치를 초과하여 튀어 오르는 현상 방어",
            "**잔차 모니터링**: 시계열적인 리듬 변화 포착 (사출시간_lag1_resid 등)"
        ]
    },
    "D10": {
        "priority": "공정 리듬 관리 (일관성)",
        "actions": [
            "**사출 흐름 개선**: 사출 시간을 짧고 일정하게 유지",
            "**보압 깊이 확보**: 보압 완료 위치가 목표치까지 깊게 도달하도록 제어",
            "**정밀 모니터링**: 충진피크 잔차 및 Cycle Time 리듬 실시간 감시"
        ]
    },
    "F09": {
        "priority": "반복 정밀도 및 동적 밸런스 유지",
        "actions": [
            "**동적 평형 유지**: 적정 변동 범위 내에서 리듬을 타도록 온도 관리",
            "**대칭적 이탈 감시**: 잔차 지표의 좌우 대칭형 이탈 확인",
            "**시계열 패턴 확보**: 직전 작업 대비 지연 여부를 통해 규칙성 확보"
        ]
    },
    "F11": {
        "priority": "변동성 최소화 (반복 정밀도)",
        "actions": [
            "**금형 온도 저점 관리**: 냉각 시스템 점검으로 안정적 저점 유지",
            "**중앙 밀집 유도**: 충진 시간/압력을 모델 신뢰 구간(-1~0)에 밀집",
            "**변동계수(CV) 제어**: 안정 구간 이탈 실시간 체크"
        ]
    },
    "G02": {
        "priority": "공정 난조 및 리듬 이탈 억제",
        "actions": [
            "**보압 종점 정밀 도달**: 보압 완료 위치가 정상 구간에 충분히 도달하도록 유도",
            "**압력 중앙 집중**: 충진 피크 압력의 산포(Dispersion) 관리",
            "**사이클 타임 준수**: 급격한 리듬 변화 방지"
        ]
    },
    "G03": {
        "priority": "사출 반복 정밀도 확보",
        "actions": [
            "**압력 변동성 제어**: 충진피크압 변동계수를 낮게 유지",
            "**사출 지연 방지**: 공정 부하 억제로 시간 지연 예방",
            "**잔차 관리**: 급격한 공정 리듬 변화 선제적 방어"
        ]
    },
    "G05": {
        "priority": "임계 영역 이탈 방지",
        "actions": [
            "**금형 고온 안정화**: 온도를 높고 안정적인 상태로 유지",
            "**보압 종점 정밀성**: 0을 넘어 우측으로 이탈하지 않도록 깊은 위치 확보",
            "**미세 변화 포착**: 잔차 로직으로 불량 구간 진입 전 징후 감지"
        ]
    },
    "G06": {
        "priority": "공정 리듬의 밀집도 유지",
        "actions": [
            "**사이클 타임 정시성**: 0 부근 밀집 구간을 벗어나지 않도록 관리",
            "**압력 하한선 관리**: 압력의 급격한 저하 방어",
            "**경계 관리**: 열적 평형 상태 보조 감시"
        ]
    },
    "G10": {
        "priority": "미세 흔들림 없는 공정 유지 (Low-CV)",
        "actions": [
            "**극단적 안정성 확보**: 변동계수(CV)가 커지지 않도록 관리",
            "**동적 이력 관리**: 직전 작업(lag1) 모니터링으로 규칙성 단절 방지",
            "**비율 및 차이 유지**: 온도 차이(diff) 등 파생 변수의 밀집도 유지"
        ]
    },
    "G01": {
        "priority": "열적 평형 유지 및 절대 임계치(Hard-Limit) 정밀 제어",
        "actions": [
            "**지속적 열적 평형**: 금형 온도가 낮아지지 않도록 충분한 온도를 확보하여 냉각 과다 방지",
            "**쿠션량 정밀 구간 관리**: 최소 쿠션이 과소/과다 영역으로 진입하지 않도록 좁은 타겟 윈도우(Window) 내 제어",
            "**절대 임계치(Limit) 중심**: 잔차(리듬) 변화보다는 설정된 절대 상/하한값 이탈 여부에 집중 감시"
        ]
    }
}

# ==========================================
# [공통 데이터 및 함수 정의]
# ==========================================

def download_file_from_google_drive(file_id, output_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(output_path):
        with st.spinner(f'파일 다운로드 중: {output_path}...'):
            gdown.download(url, output_path, quiet=True)
    return output_path

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
        st.header(" 분석 설정")
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

        st.markdown("---")
        st.error(
            """
            **주의사항**
            
            **반드시 데이터가 모두 로드된 후에 분석 월과 분석 설비를 선택해주세요.** 스트림릿의 램 용량 부족으로 인해 서버에 과부하가 걸릴 수 있습니다.
            """
        )

    # --- [메인 화면] ---
    st.title(" 사출 공정 스마트 상세 분석 대시보드 (이상치 탐지)")

    if data_ready and not full_df.empty:

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
        
        # ==============================================================================
        # [Tab 구성] 
        # Tab 1: 월간 분석 결과 (지표, 카드, SHAP)
        # Tab 2: 대응 전략 및 일별 현황 (Strategy, Daily Graph)
        # ==============================================================================
        
        tab1, tab2 = st.tabs([" 월간 분석 결과 (SHAP)", "대응 전략 및 일별 현황"])

        # ------------------------------------------------------------------------------
        # [Tab 1] 월간 분석 결과
        # ------------------------------------------------------------------------------
        with tab1:
            st.subheader(f" {target_mach} 설비 상세 분석 ({selected_month})")
            st.info(f" **{selected_month} {target_mach} 설비**의 전체 생산 Shot 중에서 공정 과정에 이상이 생겼다고 판단되는 불량입니다.\n\n(설비의 특정 센서값이 이전 패턴과 달라졌음을 의미합니다.)")

            col1, col2, col3, col4 = st.columns(4)
            
            # 그룹 정보 가져오기
            current_group = info.get('group', 'Unknown')
            
            col1.metric("설비 그룹", current_group)
            col2.metric("전체 사이클", f"{len(m_df):,} Shot")
            col3.metric("이상 판단 수", f"{m_df['y_pred'].sum():,} Shot")
            col4.metric("평균 이상 발생률", f"{(m_df['y_pred'].sum() / len(m_df) * 100):.2f}%", delta_color="inverse")
            
            # [추가됨] 그룹별 상세 설명 표시
            if current_group in GROUP_DESCRIPTIONS:
                st.caption(f" **{current_group} 특성**: {GROUP_DESCRIPTIONS[current_group]}")
            
            st.divider()

            col_sub1, col_sub2 = st.columns([1, 2])

            # 좌측: 월간 종합 상태 요약
            with col_sub1:
                st.subheader(f" {selected_month} 종합 판정 결과")

                total_ng = int(m_df['y_pred'].sum())
                total_ok = int(len(m_df) - total_ng)
                
                ng_rate = total_ng / len(m_df)
                card_color = "#dc3545" if ng_rate > 0.2 else "#28a745"
                
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

            # 우측: SHAP 기반 주요 변동 요인 분석
            with col_sub2:
                st.subheader(f" {selected_month} AI 모델 주요 판단 요인 (SHAP)")
                st.info(" 막대의 **길이**는 영향력의 크기, **색상**은 방향을 의미합니다.\n\n🔴 **빨강**: 공정 이상 확률을 높이는 요인 (위험)\n🔵 **파랑**: 공정 이상 확률을 낮추는 요인 (안정)")

                # Feature Name 리스트 생성
                base_cols = list(X.columns)
                resid_cols = list(X_res.columns)
                feature_names = base_cols + resid_cols
                
                if target_mach == 'G06' or info.get('group') == '난조군':
                    try:
                        poly_names = poly.get_feature_names_out(top_cols)
                        feature_names.extend(poly_names)
                    except:
                        current_len = len(feature_names)
                        remain_len = X_final.shape[1] - current_len
                        feature_names.extend([f"Poly_Feature_{i}" for i in range(remain_len)])
                
                # SHAP 값 계산
                with st.spinner("SHAP 중요도 및 방향성 분석 중..."):
                    try:
                        sample_size = 100
                        if X_scaled.shape[0] > sample_size:
                            indices = np.random.choice(X_scaled.shape[0], sample_size, replace=False)
                            X_sample = X_scaled[indices]
                        else:
                            X_sample = X_scaled

                        model = info['model']
                        shap_values = None

                        try:
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(X_sample, check_additivity=False)
                        except:
                            explainer = shap.KernelExplainer(model.predict_proba, X_sample)
                            shap_values = explainer.shap_values(X_sample)

                        if isinstance(shap_values, list):
                            vals = shap_values[1]
                        else:
                            vals = shap_values
                            if len(vals.shape) > 2:
                                 vals = vals[:, :, 1]

                        if vals.shape[1] != len(feature_names):
                            feature_names = [f"Var_{i}" for i in range(vals.shape[1])]

                        abs_importance = np.mean(np.abs(vals), axis=0)
                        mean_direction = np.mean(vals, axis=0)
                        
                        shap_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': abs_importance,
                            'Direction': mean_direction
                        })

                        top_5_shap = shap_df.sort_values(by='Importance', ascending=False).head(5)
                        
                        colors = ['#dc3545' if d > 0 else '#007bff' for d in top_5_shap['Direction']]

                        fig_shap = go.Figure(go.Bar(
                            x=top_5_shap['Importance'],
                            y=top_5_shap['Feature'],
                            orientation='h',
                            marker=dict(color=colors),
                            text=[f"{v:.4f}" for v in top_5_shap['Importance']],
                            textposition='auto'
                        ))

                        fig_shap.update_layout(
                            xaxis_title="평균 SHAP 중요도 (Mean |SHAP|)",
                            yaxis=dict(autorange="reversed"),
                            height=300,
                            margin=dict(l=10, r=10, t=30, b=10)
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)

                    except Exception as e:
                        st.error(f"SHAP 분석 오류: {e}")
                        st.markdown("⚠️ 모델 호환성 문제로 변수 중요도를 표시할 수 없습니다.")
        
        # ------------------------------------------------------------------------------
        # [Tab 2] 대응 전략 및 일별 현황 (최종 디자인 수정)
        # ------------------------------------------------------------------------------
        with tab2:
            st.subheader(f" {target_mach} 설비 맞춤형 대응 방안 및 품질 전략")
            
            # 설명 문구 추가
            st.info("SHAP 분석 결과에 따른 **공정의 안정된 관리를 위한 핵심 대응 방안**과 **품질 전략**입니다.")

            if target_mach in STRATEGIES:
                strat = STRATEGIES[target_mach]
                
                # 1. 최우선 과제 (요청 양식: 최우선 과제 : 이유)
                st.error(f"**최우선 과제 : {strat['priority']}**")
                
                # 2. 상세 대응 방안 (요청 양식: 1. Title : Description)
                st.subheader(" 상세 대응 방안")
                
                for i, action in enumerate(strat['actions'], 1):
                    # 리스트를 번호와 함께 간결하게 출력
                    st.markdown(f"{i}. {action}")
            else:
                st.info("해당 설비에 대한 구체적인 대응 전략 데이터가 없습니다.")

            st.divider()

            # 2. 일별 생산 및 품질 현황
            st.subheader(" 일별 생산 및 품질 현황")
            st.info(" 이상치 탐지 모델의 하루 **정상 판단 Shot** 개수와 **이상 판단 Shot** 개수를 비교해볼 수 있습니다.\n\n단순 제품 불량이 아닌, **설비의 이상(공정 변화)을 판단하는 지표**입니다.")

            daily_df = m_df.copy()
            daily_df['Date'] = daily_df['Timestamp_사출'].dt.date
            
            daily_stats = daily_df.groupby(['Date', 'y_pred']).size().reset_index(name='Count')
            daily_stats['Status'] = daily_stats['y_pred'].map({0: '정상(OK)', 1: '이상(NG)'})
            
            if not daily_stats.empty:
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
                    type='category',
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







