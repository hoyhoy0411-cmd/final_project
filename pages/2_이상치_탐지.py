import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import gdown
import requests
from io import BytesIO

BASE_DIR = os.getcwd()

# --- [페이지 설정 (가장 먼저 실행)] ---
st.set_page_config(page_title="사출 공정 통합 모니터링 시스템", layout="wide")

# ==========================================
# [설정 및 ID 매핑]
# ==========================================

# 1. 모델 파일 ID 매핑 (설비별 분리)
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

# 2. 데이터 파일 ID 매핑 (월별 분리)
# 전처리 데이터만 메인 분석에 사용된다고 가정하고 구성했습니다.
# 샷별 데이터도 필요 시 로드할 수 있도록 매핑해두었습니다.
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

def download_file_from_google_drive(file_id, output_path):
    """로컬에 파일이 없을 경우에만 구글 드라이브에서 다운로드"""
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(output_path):
        try:
            # st.spinner는 함수 밖(호출부)에서 처리하는 것이 UI상 깔끔함
            gdown.download(url, output_path, quiet=True)
        except Exception as e:
            st.error(f"다운로드 실패: {e}")
    return output_path


# 함수 이름 통일 (load_data_by_month로 사용하거나 load_local_data로 통일)
@st.cache_data
def load_local_data(month_key):
    """선택한 월의 데이터를 드라이브에서 확인 후 로드"""
    if month_key not in DATA_IDS:
        return pd.DataFrame()
    
    file_id = DATA_IDS[month_key]['prep']
    file_name = f"prep_data_{month_key}.csv"
    file_path = os.path.join(BASE_DIR, file_name)

    # 1. 파일이 없으면 다운로드 실행
    if not os.path.exists(file_path):
        with st.spinner(f'{month_key} 데이터 다운로드 중...'):
            download_file_from_google_drive(file_id, file_path)

    # 2. 파일 로드 (인코딩 처리)
    encodings = ['utf-8-sig', 'cp949', 'latin-1']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc, engine='python', on_bad_lines='skip')
            break
        except:
            continue
    
    if df is not None:
        # 데이터 전처리: float32 변환 및 날짜 변환
        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')
        
        if 'Timestamp_사출' in df.columns:
            df['Timestamp_사출'] = pd.to_datetime(df['Timestamp_사출'], errors='coerce')
            df = df.dropna(subset=['Timestamp_사출'])
            
    return df

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

# 1. 모델 로드 함수 (개별 로드)
@st.cache_resource
def load_single_model(mach_no):
    """선택한 설비의 모델 로드"""
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
    
# 2. 데이터 로드 함수 (월별 로드 및 메모리 최적화)
@st.cache_data
def load_local_data(month_key):
    """선택한 월의 Parquet 데이터를 로드"""
    if month_key not in DATA_IDS:
        return pd.DataFrame()
    
    file_id = DATA_IDS[month_key]['prep']
    # 확장자를 .parquet으로 변경
    file_name = f"prep_data_{month_key}.parquet"
    file_path = os.path.join(BASE_DIR, file_name)

    # 1. 파일 다운로드
    if not os.path.exists(file_path):
        with st.spinner(f'{month_key} 데이터 다운로드 중...'):
            download_file_from_google_drive(file_id, file_path)

    # 2. Parquet 로드 (인코딩 문제 없음)
    try:
        df = pd.read_parquet(file_path)
        
        # 컬럼명 공백 제거
        df.columns = [col.strip() for col in df.columns]
        
        # 메모리 최적화
        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')
        
        # 날짜 변환
        target_col = 'Timestamp_사출'
        if target_col in df.columns:
            df[target_col] = pd.to_datetime(df[target_col], errors='coerce')
            df = df.dropna(subset=[target_col])
        else:
            # Parquet임에도 컬럼이 없다면 실제 컬럼명 확인 출력
            st.error(f"'{target_col}' 컬럼을 찾을 수 없습니다. 실제 컬럼: {df.columns.tolist()}")
            
        return df
    except Exception as e:
        st.error(f"데이터 로드 실패 ({month_key}): {e}")
        # 만약 로드 실패 시 깨진 파일일 수 있으므로 삭제 제안
        if os.path.exists(file_path):
            os.remove(file_path)
        return pd.DataFrame()

@st.cache_resource
def load_models():
    """로컬 모델 파일 로드"""
    model_path = os.path.join(BASE_DIR, MODEL_FILE_NAME)
    if not os.path.exists(model_path):
        st.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return None
    return joblib.load(model_path)

# ==========================================
# [Tab 1: 상세 분석용 함수]
# ==========================================

def get_status_color_detail(prob, threshold, mach_no):
    """Tab 1: 상세 분석용 신호등 로직"""
    # threshold가 0인 경우 방어 로직
    if threshold <= 0: threshold = 0.5
        
    ratio = prob / threshold
    if ratio < 0.7:
        return "🟢 정상", "#28a745", "공정이 매우 안정적입니다."
    elif ratio < 1.0:
        return "🟡 주의", "#ffc107", f"[{mach_no}] 리듬 이탈 전조가 감지되었습니다. SOP를 확인하세요."
    else:
        return "🔴 위험", "#dc3545", f"🚨 [{mach_no}] 불량 발생 위험이 매우 높습니다! 즉시 점검 바랍니다."

def get_sop_guide(mach_no):
    """Tab 1: SOP 가이드 로직"""
    if mach_no in ['D01', 'F11', 'G10']:
        return "Group A (저점 안정형)", "✅ 금형 온도 하한 사수 및 압력 기저치 관리"
    elif mach_no in ['F09', 'G03', 'G06']:
        return "Group B (미세 흔들림형)", "✅ 잔차(Resid) 및 변동계수(CV) 기반 리듬 유지"
    elif mach_no in ['D10', 'G02', 'G05']:
        return "Group C (종점 제어형)", "✅ 보압 완료 위치 종점 도달 및 쿠션량 정밀 제어"
    else:
        return "Group D (절대 수치형)", "✅ 금형 온도 평형 유지 및 절대 임계치 준수"

# ==========================================
# [Tab 2: 예지보전(RUL)용 함수]
# ==========================================

def get_status_color_rul(status):
    colors = {
        "🔴 즉시점검": "#FF4B4B",
        "🟡 주의": "#FFA500",
        "🟢 정상": "#28A745",
        "🟢 매우안정": "#007BFF"
    }
    return colors.get(status, "#000000")

def process_machine_data(res_df, cycle_time_sec=12, analysis_window=500):
    """Tab 2: RUL 계산 로직"""
    rul_results = []
    res_df = res_df.sort_values(['MACHNO', 'Timestamp_사출'])
    machines = sorted(res_df['MACHNO'].unique())
    CAVITY = 4
    
    for m in machines:
        m_data = res_df[res_df['MACHNO'] == m].copy().reset_index(drop=True)
        # 건강지수: 확률이 높을수록(위험할수록) 점수가 낮아지도록 변환
        m_data['health_score'] = (1 - m_data['refined_probs']) * 100
        
        recent_df = m_data.tail(analysis_window).copy()
        if len(recent_df) < 50: continue

        current_health = recent_df['health_score'].tail(50).mean()
        limit_score = (1 - recent_df['best_th'].iloc[-1]) * 100
        
        X = np.array(range(len(recent_df))).reshape(-1, 1)
        y = recent_df['health_score'].values
        model = LinearRegression().fit(X, y)
        slope = model.coef_[0]
        
        if slope < -0.0001:
            rem_cycles = int((limit_score - current_health) / slope)
            rem_cycles = max(0, rem_cycles)
            rem_hours = (rem_cycles * cycle_time_sec) / 3600
            
            if rem_cycles < 1500: status = "🔴 즉시점검"
            elif rem_cycles < 4500: status = "🟡 주의"
            else: status = "🟢 정상"
        else:
            safety_margin = max(0, current_health - limit_score)
            rem_cycles = int(safety_margin * 1000)
            rem_hours = (rem_cycles * cycle_time_sec) / 3600
            status = "🟢 매우안정"
            
        rul_results.append({
            '설비명': m,
            '현재건강도': round(current_health, 1),
            '임계치': round(limit_score, 1),
            '상태': status,
            '남은시간': round(rem_hours, 1),
            '남은사이클': rem_cycles,
            '예상수량': rem_cycles * CAVITY
        })
    return pd.DataFrame(rul_results)

def analyze_mtbfa_from_optimizer(res_df, target_multiplier=3):
    """Tab 3: MTBFA 분석 로직"""
    res_df = res_df.copy()
    res_df['Timestamp_사출'] = pd.to_datetime(res_df['Timestamp_사출'])
    
    if '합불' not in res_df.columns:
        res_df['합불'] = 0 
        
    res_df['is_fp'] = ((res_df['합불'] == 0) & (res_df['y_pred'] == 1)).astype(int)
    res_df['is_fault'] = res_df['y_pred']
    
    # 예시 가동 시간 정보 (실제 프로젝트에서는 데이터에서 추출하거나 외부 config 사용 권장)
    equipment_info = {
        'D01': {'start': '2025-09-08 01:45:57', 'end': '2025-10-01 08:27:50'},
        'D10': {'start': '2025-09-01 22:15:49', 'end': '2025-10-31 23:22:39'},
        'F09': {'start': '2025-10-11 12:54:44', 'end': '2025-10-24 08:42:56'},
        'F11': {'start': '2025-09-01 08:44:50', 'end': '2025-10-31 22:51:53'},
        'G01': {'start': '2025-09-08 14:59:50', 'end': '2025-11-01 07:56:08'},
        'G02': {'start': '2025-09-01 08:48:34', 'end': '2025-10-31 20:05:56'},
        'G03': {'start': '2025-09-08 18:42:18', 'end': '2025-10-17 02:33:17'},
        'G05': {'start': '2025-09-05 17:35:39', 'end': '2025-10-29 09:48:22'},
        'G06': {'start': '2025-09-08 14:52:04', 'end': '2025-11-01 07:56:51'},
        'G10': {'start': '2025-09-03 15:49:47', 'end': '2025-10-31 20:23:35'}
    }
    
    report_list = []
    for eq_id in sorted(res_df['MACHNO'].unique()):
        # 해당 설비 정보가 없으면 전체 기간으로 추정하거나 생략
        start_dt = res_df[res_df['MACHNO']==eq_id]['Timestamp_사출'].min()
        end_dt = res_df[res_df['MACHNO']==eq_id]['Timestamp_사출'].max()
        
        # config에 있으면 override
        if eq_id in equipment_info:
            try:
                start_dt = datetime.strptime(equipment_info[eq_id]['start'], '%Y-%m-%d %H:%M:%S')
                end_dt = datetime.strptime(equipment_info[eq_id]['end'], '%Y-%m-%d %H:%M:%S')
            except:
                pass
                
        total_hours = (end_dt - start_dt).total_seconds() / 3600
        if total_hours <= 0: total_hours = 0.1 # div by zero 방지
        
        m_df = res_df[res_df['MACHNO'] == eq_id].copy().sort_values('Timestamp_사출')
        
        fault_count = m_df['is_fault'].sum()
        mtbf_hr = total_hours / fault_count if fault_count > 0 else total_hours
        
        m_df['consecutive_fp'] = m_df['is_fp'].rolling(window=3).sum()
        fa_count = len(m_df[m_df['consecutive_fp'] == 3])
        
        mtbfa_hr = total_hours / fa_count if fa_count > 0 else total_hours
        target_mtbfa = mtbf_hr * target_multiplier
        reliability_score = min(100, (mtbfa_hr / target_mtbfa) * 100) if target_mtbfa > 0 else 100
        
        report_list.append({
            '설비ID': eq_id,
            '가동시간(hr)': round(total_hours, 1),
            '불량판정(건)': fault_count,
            'MTBF(분)': round(mtbf_hr * 60, 2),
            '현재MTBFA(hr)': round(mtbfa_hr, 2),
            '목표MTBFA(분)': round(target_mtbfa * 60, 2),
            '신뢰도점수': round(reliability_score, 1)
        })
    return pd.DataFrame(report_list)

# ==========================================
# [메인 실행 함수]
# ==========================================
def main():
    # 1. 사이드바: 데이터 로드 및 설비 선택
    with st.sidebar:
        st.header("⚙️ 분석 설정")
        month_options = list(DATA_IDS.keys()) 
        selected_month = st.selectbox("1. 분석 월 선택", month_options)
        
        # 데이터 로드
        full_df = load_local_data(selected_month)        

        data_ready = False
        if not full_df.empty:
            available_machines = sorted(full_df['MACHNO'].unique())
            target_mach = st.selectbox("2. 분석 설비 선택", available_machines)
            
            model_info = load_single_model(target_mach)
            if model_info:
                st.success(f"✅ {target_mach} 모델 로드 완료")
                data_ready = True
            else:
                st.warning(f"⚠️ {target_mach} 모델 없음")
        else:
            st.error("데이터를 불러올 수 없습니다.")

    # --- [메인 화면: 탭 구성] ---
    st.title("🏭 사출 공정 스마트 통합 대시보드")
    tab1, tab2, tab3 = st.tabs(["🔍 설비별 상세 분석", "⏳ 전체 설비 예지보전 (RUL)","📊 AI 신뢰도 리포트"])

    # ----------------------------------------------------------------
    # [TAB 1] 설비별 사출 품질 및 이상치 분석
    # ----------------------------------------------------------------
    with tab1:
        if data_ready and not full_df.empty:
            st.subheader(f"📊 {target_mach} 설비 상세 분석 ({selected_month})")
            
            # 1. 데이터 필터링
            m_df = full_df[full_df['MACHNO'] == target_mach].copy().reset_index(drop=True)
            
            # 2. 분석 수행
            info = model_info # load_single_model에서 로드한 개별 모델 정보
            
            # --- 전처리 및 Feature Engineering ---
            # 모델 pkl 안에 'features', 'scaler', 'model' 등이 딕셔너리로 들어있다고 가정
            trained_features = info['features']
            X = pd.DataFrame(index=m_df.index)
            
            # 필요한 컬럼만 추출 (없으면 0 채움)
            for col in trained_features:
                X[col] = m_df[col] if col in m_df.columns else 0
            X = X.fillna(0)
            
            # 잔차(Residual) 생성
            X_res = X - X.rolling(window=10, min_periods=1).mean()
            X_res.columns = [f"{c}_resid" for c in X_res.columns]
            X_combined = pd.concat([X, X_res], axis=1)

            # Polynomial Features
            if target_mach == 'G06':
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                top_cols = [c for c in X.columns if any(s in c for s in ['금형온도', '최소쿠션', '충진시간', '압력'])][:6]
                X_poly = poly.fit_transform(X_combined[top_cols])
                X_final = np.hstack([X_combined.values, X_poly])
            elif info.get('group') == '난조군': # group 키가 없을 수도 있으므로 get 사용
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                top_cols = [c for c in X.columns if any(s in c for s in ['금형온도', '최소쿠션', '충진시간'])][:5]
                X_poly = poly.fit_transform(X_combined[top_cols])
                X_final = np.hstack([X_combined.values, X_poly])
            else:
                X_final = X_combined.values

            # --- AI 모델 예측 ---
            # scaler가 float32를 받으면 경고가 뜰 수 있으나 작동은 함
            X_scaled = info['scaler'].transform(X_final)
            
            # XGBoost CPU 설정 (호환성)
            try:
                info['model'].set_params(device="cpu")
            except:
                pass # sklearn 모델 등일 경우 set_params 에러 무시

            probs = info['model'].predict_proba(X_scaled)[:, 1]
            
            # Anomaly Detection (LOF)
            d_scores = info['lof'].predict(X_scaled) 
            refined_probs = probs.copy()
            refined_probs[d_scores == 1] *= info['lof_penalty']
            
            # 결과 병합
            m_df['predict_prob'] = refined_probs
            m_df['y_pred'] = (refined_probs >= info['threshold']).astype(int)
            m_df['is_outlier'] = d_scores
            
            # 시각화를 위해 잔차 데이터도 m_df에 일부 붙임 (인덱스 리셋 필요)
            X_res_reset = X_res.reset_index(drop=True)
            m_df = pd.concat([m_df, X_res_reset], axis=1)
            
            latest_shot = m_df.iloc[-1]

            # --- 대시보드 UI ---
            col1, col2, col3 = st.columns(3)
            col1.metric("설비 그룹", info.get('group', 'Unknown'))
            col2.metric("전체 사이클", f"{len(m_df)} 건")
            col3.metric("예측 불량 건수", f"{m_df['y_pred'].sum()} 건", delta_color="inverse")

            st.divider()
            
            col_sub1, col_sub2 = st.columns([1, 2])

            # [좌측] 상태 카드
            with col_sub1:
                st.subheader(f"📍 {target_mach} 상태 요약")
                current_prob = latest_shot['predict_prob']
                target_threshold = info['threshold']
                
                status_text, color, msg = get_status_color_detail(current_prob, target_threshold, target_mach)
                group_name, sop_text = get_sop_guide(target_mach)
                
                st.markdown(f"""
                    <div style="background-color:{color}; padding:25px; border-radius:15px; text-align:center; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
                        <h1 style="color:white; margin:0; font-size: 2.5rem;">{status_text}</h1>
                        <p style="color:white; margin:5px 0 0 0;">위험도: {current_prob:.2%}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.info(f"**소속 그룹:** {group_name}\n\n**핵심 SOP:** {sop_text}")
                st.warning(f"**AI 진단:** {msg}")

            # [우측] 변수 기여도
            with col_sub2:
                st.subheader("🔍 실시간 변수 기여도 분석 (최신 샷)")
                res_cols = [c for c in X_res.columns]
                # 최근 데이터 중 잔차 절대값이 큰 상위 6개
                latest_res = latest_shot[res_cols]
                top_res = latest_res.abs().sort_values(ascending=True).tail(6)
                
                fig_shap = go.Figure(go.Bar(
                    x=latest_res[top_res.index].values,
                    y=[c.replace('_resid', '') for c in top_res.index],
                    orientation='h',
                    marker_color=['#FF4B4B' if x > 0 else '#3366CC' for x in latest_res[top_res.index].values]
                ))
                fig_shap.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_shap, use_container_width=True)

            st.divider()
            st.subheader("📈 품질 리듬 모니터링")
            
            # 날짜 슬라이더
            all_dates = sorted(m_df['Timestamp_사출'].dt.date.unique())
            
            if len(all_dates) > 0:
                start_date, end_date = all_dates[0], all_dates[-1]
                display_df = m_df
                
                if len(all_dates) > 1:
                    selected_range = st.select_slider(
                        "날짜 범위 선택",
                        options=all_dates,
                        value=(all_dates[0], all_dates[-1]),
                        format_func=lambda x: x.strftime("%y-%m-%d"),
                        key='t1_slider'
                    )
                    start_date, end_date = selected_range
                    display_df = m_df[(m_df['Timestamp_사출'].dt.date >= start_date) & 
                                      (m_df['Timestamp_사출'].dt.date <= end_date)]

                # 비가동 구간 시각화 헬퍼
                full_range = pd.date_range(start=start_date, end=end_date, freq='D').date
                data_exist_dates = set(display_df['Timestamp_사출'].dt.date.unique())
                missing_dates = sorted([d for d in full_range if d not in data_exist_dates])

                def add_off_shading(fig):
                    for m_date in missing_dates:
                        fig.add_vrect(
                            x0=pd.to_datetime(m_date) - pd.Timedelta(hours=12),
                            x1=pd.to_datetime(m_date) + pd.Timedelta(hours=12),
                            fillcolor="Gray", opacity=0.15, layer="below", line_width=0
                        )
                    return fig

                # [차트 1] 확률 추이
                fig_prob = px.line(display_df, x='Timestamp_사출', y='predict_prob', color_discrete_sequence=['#FF4B4B'])
                fig_prob.add_hline(y=info['threshold'], line_dash="dash", line_color="black", annotation_text="임계치")
                fig_prob = add_off_shading(fig_prob)
                fig_prob.update_xaxes(type='date', tickformat="%y-%m-%d", tickangle=-45)
                fig_prob.update_layout(height=300, legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_prob, use_container_width=True)

                # [차트 2] 핵심 변수 리듬
                if len(res_cols) > 0:
                    top_feature_resid = latest_shot[res_cols].abs().astype(float).idxmax()
                    top_feature_name = top_feature_resid.replace('_resid', '')
                    
                    st.write(f"**핵심 변수 리듬: {top_feature_name}**")
                    fig_line = px.line(display_df, x='Timestamp_사출', y=top_feature_resid, color_discrete_sequence=['#28a745'])
                    fig_line.add_hline(y=0, line_dash="solid", line_color="black")
                    fig_line = add_off_shading(fig_line)
                    fig_line.update_xaxes(type='date', tickformat="%y-%m-%d", tickangle=-45)
                    fig_line.update_layout(height=300, legend=dict(orientation="h", y=1.1))
                    st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("표시할 데이터가 없습니다.")

        else:
            st.info("👈 사이드바에서 월(Month)과 설비를 선택하고 데이터 로드를 기다려주세요.")

    # ----------------------------------------------------------------
    # [TAB 2] 전체 설비 예지보전 (RUL)
    # ----------------------------------------------------------------
    with tab2:
        st.subheader("📊 설비 수명 예측 및 예지보전")
        st.markdown("최적화 모델에서 추출한 **분석용 CSV**를 업로드하여 리포트를 생성합니다.")

        uploaded_file = st.file_uploader("추출된 CSV 파일을 선택하세요 (injection_machine_monitoring_data.csv)", type="csv", key='t2_uploader')

        if uploaded_file is not None:
            final_results_df = pd.read_csv(uploaded_file)
            
            # float 최적화 (Tab 2에서도 적용)
            float_cols_rul = final_results_df.select_dtypes(include=['float64']).columns
            if len(float_cols_rul) > 0:
                final_results_df[float_cols_rul] = final_results_df[float_cols_rul].astype('float32')

            required_cols = ['Timestamp_사출', 'MACHNO', 'refined_probs', 'best_th']
            if all(col in final_results_df.columns for col in required_cols):
                with st.spinner('설비별 RUL 분석 중...'):
                    report_df = process_machine_data(final_results_df)
                
                if report_df.empty:
                    st.warning("분석할 설비 데이터가 충분하지 않습니다.")
                else:
                    # 요약 Metric
                    st.write("#### ✅ 설비 가동 요약")
                    cols = st.columns(len(report_df))
                    for i, row in report_df.iterrows():
                        if i < len(cols):
                            with cols[i]:
                                st.metric(label=str(row['설비명']), 
                                          value=f"{row['현재건강도']}%", 
                                          delta=row['상태'],
                                          delta_color="normal" if "정상" in row['상태'] or "안정" in row['상태'] else "inverse")
                                st.progress(min(max(row['현재건강도']/100, 0.0), 1.0))

                    st.divider()

                    # 상세 리포트
                    st.write("#### 📋 설비별 상세 분석 리포트")
                    status_order = {"🔴 즉시점검": 0, "🟡 주의": 1, "🟢 정상": 2, "🟢 매우안정": 3}
                    report_df['sort'] = report_df['상태'].map(status_order)
                    report_df = report_df.sort_values('sort').drop('sort', axis=1)

                    st.dataframe(
                        report_df.style.map(
                            lambda x: f"color: {get_status_color_rul(x)}; font-weight: bold;" if x in status_order else "",
                            subset=['상태']
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                    # 시각화 차트
                    st.write("#### ⏳ 설비별 남은 가동 가능 시간 (Hours)")
                    fig = px.bar(report_df, x='설비명', y='남은시간', color='상태',
                                 color_discrete_map={
                                     "🔴 즉시점검": "#FF4B4B", "🟡 주의": "#FFA500", 
                                     "🟢 정상": "#28A745", "🟢 매우안정": "#007BFF"
                                 },
                                 text='남은시간',
                                 labels={'남은시간': '잔여 수명 (시간)'})
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"필수 컬럼 부족: {required_cols}")
        else:
            st.info("👆 위 영역에서 CSV 파일을 업로드해주세요.")

    # ----------------------------------------------------------------
    # [TAB 3] AI 신뢰도 리포트
    # ----------------------------------------------------------------
    with tab3:
        st.subheader("📊 AI 모델 기반 현장 신뢰도(MTBFA) 리포트")
        file_t3 = st.file_uploader("신뢰도 분석용 CSV 업로드", type="csv", key='uploader_tab3')

        if file_t3:
            try:
                df_t3 = pd.read_csv(file_t3)
                
                # float 최적화
                float_cols_t3 = df_t3.select_dtypes(include=['float64']).columns
                if len(float_cols_t3) > 0:
                    df_t3[float_cols_t3] = df_t3[float_cols_t3].astype('float32')

                if 'y_pred' not in df_t3.columns and 'refined_probs' in df_t3.columns:
                    th = df_t3['best_th'].iloc[0] if 'best_th' in df_t3.columns else 0.5
                    df_t3['y_pred'] = (df_t3['refined_probs'] >= th).astype(int)

                df_report = analyze_mtbfa_from_optimizer(df_t3)
                
                if not df_report.empty:
                    avg_score = df_report['신뢰도점수'].mean()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("평균 신뢰도 점수", f"{avg_score:.1f}점")
                    c2.metric("분석 설비 수", f"{len(df_report)}대")
                    c3.metric("최고 신뢰 설비", f"{df_report.loc[df_report['신뢰도점수'].idxmax(), '설비ID']}")
                    
                    st.divider()
                    st.dataframe(
                        df_report.style.format({
                            '신뢰도점수': '{:.1f}%', 
                            '현재MTBFA(hr)': '{:.2f} hr',
                            'MTBF(분)': '{:.1f} min'
                        }), 
                        use_container_width=True
                    )
                    
                    with st.expander("ℹ️ 신뢰도 지표(MTBFA) 계산 기준"):
                        st.latex(r"Score = \min\left(100, \frac{MTBFA_{current}}{MTBF \times 3} \times 100\right)")
                        st.write("모델이 예측한 불량 주기(MTBF) 대비 실제 오경보 없이 유지된 시간(MTBFA)을 비교합니다.")
                else:
                    st.warning("분석할 설비 데이터가 없습니다.")
            except Exception as e:
                st.error(f"데이터 처리 중 오류 발생: {e}")
        else:
            st.info("💡 AI 모델의 현장 적합성을 평가하려면 결과 데이터를 업로드하세요.")

if __name__ == "__main__":
    main()

