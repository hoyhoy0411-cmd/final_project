import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from io import BytesIO
import requests

# 1. 경로 설정
# --- 구글 드라이브 ID 매핑 ---
DATA_ID = "11nuPSVeJSFEk5E3wMkFn-lnTaY9p7Vlt"  # 대시보드_분류.csv
MODEL_IDS = {
    "D01": "1ZdObvW_egYKPIdJNRSDzn8ch-62HD9tB",
    "D10": "1MwiHRWIUe5dFSaVmQ8l19SLLdAKbp4pK",
    "F09": "15RtlRQ2wIhRneDHsp3aroBv4eMllv_Wp",
    "F11": "1CwJ6WJ7eUwbeBr2eY9O9G-BU5eyTEo_4",
    "G01": "1RXzfwlg3_Riz1r347GTuNIgoekVUPF8g",
    "G02": "1FSD2R7ACfLCvsMAPuJZJuF4fro7i-Wkv",
    "G03": "1lA95dCLsrMLV9VVmuQHpaxm8_zkwuGTQ",
    "G05": "1WDkJJaKLGJAE7LpQGnkFiymPZVkONUbb",
    "G06": "1u8zblexoK4jH-vZEao_mp0WHtGCC_RhK",
    "G10": "1tKJ-r1ejccx72yb9dDqwAxUYxJymMsUs"
}

def get_drive_url(file_id):
    return f'https://drive.google.com/uc?id={file_id}'

# 2. 로드 함수
@st.cache_resource
def load_machine_model(mach_no):
    if mach_no in MODEL_IDS:
        try:
            url = get_drive_url(MODEL_IDS[mach_no])
            response = requests.get(url)
            return joblib.load(BytesIO(response.content))
        except Exception as e:
            st.error(f"{mach_no} 모델 로드 실패: {e}")
    return None

@st.cache_data
def load_csv_data():
    try:
        url = get_drive_url(DATA_ID)
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return None

# --- 대시보드 메인 설정 ---
st.set_page_config(page_title="AI 품질 정밀 진단", layout="wide")
st.title(" 설비별 AI 품질 정밀 진단 & 시뮬레이터")

df = load_csv_data()

if df is not None:
    # 3. 모델이 있는 설비 리스트만 추출
    machine_list = sorted(list(MODEL_IDS.keys()))
    selected_machine = st.sidebar.selectbox(" 분석할 설비 선택", machine_list)

    if selected_machine:
        m_df = df[df['MACHNO'].astype(str) == str(selected_machine)].copy()
        time_col = 'Time_row' if 'Time_row' in m_df.columns else (m_df.columns[0] if not m_df.empty else None)

        if not m_df.empty:
            model_pack = load_machine_model(selected_machine)
            
            if model_pack:
                # --- 이후 시뮬레이션 및 그래프 로직은 기존과 동일 ---
                # (중략: 질문하신 기존 코드의 시뮬레이션 부분 그대로 사용)
                model = model_pack['model']
                features = model_pack['selected_features']
                fixed_threshold = float(model_pack.get('best_threshold', 0.5))

            # --- [1. 사이드바 시뮬레이션 설정] --- 
                # 슬라이더는 탭 외부(사이드바)에 있어야 어떤 탭에서도 시뮬레이션 값을 유지할 수 있습니다.
                st.sidebar.markdown("---")
                st.sidebar.subheader(" 시뮬레이션 설정")
    
                sim_threshold = st.sidebar.slider(" 판정 임계값 설정", 0.0, 1.0, fixed_threshold, 0.05)
    
                # 중요도 기반 상위 10개 변수 슬라이더 생성
                importances = model.feature_importances_
                fi_df = pd.DataFrame({'feature': features, 'importance': importances}).sort_values(by='importance', ascending=False)
                top_10_candidates = fi_df.head(10)['feature'].tolist()
    
                user_inputs = {}
                last_row = m_df.iloc[-1]
                for col in top_10_candidates:
                    v_min, v_max = float(m_df[col].min()), float(m_df[col].max())
                    if v_min == v_max:
                        user_inputs[col] = v_min
                        continue
                    user_inputs[col] = st.sidebar.slider(f"{col}", v_min, v_max, float(last_row[col]))

                # --- [공통 데이터 계산] ---
                X_all = m_df[features]
                probs = model.predict_proba(X_all)[:, 1]
                preds_fixed = (probs >= fixed_threshold).astype(int)
                m_df['AI_확률'] = probs
                m_df['AI_판정'] = preds_fixed 

                # 탭 생성
                tab_perf, tab_sim = st.tabs([" 로트별 예측 추이", " 모델성능 지표 및 예측 시뮬레이션"])

                # =================================================================
                # TAB 1: 전체 데이터 예측 및 모델 성능 진단
                # =================================================================
                with tab_perf:
                    st.subheader(" 전체 데이터 불량 위험도 추이")
                    
                    # --- [추가 1] 위험도 추이 설명 문구 ---
                    st.info(" 각 24shot 묶음에 대해 머신러닝 모델이 불량을 예측한 추이입니다. **판정 기준선(Threshold)** 을 초과할 경우 불량 묶음으로 예측합니다.")
                    # ------------------------------------

                    fig_line = px.line(
                        m_df, x=time_col, y='AI_확률',
                        title=f"{selected_machine} 시간대별 불량 예측 확률",
                        labels={'AI_확률': '불량 확률'}
                    )
                    fig_line.add_hline(y=fixed_threshold, line_dash="solid", line_color="red", 
                                       annotation_text=f"판정 기준선 ({fixed_threshold:.2f})")
                    st.plotly_chart(fig_line, use_container_width=True)

                    st.markdown("---")
                    
                    # (B-1) 실제 vs 예측 타임라인
                    st.subheader(" 실제 vs 예측 불량 발생 타임라인")
                    
                    # --- [추가 2] 타임라인 설명 문구 ---
                    st.info("24shot 묶음에서 발생한 **예측 불량**과 **실제 불량**을 시간순으로 시각적으로 비교하여 확인할 수 있습니다.")
                    # ----------------------------------

                    timeline_data = []
                    for idx, row in m_df.iterrows():
                        if row['NG_판정'] == 1:
                            timeline_data.append({'시간': row[time_col], '구분': '실제 불량 (Actual)', '상태': 'NG'})
                        if row['AI_판정'] == 1:
                            timeline_data.append({'시간': row[time_col], '구분': '예측 불량 (Predicted)', '상태': 'NG'})
                    
                    if timeline_data:
                        tl_df = pd.DataFrame(timeline_data)
                        fig_timeline = px.scatter(tl_df, x='시간', y='구분', color='구분', symbol='구분',
                                                  color_discrete_map={'실제 불량 (Actual)': '#e74c3c', '예측 불량 (Predicted)': '#f39c12'})
                        fig_timeline.update_traces(marker=dict(size=12, symbol='square'))
                        fig_timeline.update_layout(height=250, yaxis_title="", showlegend=False)
                        st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    st.markdown("---")

                with tab_sim:
                    # (B-2) 모델 성능 평가지표 (Confusion Matrix 등)
                    st.subheader(f" 모델 성능 정밀 진단")
                    
                    # --- [추가 3] 모델 성능 설명 문구 ---
                    st.info("""
                    전체 데이터로 구현한 제품 불량 예측 모델의 성능 지표입니다.
                    - **F1 Score**가 높을수록 전반적인 분류 성능이 우수합니다.
                    - 불량을 잘 찾아낼수록 **재현율(Recall)** 이 높고, 정상을 잘 판별할수록 **정밀도(Precision)** 수치가 증가합니다.
                    - 이는 우측의 **혼동 행렬(Confusion Matrix)** 지표를 통해 시각적으로 상세히 확인할 수 있습니다.
                    """)
                    # ----------------------------------

                    if 'NG_판정' in m_df.columns:
                        y_true = m_df['NG_판정'].astype(int)
                        y_pred = preds_fixed
                        
                        col_p1, col_p2, col_p3 = st.columns([1, 1, 1.5])
                        
                        with col_p1:
                            st.write("**예측 불량률**")
                            ng_count = y_pred.sum()
                            fig_donut = px.pie(names=['정상', '불량'], values=[len(y_pred)-ng_count, ng_count], hole=0.5,
                                               color_discrete_sequence=['#2ecc71', '#e74c3c'])
                            st.plotly_chart(fig_donut, use_container_width=True)
                        
                        with col_p2:
                            st.write("**평가지표**")
                            
                            # 1. 지표 계산
                            acc = accuracy_score(y_true, y_pred)
                            f1 = f1_score(y_true, y_pred, zero_division=0)
                            prec = precision_score(y_true, y_pred, zero_division=0) # 정밀도 추가
                            rec = recall_score(y_true, y_pred, zero_division=0)     # 재현율 추가

                            # 2. 성능 상태 알림
                            if f1 < 0.6: 
                                st.warning(" 예측도 낮음 주의")
                                st.metric("정확도", f"{acc:.2%}")
                                st.metric("F1 Score", f"{f1:.4f}")
                                st.metric("재현율", f"{rec:.4f}")   # Recall
                                st.metric("정밀도", f"{prec:.4f}") # Precision
                            else: 
                                st.success(" 성능 안정적")

                                st.metric("정확도", f"{acc:.2%}")
                                st.metric("F1 Score", f"{f1:.4f}")
                                st.metric("재현율", f"{rec:.4f}")   # Recall
                                st.metric("정밀도", f"{prec:.4f}") # Precision
                        
                        with col_p3:
                            st.write("**혼동 행렬**")
                            cm = confusion_matrix(y_true, y_pred)
                            fig_cm = ff.create_annotated_heatmap(cm.tolist(), x=['예측 정상','예측 불량'], y=['실제 정상','실제 불량'], colorscale='Blues')
                            st.plotly_chart(fig_cm, use_container_width=True)

                        # =================================================================
                        # TAB 2: 변수 조절 시뮬레이션 (What-If)
                        # =================================================================
                        st.divider()
                        st.subheader(" 변수 조절 시뮬레이션 (What-If)")

                        # --- [추가 4] 시뮬레이션 설명 문구 ---
                        st.info(" **사이드바의 시뮬레이션 설정**을 통해 변수(공정 조건) 변화에 따른 모델의 불량 예측 결과를 실시간으로 확인할 수 있으며, **24 Shot 묶음**에 미치는 영향도를 시각적으로 보여줍니다.")
                        # ------------------------------------

                        # 1. 데이터 계산
                        sim_df = pd.DataFrame([m_df.iloc[-1][features]])
                        for col, val in user_inputs.items(): sim_df[col] = val
                        
                        sim_prob = model.predict_proba(sim_df)[:, 1][0]
                        sim_pred = 1 if sim_prob >= sim_threshold else 0
                        
                        # 2. 결과 박스 스타일 설정
                        res_color = "#e74c3c" if sim_pred == 1 else "#2ecc71"
                        res_label = " 24shot 샘플링 불량 예상" if sim_pred == 1 else "✅ 24shot 샘플링 정상 예상"

                        sc1, sc2 = st.columns([1, 1.2])
                        
                        with sc1:
                            # 상단 메트릭 배치
                            m1, m2 = st.columns([1, 1.2])
                            with m1:
                                st.metric("**시뮬레이션 확률**", f"{sim_prob*100:.1f}%", 
                                          delta=f"{(sim_prob - probs[-1])*100:.1f}%p", delta_color="inverse")
                            with m2:
                                st.markdown(f"""
                                    <div style="background-color:{res_color}; padding:8px 12px; border-radius:8px; 
                                                text-align:center; margin-top:18px; border: 1px solid rgba(255,255,255,0.2);">
                                        <span style="color:white; font-size:14px; font-weight:bold;">{res_label}</span>
                                    </div>
                                    """, unsafe_allow_html=True)

                            # --- 1. 시뮬레이션 확률 바 차트 (Y축 전체를 채우는 설정) ---
                            fig_bar = px.bar(x=[sim_prob * 100], y=[" "], orientation='h', range_x=[0, 100],
                                             color_discrete_sequence=[res_color])
                            
                            fig_bar.update_layout(
                                height=150, 
                                margin=dict(l=10, r=40, t=50, b=40),
                                xaxis=dict(
                                    showgrid=False, 
                                    showline=True,
                                    linewidth=1, 
                                    linecolor='black', 
                                    showticklabels=True, 
                                    tickmode='linear',
                                    tick0=0,
                                    dtick=10,
                                    title="불량률 (%)"
                                ),
                                yaxis=dict(
                                    showgrid=False, 
                                    showline=False, 
                                    showticklabels=False, 
                                    title="",
                                    range=[-0.5, 0.5] # 바가 Y축을 꽉 채우도록 범위 제한
                                ),
                                plot_bgcolor="rgba(0,0,0,0)",
                                bargap=0,         # 바 사이의 간격을 없앰
                                bargroupgap=0     # 그룹 간의 간격을 없앰
                            )
                            
                            # 임계값 점선 추가
                            fig_bar.add_vline(
                                x=sim_threshold * 100, 
                                line_dash="dash", 
                                line_color="black", 
                                line_width=2,
                                layer="above"
                            )
                            
                            # 임계값 % 수치 표시 (점선 위쪽 옆)
                            fig_bar.add_annotation(
                                x=sim_threshold * 100,
                                y=0,
                                text=f"<b>{sim_threshold*100:.0f}%</b>", 
                                showarrow=False,
                                xshift=7,
                                yshift=60, # 바가 두꺼워졌으므로 위치 상향 조정
                                font=dict(color="black", size=14)
                            )
                            
                            st.plotly_chart(fig_bar, use_container_width=True)

                        with sc2:
                            # 서브제목을 오른쪽으로 살짝 밀기 (&nbsp; 추가)
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp; **24shot 품질 현황** ")
                            
                            # --- [수정] 설비별 NG 판정률(%) 기준 정의 ---
                            # 1로트 전체의 불량률이 이 수치를 넘어가면 불량 로트로 판정
                            LOT_THRESHOLD_PCT = {
                                "D01": 15.289256, "D10": 14.025067, "F09": 16.692190, "F11": 14.945010,
                                "G01": 16.858196, "G02": 16.981823, "G03": 15.340087, "G05": 17.495560,
                                "G06": 19.157184, "G10": 14.832113
                            }
                            
                            num_shots = 24
                            # 현재 선택된 설비의 불량률 기준값 가져오기 (기본값 15.0%)
                            current_threshold_pct = LOT_THRESHOLD_PCT.get(str(selected_machine), 15.0)
                            
                            # 불량률(%)을 기준으로 24Shot 중 허용 가능한 최대 불량 Shot 개수 계산 (소수점 버림)
                            # 예: 15.28% -> 3.66개 -> 3개까지는 정상, 4개부터 불량 로트
                            max_ok_shot_count = int(num_shots * (current_threshold_pct / 100))

                            # [고정 개수 로직 적용]
                            # AI 모델이 '불량'으로 예측하면 -> 기준값 + 1개 (불량 최소 기준) 표시
                            # AI 모델이 '정상'으로 예측하면 -> 기준값 (정상 최대 기준) 표시
                            if sim_pred == 1:
                                num_shot_ng = max_ok_shot_count + 1
                                if num_shot_ng > 24: num_shot_ng = 24
                            else:
                                num_shot_ng = max_ok_shot_count # 정상을 나타내는 대표값 (허용 한계치)
                            
                            # Grid 구성을 위한 리스트 생성
                            status_list = ([1] * num_shot_ng) + ([0] * (num_shots - num_shot_ng))
                            import random
                            random.seed(42) # 고정된 패턴을 원하시면 seed 유지, 매번 섞고 싶으면 제거
                            random.shuffle(status_list)

                            lot_data = pd.DataFrame({
                                'shot_id': [f"S{i+1}" for i in range(num_shots)],
                                '상태': ["🚨 불량" if s == 1 else "✅ 정상" for s in status_list],
                                'x': [i % 6 for i in range(num_shots)],
                                'y': [i // 6 for i in range(num_shots)]
                            })
                            
                            # 시각화 (Scatter Plot)
                            fig_sim = px.scatter(
                                lot_data, x='x', y='y', text='shot_id', color='상태',
                                color_discrete_map={"🚨 불량": "#e74c3c", "✅ 정상": "#2ecc71"},
                                range_x=[-0.6, 5.6], range_y=[-0.6, 3.6]
                            )
                            fig_sim.update_traces(
                                marker=dict(size=35, symbol='square', line=dict(width=1, color='white')),
                                textfont=dict(color='white', size=8)
                            )
                            fig_sim.update_layout(
                                height=280, 
                                margin=dict(l=10, r=10, t=10, b=10),
                                xaxis=dict(showgrid=False, showline=False, showticklabels=False, title=""),
                                yaxis=dict(showgrid=False, showline=False, showticklabels=False, title="", autorange="reversed"),
                                showlegend=False,
                                plot_bgcolor="rgba(0,0,0,0)"
                            )
                            st.plotly_chart(fig_sim, use_container_width=True)

                            
                            # 상태에 따른 색상 및 텍스트 설정
                            if sim_pred == 1:
                                state_text = f"<span style='color:#e74c3c; font-weight:bold;'>불량(NG) (불량 샷 {num_shot_ng}개 이상)</span>"
                            else:
                                state_text = f"<span style='color:#2ecc71; font-weight:bold;'>정상(OK) (불량 샷 {num_shot_ng}개 이하)</span>"

                            st.markdown(f"""
                                <div style="text-align: center; color: black; font-size: 0.95em; margin-top: -5px; line-height: 1.5; background-color: #f0f2f6; padding: 10px; border-radius: 5px;">
                                     <b>{selected_machine} 설비</b>의 24shot 묶음에서 불량 샷(NG Shot)은<br>
                                     <b>{max_ok_shot_count}개 이하</b>일 때 로트가 <b>정상</b>으로 판정됩니다.<br>
                                     (현재 시뮬레이션 상태: {state_text})
                                </div>
                            """, unsafe_allow_html=True)
            else:
                st.error("모델 파일을 찾을 수 없습니다.")
else:
    st.error("CSV 파일을 찾을 수 없습니다.")





