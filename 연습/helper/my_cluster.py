from pandas import DataFrame
from . import my_prep
from . import my_plot
from . import RANDOM_STATE

# 군집분석 관련 참조
from sklearn.cluster import KMeans


# -----------------------------------------------------
# KMeans 군집화 함수 정의 및 작업단계 구성
# -----------------------------------------------------
def kmeans(data, k, columns=None, scaling='standard', cluster_name='그룹번호', random_state=RANDOM_STATE, n_init=10, verbose=True, plot=True, x=None, y=None,
            title=None, palette='tab10', size=100, edgecolor="#ffffff", linewidth=1.5, alpha=1, outline=True, center=True, center_marker='X', center_size=150,
            center_color="#ff0000", center_edgecolor="#000000", center_linewidth=1.5, equal_scale=False, width=1280, height=640, save_path=None, ax=None):
    """데이터를 k개의 군집으로 나누고, 군집 결과와 중심점을 시각화하는 함수
    
    K-Means는 데이터 사이의 '거리'로 그룹을 나누므로, 변수마다 값의 범위가 다르면
    범위가 큰 변수 하나가 거리를 지배하게 된다. 그래서 스케일링을 기본으로 수행한다.
    
    Args (기본값은 위의 함수 정의 참고):
        data, k: 군집화할 데이터프레임과 나눌 군집의 개수
        columns, cluster_name, random_state: 사용할 컬럼(None이면 수치형 전체),
            군집 번호를 저장할 컬럼명, 중심점의 초기 위치를 결정하는 랜덤시드
        scaling: 스케일러 이름('standard' / 'minmax' / 'robust' / 'maxabs', None이면 원본 값)
        verbose: 스케일링 전후의 값의 범위를 출력할지 여부
        plot, x, y, title: 시각화 여부, 산점도의 x·y축 컬럼명(None이면 대상 컬럼의 앞 두 개),
            그래프 제목(None이면 군집 개수를 포함하여 자동 생성)
        palette, size, edgecolor, linewidth, alpha : 데이터 포인트의 색상 팔레트.
            마커 크기, 테두리 색상, 테두리 두께, 투명도
        outline : 군집의 외곽선(ConvexHull)을 표시할지 여부
        center : 모델이 찾은 중심점을 표시할지 여부
        center_marker, center_size, center_color, center_edgecolor, center_linewidth:
            중심점의 마커 모양, 크기, 색상, 테두리 색상, 테두리 두께
        equal_scale : x축의 범위를 y축과 동일하게 맞출지 여부
            (모델이 실제로 본 거리 관계를 그대로 확인할 때 사용한다)
        width, height, save_path, ax: 캔버스 가로·세로 픽셀, 저장 경로,
            그래프를 그릴 Axes 객체(None이면 새로 생성)
            
        Returns:
            tuple : (estimator, df, center_df) - 학습이 완료된 모델,
                군집 번호 컬럼이 추가된 데이터(스케일링 적용 후),
                각 군집의 중심점 좌표(컬럼명은 대상 컬럼과 동일)
        """

    # 1) 군집화에 사용할 컬럼 결정
    # 지정이 없으면 수치형 컬럼만 자동 선택 (문자열 컬럼은 거리 계산이 불가능하다)
    if columns is None:
        columns = list(data.select_dtypes(include='number').columns)

    # 2) 스케일링 적용
    # 변수마다 값의 범위가 다르면 범위가 큰 변수 하나만 보고 그룹을 나눈 셈이 되므로
    # 거리 계산 전에 두 변수의 눈금을 같은 기준으로 맞춘다
    if scaling:
        df = my_prep.scaling(data[columns], method=scaling, verbose=verbose)
    else:
        df = data[columns].copy()

    # 3) 모델 생성 및 학습 (중심점을 찾는 과정)
    estimator = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    estimator.fit(df)

    # 4) 각 데이터가 몇 번 그룹인지 예측하여 컬럼으로 추가
    df[cluster_name] = estimator.predict(df)

    # 5) 모델이 찾은 중심점을 데이터프레임으로 구성
    center_df = DataFrame(estimator.cluster_centers_, columns=columns)

    # 6) 군집 결과 시각화
    if plot:
        # 축으로 사용할 컬럼 결정 (지정이 없으면 대상 컬럼의 앞에서 두 개)
        if x is None:           x = columns[0]
        if y is None:           y = columns[1]

        # 제목을 지정하지 않은 경우 군집 개수를 포함한 제목을 자동으로 생성
        if title is None:       title = f'K-Means 군집 결과 (k={k})'

        # 그래프 초기화 (ax를 전달받은 경우에는 그 위에 겹쳐 그린다)
        fig = None
        if ax is None:
            fig, ax = my_plot.init(width=width, height=height, title=title,
                                   xlabel=x, ylabel=y)

        # 군집별 산점도 (outline=True이면 각 군집의 외곽선까지 표시)
        my_plot.scatterplot(data=df, x=x, y=y, hue=cluster_name, palette=palette, size=size, edgecolor=edgecolor,
                            linewidth=linewidth, alpha=alpha, outline=outline, ax=ax)

        # 모델이 찾은 중심점을 덧그리기 (중심점의 x좌표에 대한 y좌표 산점도)
        if center:
            my_plot.scatterplot(data=center_df, x=x, y=y, marker=center_marker, size=center_size,
                                color=center_color, edgecolor=center_edgecolor,
                                linewidth=center_linewidth, outline=False, ax=ax)

        # x축의 범위를 y축과 동일하게 맞춰 실제 거리 관계를 확인
        if equal_scale:
            ax.set_xlim(ax.get_ylim())

        # 그래프 표시 (ax를 전달받은 경우에는 호출한 쪽에서 표시한다)
        if fig is not None:
            my_plot.show(save_path=save_path)

    # 7) 모델, 군집결과, 중심점 좌표 반환
    return estimator, df, center_df


# --------------------------------------------------------
# 최적 k 탐색 공통함수 정의
# --------------------------------------------------------
def _prepare_k_search(data, klist, columns, scaling, verbose):
    """최적 k 탐색 함수들이 공통으로 수행하는 컬럼 선택·스케일링·k 목록 정리
    
    여러 k를 같은 데이터로 비교해야 하므로 스케일링은 반복문 밖에서 한 번만 수행한다.
    
    Args:
        data, klist : 군집화할 데이터, 확인할 k 목록(None이면 2~10)
        columns, scaling, verbose: 사용할 컬럼(None이면 수치형 전체),
            스케일러 이름(None이면 원본 값), 스케일링 전후의 값의 범위 출력 여부
            
    Returns:
        tuple : (df, klist) - 스케일링이 끝난 데이터, 2 이상만 남긴 k 목록
    """

    # 대상 컬럼이 없다면 숫자형태의 컬럼만 추려낸다.
    # --> 거리기반 알고리즘이므로 문자열 형태는 처리하지 못한다.
    if columns is None:
        columns = list(data.select_dtypes(include='number').columns)

    # 스케일링 k마다 반복하면 매번 같은 계산을 다시 하는 셈이므로 여기서 한 번만 처리한다
    if scaling:
        df = my_prep.scaling(data[columns], method=scaling, verbose=verbose)
    else:
        df = data[columns].copy()

    # 실루엣 개수는 "다른 군집과의 거리"가 있어야 정의되므로 k는 2부터 확인한다
    klist = list(range(2, 11)) if klist is None else [k for k in klist if k >= 2]

    # 스케일링 후의 데이터와 2 이상만 남긴 k 목록을 반환한다
    return df, klist


# -----------------------------------------------
# 엘보우 포인트 기반 최적 k찾기 함수
# -----------------------------------------------
def best_k_elbow(data, klist=None, columns=None, scaling='standard', sensitivity=0.01, random_state=RANDOM_STATE, n_init=10, verbose=True,
                 plot=True, title=None, color="#1f77b4", marker='o', linestyle=':', best_color='#ff0000', width=1280, height=640, save_path=None, ax=None):
    """이너셔의 감소폭이 꺾이는 지점(엘보우 포인트)을 찾아 최적의 k를 추정하는 함수
    
    이너셔(군집 중심까지 거리의 제곱합)는 k가 커지면 항상 줄어들기 때문에 가장 작은 값을
    고르는 것은 의미가 없다. 대신 "k를 하나 더 늘려도 이제 별 이득이 없어지는" 지점을 찾는다.
    꺾이는 지점은 KneeLocator가 계산하며, 이너셔는 뭉침만 보는 지표이므로 이 결과는 최종 답이 아니라 후보다.
    
    Args (기본값은 위의 함수 정의 참고) : 
        data, klist : 군집화할 데이터, 확인할 k 목록(None이면 2~10)
        columns, scaling, random_state : 사용할 컬럼(None이면 수치형 전체),
            스케일러 이름(None이면 원본 값), 중심점의 초기 위치를 결정하는 랜덤시드
        n_init: 시작 위치를 바꿔 가며 시도할 횟수 (k끼리 공정하게 비교하려면 1회로는 부족하다)
        sensitivity : KneeLocator의 민감도(S). 작을수록 작은 꺾임에도 반응한다
        verbose, plot, title : 계산 결과 출력 여부, 시각화 여부, 그래프 제목 (None이면 자동 생성)
        color, marker, linestyle, best_color: 이너셔 선의 색상·마커 모양·선 스타일,
            엘보우 포인트를 표시할 세로선의 색상
        width, height, save_path, ax : 캔버스 가로·세로 픽셀, 저장 경로,
            그래프를 그릴 Axes 객체(None이면 새로 생성)
            
    Returns:
        tuple : (best_k, result_df) - 엘보우 포인트, k별 이너셔와 감소량이 담긴 데이터프레임
    """

    # 0) 공통 준비 작업
    df, klist = _prepare_k_search(data, klist, columns, scaling, verbose)

    # 1) k를 눌러 가며 이너셔 수집
    inertia = []
    # 2) k가 1 늘어날 때마다 이너셔가 얼마나 줄었는지 계산

    # 3) 결과 정리

    # 4) 엘보우 포인트 찾기

    # 5) 시각화
