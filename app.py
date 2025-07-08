import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappop, heappush
import time
from matplotlib.colors import ListedColormap

# A* 알고리즘
def a_star(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

    while open_set:
        _, current = heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            neighbor = (current[0]+dx, current[1]+dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor] == 1:
                    continue
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal)
                    heappush(open_set, (f_score, neighbor))
    return None

# 격자 지도 생성
def generate_grid(rows, cols, obstacle_ratio):
    grid = np.zeros((rows, cols), dtype=int)
    obstacle_count = int(rows * cols * obstacle_ratio)
    obstacle_indices = np.random.choice(rows * cols, obstacle_count, replace=False)
    for idx in obstacle_indices:
        x, y = divmod(idx, cols)
        grid[x, y] = 1
    return grid

# 장애물 이동
def update_obstacles(grid, move_count=3):
    rows, cols = grid.shape
    obstacle_positions = np.argwhere(grid == 1)
    empty_positions = np.argwhere(grid == 0)
    for _ in range(move_count):
        if len(obstacle_positions) == 0 or len(empty_positions) == 0:
            break
        from_pos = tuple(obstacle_positions[np.random.randint(len(obstacle_positions))])
        to_pos = tuple(empty_positions[np.random.randint(len(empty_positions))])
        grid[from_pos] = 0
        grid[to_pos] = 1
    return grid

# 시각화 함수 (색상 명확하게)
def plot_path(grid, path, start, goal, title="A* Pathfinding"):
    display_grid = np.copy(grid).astype(float)
    for (x, y) in path or []:
        if (x, y) != start and (x, y) != goal:
            display_grid[x, y] = 0.5  # path

    display_grid[start] = 0.7
    display_grid[goal] = 0.9

    # 색상 맵 설정
    custom_cmap = ListedColormap(["black", "blue", "green", "red", "white"])
    cmap_indices = np.zeros_like(display_grid)
    cmap_indices[display_grid == 0] = 0      # 빈 공간
    cmap_indices[display_grid == 0.5] = 1    # 경로
    cmap_indices[display_grid == 0.7] = 2    # 시작
    cmap_indices[display_grid == 0.9] = 3    # 목표
    cmap_indices[display_grid == 1] = 4      # 장애물

    fig, ax = plt.subplots()
    ax.imshow(cmap_indices, cmap=custom_cmap)
    ax.set_title(title)
    ax.axis('off')
    st.pyplot(fig)
    plt.close(fig)

# Streamlit 앱 설정
st.set_page_config(page_title="실시간 A* 경로 탐색", layout="centered")
st.title("🚗 실시간 장애물 대응 A* 경로 탐색 시뮬레이터")

# 사용자 입력
rows = st.sidebar.slider("격자 행 수", 5, 20, 10)
cols = st.sidebar.slider("격자 열 수", 5, 20, 10)
obstacle_ratio = st.sidebar.slider("초기 장애물 비율", 0.0, 0.4, 0.2, step=0.05)
move_interval = st.sidebar.slider("장애물 변경 주기 (초)", 0.5, 3.0, 1.0, step=0.5)
steps = st.sidebar.slider("시뮬레이션 반복 횟수", 1, 10, 5)

start = (0, 0)
goal = (rows - 1, cols - 1)

if st.button("▶️ 시뮬레이션 시작"):
    grid = generate_grid(rows, cols, obstacle_ratio)
    for step in range(steps):
        st.subheader(f"🔁 Step {step + 1} / {steps}")
        path = a_star(grid, start, goal)
        if path:
            st.success(f"경로 길이: {len(path)}")
        else:
            st.warning("경로를 찾을 수 없습니다.")
        plot_path(grid, path, start, goal, title=f"Step {step + 1}")
        grid = update_obstacles(grid, move_count=2)
        time.sleep(move_interval)
