import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import deque

from sympy.abc import epsilon

boundary_return = -1
forbidden_return = -10
target_return = 1
available_return = 0
gama = 0.95
mc_q_times = 12
epsilon_start = 0.5
epsilon_end = 0.01
exploring_steps = 100000000

World_Matrix = np.zeros((5, 5))

Action_Reward_Matrix = np.zeros((5, 5, 5))

Pi_Matrix = np.zeros((5, 5, 5))

State_Matrix = np.zeros((5, 5))

Q_Matrix = np.zeros((5, 5, 5))

Move_Matrix = np.zeros((2, 5, 5, 5))

Visit_Count_Matrix = np.zeros((5, 5, 5))

def Make_World(matrix):
    matrix[1][1] = -1
    matrix[1][2] = -1
    matrix[2][2] = -1
    matrix[3][1] = -1
    matrix[3][2] = 1
    matrix[3][3] = -1
    matrix[4][1] = -1

def Make_Action_Reward(matrix, world_matrix):
    for a in range(0, 5):
        if a == 0:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if i == 0:
                        matrix[a][i][j] = boundary_return
                    else:
                        if world_matrix[i - 1][j] == 1:
                            matrix[a][i][j] = target_return
                        elif world_matrix[i - 1][j] == -1:
                            matrix[a][i][j] = forbidden_return
                        else:
                            matrix[a][i][j] = available_return

        if a == 1:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if j == matrix.shape[1] - 1:
                        matrix[a][i][j] = boundary_return
                    else:
                        if world_matrix[i][j + 1] == 1:
                            matrix[a][i][j] = target_return
                        elif world_matrix[i][j + 1] == -1:
                            matrix[a][i][j] = forbidden_return
                        else:
                            matrix[a][i][j] = available_return

        if a == 2:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if i == matrix.shape[0] - 1:
                        matrix[a][i][j] = boundary_return
                    else:
                        if world_matrix[i + 1][j] == 1:
                            matrix[a][i][j] = target_return
                        elif world_matrix[i + 1][j] == -1:
                            matrix[a][i][j] = forbidden_return
                        else:
                            matrix[a][i][j] = available_return

        if a == 3:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if j == 0:
                        matrix[a][i][j] = boundary_return
                    else:
                        if world_matrix[i][j - 1] == 1:
                            matrix[a][i][j] = target_return
                        elif world_matrix[i][j - 1] == -1:
                            matrix[a][i][j] = forbidden_return
                        else:
                            matrix[a][i][j] = available_return

        if a == 4:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if world_matrix[i][j] == 1:
                        matrix[a][i][j] = target_return
                    elif world_matrix[i][j] == -1:
                        matrix[a][i][j] = forbidden_return
                    else:
                        matrix[a][i][j] = available_return

def Make_Move_Matrix(move_matrix):
    for i in range(move_matrix.shape[2]):
        for j in range(move_matrix.shape[3]):

            for a in range(move_matrix.shape[1]):
                if a == 0:
                    if i == 0:
                        move_matrix[0, a, i, j] = i
                        move_matrix[1, a, i, j] = j
                    else:
                        move_matrix[0, a, i, j] = i - 1
                        move_matrix[1, a, i, j] = j
                elif a == 1:
                    if j == move_matrix.shape[3] - 1:
                        move_matrix[0, a, i, j] = i
                        move_matrix[1, a, i, j] = j
                    else:
                        move_matrix[0, a, i, j] = i
                        move_matrix[1, a, i, j] = j + 1
                elif a == 2:
                    if i == move_matrix.shape[2] - 1:
                        move_matrix[0, a, i, j] = i
                        move_matrix[1, a, i, j] = j
                    else:
                        move_matrix[0, a, i, j] = i + 1
                        move_matrix[1, a, i, j] = j
                elif a == 3:
                    if j == 0:
                        move_matrix[0, a, i, j] = i
                        move_matrix[1, a, i, j] = j
                    else:
                        move_matrix[0, a, i, j] = i
                        move_matrix[1, a, i, j] = j - 1
                else:
                    move_matrix[0, a, i, j] = i
                    move_matrix[1, a, i, j] = j

def Init_Pi_Matrix(pi_matrix):
    for i in range(0,5):
        if i == 4:
            pi_matrix[i, :, :] = 1 - 0.8 * epsilon_start
        else:
            pi_matrix[i, :, :] = 0.2 * epsilon_start

def Mov_Iteration(pi_matrix, mov_matrix, q_matrix, action_reward_matrix, visit_count_matrix):

    dq = deque(maxlen=mc_q_times)
    snake = [0,0]
    epsilon = epsilon_start
    for step in range(exploring_steps):
        random_action = np.random.rand()
        for a in range(0, 5):
            random_action -= pi_matrix[a, snake[0], snake[1]]
            if random_action <= 0:
                break

        dq.append([a, snake[0], snake[1]])
        if len(dq) == dq.maxlen:
            reward = 0
            mul_gama = 1
            for i in range(dq.maxlen):
                reward += action_reward_matrix[dq[i][0],dq[i][1],dq[i][2]] * mul_gama
                mul_gama *= gama

            visit_count_matrix[dq[0][0],dq[0][1],dq[0][2]] += 1

            q_matrix[dq[0][0],dq[0][1],dq[0][2]] += (reward - q_matrix[dq[0][0],dq[0][1],dq[0][2]]) / visit_count_matrix[dq[0][0],dq[0][1],dq[0][2]]

            num_actions, height, width = pi_matrix.shape
            max_indices = np.argmax(q_matrix, axis=0)
            pi_matrix.fill(epsilon / num_actions)
            for i in range(height):
                for j in range(width):
                    best_action = max_indices[i, j]
                    pi_matrix[best_action, i, j] = (1 - epsilon) + (epsilon / num_actions)


        snake = [int(mov_matrix[0, a, snake[0], snake[1]]), int(mov_matrix[1, a, snake[0], snake[1]])]

        if step % 100000 == 0:
            epsilon = -(epsilon_start-epsilon_end)/exploring_steps*step + epsilon_start
            print('epoch', step, epsilon)

    Draw_Result(World_Matrix, pi_matrix)

def Draw_Result(matrix,pi_matrix):

    colors = ['yellow', 'white', 'green']  # 对应 -1, 0, 1
    cmap = ListedColormap(colors)

    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # 绘制矩阵，vmin和vmax确保颜色映射正确
    im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1)

    # 设置行列标号
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xticklabels(range(matrix.shape[1]))
    ax.set_yticklabels(range(matrix.shape[0]))

    # 添加网格线
    for i in range(matrix.shape[0] + 1):
        ax.axhline(i - 0.5, color='black', linewidth=1)
    for j in range(matrix.shape[1] + 1):
        ax.axvline(j - 0.5, color='black', linewidth=1)

    max_indices = np.argmax(pi_matrix, axis=0)
    for i in range(pi_matrix.shape[1]):
        for j in range(pi_matrix.shape[2]):
            a = max_indices[i, j]
            if a == 0:
                ax.text(j, i, f'↑',
                        ha="center", va="center", color="black", fontsize=12)
            elif a == 1:
                ax.text(j, i, f'→',
                        ha="center", va="center", color="black", fontsize=12)
            elif a == 2:
                ax.text(j, i, f'↓',
                        ha="center", va="center", color="black", fontsize=12)
            elif a == 3:
                ax.text(j, i, f'←',
                        ha="center", va="center", color="black", fontsize=12)
            else:
                ax.text(j, i, f'o',
                        ha="center", va="center", color="black", fontsize=12)

    # 设置标题和标签
    ax.set_title('World Matrix Visualization', fontsize=16, fontweight='bold')
    ax.set_xlabel('Column Index', fontsize=12)
    ax.set_ylabel('Row Index', fontsize=12)

    # 创建颜色条说明
    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1])
    cbar.set_ticklabels(['forbidden (Yellow)', 'available (White)', 'target (Green)'])
    cbar.set_label('Matrix Values', fontsize=12)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    Make_World(World_Matrix)
    Make_Action_Reward(Action_Reward_Matrix, World_Matrix)
    Make_Move_Matrix(Move_Matrix)
    Init_Pi_Matrix(Pi_Matrix)
    Mov_Iteration(Pi_Matrix, Move_Matrix, Q_Matrix, Action_Reward_Matrix, Visit_Count_Matrix)