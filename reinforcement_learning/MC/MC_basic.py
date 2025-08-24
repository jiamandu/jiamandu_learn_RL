import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

boundary_return = -1
forbidden_return = -10
target_return = 1
available_return = 0
gama = 0.9
mc_q_times = 16

World_Matrix = np.zeros((5, 5))

Action_Reward_Matrix = np.zeros((5, 5, 5))

Pi_Matrix = np.zeros((5, 5, 5))

State_Matrix = np.zeros((5, 5))

Q_Matrix = np.zeros((5, 5, 5))

Move_Matrix = np.zeros((2, 5, 5, 5))

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

def Make_Q_Matrix(action_reward_matrix, pi_matrix, q_matrix, mov_matrix):
    for i in range(action_reward_matrix.shape[1]):
        for j in range(action_reward_matrix.shape[2]):
            for a in range(action_reward_matrix.shape[0]):

                q_matrix[a, i, j] = action_reward_matrix[a, i, j]
                state = [int(mov_matrix[0, a, i, j]), int(mov_matrix[1, a, i, j])]

                temp_gama = 1
                for n in range(1, mc_q_times):
                    temp_gama = gama * temp_gama
                    for next_a in range(action_reward_matrix.shape[0]):
                        if pi_matrix[next_a, state[0], state[1]] == 1:
                            q_matrix[a, i, j] = q_matrix[a, i, j] + temp_gama * action_reward_matrix[next_a, state[0], state[1]]
                            state = [int(mov_matrix[0, next_a, state[0], state[1]]), int(mov_matrix[1, next_a, state[0], state[1]])]

def MC_Policy_Itaration(action_reward_matrix, pi_matrix, q_matrix, mov_matrix):
    pi_matrix[0, :, :] = 1
    while True:
        Make_Q_Matrix(action_reward_matrix, pi_matrix, q_matrix, mov_matrix)

        max_indices = np.argmax(q_matrix, axis=0)
        new_pi_matrix = np.eye(action_reward_matrix.shape[0])[:,max_indices]


        ep = np.linalg.norm(pi_matrix - new_pi_matrix)
        pi_matrix = new_pi_matrix
        print('ep',ep)
        if ep <= 1e-6:
            break

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

    for i in range(pi_matrix.shape[1]):
        for j in range(pi_matrix.shape[2]):
            for a in range(5):
                if pi_matrix[a, i, j] == 1:
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
    MC_Policy_Itaration(Action_Reward_Matrix, Pi_Matrix, Q_Matrix, Move_Matrix)