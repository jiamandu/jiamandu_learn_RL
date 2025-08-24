import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

boundary_return = -1
forbidden_return = -10
target_return = 1
available_return = 0
gama = 0.9

World_Matrix = np.zeros((5, 5))

Action_Reward_Matrix = np.zeros((5, 5, 5))

Pi_Matrix = np.zeros((5, 5, 5))

State_Matrix = np.zeros((5, 5))

Q_Matrix = np.zeros((5, 5, 5))

def Make_World(matrix):
    matrix[1][1] = -1
    matrix[1][2] = -1
    matrix[2][2] = -1
    matrix[3][1] = -1
    matrix[3][2] = 1
    matrix[3][3] = -1
    matrix[4][1] = -1

def Make_Action_Reward(matrix,world_matrix):
    for a in range(0,5):
        if a == 0:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if i == 0:
                        matrix[i][j][a] = boundary_return
                    else:
                        if world_matrix[i-1][j] == 1:
                            matrix[i][j][a] = target_return
                        elif world_matrix[i-1][j] == -1:
                            matrix[i][j][a] = forbidden_return
                        else:
                            matrix[i][j][a] = available_return

        if a == 1:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if j == matrix.shape[1]-1:
                        matrix[i][j][a] = boundary_return
                    else:
                        if world_matrix[i][j+1] == 1:
                            matrix[i][j][a] = target_return
                        elif world_matrix[i][j+1] == -1:
                            matrix[i][j][a] = forbidden_return
                        else:
                            matrix[i][j][a] = available_return

        if a == 2:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if i == matrix.shape[0]-1:
                        matrix[i][j][a] = boundary_return
                    else:
                        if world_matrix[i+1][j] == 1:
                            matrix[i][j][a] = target_return
                        elif world_matrix[i+1][j] == -1:
                            matrix[i][j][a] = forbidden_return
                        else:
                            matrix[i][j][a] = available_return

        if a == 3:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if j == 0:
                        matrix[i][j][a] = boundary_return
                    else:
                        if world_matrix[i][j-1] == 1:
                            matrix[i][j][a] = target_return
                        elif world_matrix[i][j-1] == -1:
                            matrix[i][j][a] = forbidden_return
                        else:
                            matrix[i][j][a] = available_return

        if a == 4:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if world_matrix[i][j] == 1:
                        matrix[i][j][a] = target_return
                    elif world_matrix[i][j] == -1:
                        matrix[i][j][a] = forbidden_return
                    else:
                        matrix[i][j][a] = available_return

def Make_Q_Reward(action_reward_matrix, state_matrix):
    next_states = np.stack([
        np.concatenate([state_matrix[:1, :], state_matrix[:-1, :]], axis=0),
        np.concatenate([state_matrix[:, 1:], state_matrix[:, -1:]], axis=1),
        np.concatenate([state_matrix[1:, :], state_matrix[-1:, :]], axis=0),
        np.concatenate([state_matrix[:, :1], state_matrix[:, :-1]], axis=1),
        state_matrix
    ], axis=2)

    return action_reward_matrix + gama * next_states

def Make_Gama_P_Vk(state_matrix, pi_matrix):
    next_states = np.stack([
        np.concatenate([state_matrix[:1, :], state_matrix[:-1, :]], axis=0),
        np.concatenate([state_matrix[:, 1:], state_matrix[:, -1:]], axis=1),
        np.concatenate([state_matrix[1:, :], state_matrix[-1:, :]], axis=0),
        np.concatenate([state_matrix[:, :1], state_matrix[:, :-1]], axis=1),
        state_matrix
    ], axis=2)

    # 计算加权的下一个状态价值
    return gama * np.sum(pi_matrix * next_states, axis=2)

def policy_iteration(pi_matrix, action_reward_matrix, state_matrix,world_matrix):
    while True:
        q_matrix = Make_Q_Reward(action_reward_matrix, state_matrix)
        max_indices = np.argmax(q_matrix, axis=2)
        while True:
            r_matrix = np.take_along_axis(action_reward_matrix, max_indices[..., np.newaxis], axis=2).squeeze(axis=2)
            gama_p_vk_matrix = Make_Gama_P_Vk(state_matrix, pi_matrix)

            ep = np.linalg.norm(state_matrix-r_matrix-gama_p_vk_matrix)
            if ep <= 1e-6:
                print('inner')
                break
            state_matrix = r_matrix + gama_p_vk_matrix


        q_matrix = Make_Q_Reward(action_reward_matrix, state_matrix)
        max_indices = np.argmax(q_matrix, axis=2)
        new_pi_matrix = np.eye(action_reward_matrix.shape[2])[max_indices]

        ep = np.linalg.norm(pi_matrix - new_pi_matrix)
        print('epoch',ep)
        if ep <= 1e-6:
            Draw_Result(world_matrix, pi_matrix)
            break

        pi_matrix = new_pi_matrix

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

    for i in range(pi_matrix.shape[0]):
        for j in range(pi_matrix.shape[1]):
            for a in range(5):
                if pi_matrix[i, j, a] == 1:
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

if __name__ == "__main__":
    Make_World(World_Matrix)
    Make_Action_Reward(Action_Reward_Matrix, World_Matrix)
    policy_iteration(Pi_Matrix, Action_Reward_Matrix, State_Matrix, World_Matrix)



