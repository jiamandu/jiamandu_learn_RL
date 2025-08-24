import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from MC.MC_epsilon_greedy import available_return

boundary_return = -10
forbidden_return = -10
target_return = 1
available_return = 0

epsilon = 0.4
alpha = 0.1
gama = 0.95

TD_times = 2000000
truncate_count = 500

World_Matrix = np.zeros((5, 5))

Action_Reward_Matrix = np.zeros((5, 5, 5))

Pi_Matrix = np.zeros((5, 5, 5))

State_Matrix = np.zeros((5, 5))

Q_Matrix = np.zeros((5, 5, 5))

Move_Matrix = np.zeros((2, 5, 5, 5))

pb = 1 - epsilon * (1-1/Pi_Matrix.shape[0])
ps = epsilon / Pi_Matrix.shape[0]

def Make_World_Matrix(matrix):
    matrix[1][1] = -1
    matrix[1][2] = -1
    matrix[2][2] = -1
    matrix[3][1] = -1
    matrix[3][2] = 1
    matrix[3][3] = -1
    matrix[4][1] = -1

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

def Make_Pi_Matrix(pi_matrix, epsilon):
    for i in range(pi_matrix.shape[1]):
        for j in range(pi_matrix.shape[2]):
            greedy_index = np.random.randint(0, pi_matrix.shape[0])
            for k in range(pi_matrix.shape[0]):
                if k == greedy_index:
                    pi_matrix[k, i, j] = pb
                else:
                    pi_matrix[k, i, j] = ps

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

def Choose_Action(pi_matrix, mov_matrix, row, colum, action_reward_matrix):
    random_p = np.random.rand()
    for a in range(0, 5):
        random_p -= pi_matrix[a, row, colum]
        if random_p <= 0:
            break
    return a, int(mov_matrix[0, a, row, colum]), int(mov_matrix[1, a, row, colum]), action_reward_matrix[a, row, colum]

def Calculate_Q_Matrix(q_matrix, state_matrix, mov_matrix, action_reward_matrix):
    for i in range(q_matrix.shape[1]):
        for j in range(q_matrix.shape[2]):
            for a in range(q_matrix.shape[0]):
                i_mov = int(mov_matrix[0, a, i, j])
                j_mov = int(mov_matrix[1, a, i, j])
                q_matrix[a, i, j] = action_reward_matrix[a, i, j] + gama * state_matrix[i_mov, j_mov]

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

def TD_Iteration(world_matrix, pi_matrix, mov_matrix, state_matrix, action_reward_matrix, q_matrix):
    for iteration in range(TD_times):
        row = 2
        colum = 0
        count = 0
        while world_matrix[row, colum] != target_return and count != truncate_count:
            count += 1

            #q update
            action_now, row_next, colum_next, action_reward = Choose_Action(pi_matrix, mov_matrix, row, colum,action_reward_matrix)
            action_next, _, _, _ = Choose_Action(pi_matrix, mov_matrix, row_next, colum_next,action_reward_matrix)

            q_t = q_matrix[action_now, row, colum]
            q_t_next = q_matrix[action_next, row_next, colum_next]

            q_tp1 = q_t - alpha * (q_t - (action_reward + gama * q_t_next))

            q_matrix[action_now, row, colum] = q_tp1

            #policy update
            if action_now == q_matrix[:, row, colum].argmax():
                for k in range(pi_matrix.shape[0]):
                    if k == action_now:
                        pi_matrix[k, row, colum] = pb
                    else:
                        pi_matrix[k, row, colum] = ps

            row = row_next
            colum = colum_next

        if iteration % 5000 == 0:
            print('epoch', iteration)
    Draw_Result(world_matrix, pi_matrix)




if __name__ == '__main__':
    Make_World_Matrix(World_Matrix)
    Make_Move_Matrix(Move_Matrix)
    Make_Pi_Matrix(Pi_Matrix,epsilon)
    Make_Action_Reward(Action_Reward_Matrix, World_Matrix)

    TD_Iteration(World_Matrix, Pi_Matrix, Move_Matrix,State_Matrix, Action_Reward_Matrix, Q_Matrix)
