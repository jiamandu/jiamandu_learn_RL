import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

'''
Macro Parameters
'''
epsilon_start = 1
epsilon_end = 0.1
iteration_time = 10000
policy_evaluate_time = 10000
delta_epsilon = (epsilon_start - epsilon_end) / iteration_time

alpha = 0.01

gama = 0.95

boundary_return = -1
forbidden_return = -2
target_return = 1

'''
Matrices
'''
World_Matrix = np.zeros((5, 5))
State_Number = World_Matrix.shape[0] * World_Matrix.shape[1]

Action_Number = 5
Policy_Matrix = np.zeros((State_Number, Action_Number))

Pi_Matrix = np.zeros((State_Number, State_Number))

Q_Matrix = np.zeros((State_Number, Action_Number))

Reward_Matrix = np.zeros((State_Number, Action_Number))

Omega_number = 28
Phi_Matrix = np.zeros((State_Number,Omega_number))

Omega_Vector = np.zeros((Omega_number))

def Make_World():
    global Reward_Matrix
    World_Matrix[1][1] = -1
    World_Matrix[1][2] = -1
    World_Matrix[2][2] = -1
    World_Matrix[3][1] = -1
    World_Matrix[3][2] = 1
    World_Matrix[3][3] = -1
    World_Matrix[4][1] = -1

    for state in range(State_Number):
        row, col = state // 5, state % 5
        for action in range(Action_Number):
            if action == 0:
                new_row, new_col = max(0, row - 1), col
            elif action == 1:
                new_row, new_col = row, min(4, col + 1)
            elif action == 2:
                new_row, new_col = min(4, row + 1), col
            elif action == 3:
                new_row, new_col = row, max(0, col - 1)
            else:
                new_row, new_col = row, col

            if new_row == row and new_col == col and action != 4:  # 撞墙
                Reward_Matrix[state, action] = boundary_return
            elif World_Matrix[new_row, new_col] == 1:  # 目标
                Reward_Matrix[state, action] = target_return
            elif World_Matrix[new_row, new_col] == -1:  # 禁区
                Reward_Matrix[state, action] = forbidden_return

def Init_Policy_Matrix():
    p_sub = epsilon_start / Action_Number
    p_main = 1 - (Action_Number - 1) * p_sub

    policy_matrix = np.full((State_Number, Action_Number), p_sub)
    chosen_actions = np.random.randint(0, Action_Number, State_Number)
    policy_matrix[np.arange(State_Number), chosen_actions] = p_main

    return policy_matrix

def Calculate_Pi_Matrix(policy_matrix):
    pi_matrix = np.zeros((State_Number, State_Number))

    states = np.arange(State_Number)

    # Action 0: 向上 (UP)
    if Action_Number > 0:
        targets = np.where(states < 5, states, states - 5)
        pi_matrix[states, targets] += policy_matrix[:, 0]

    # Action 1: 向右 (RIGHT)
    if Action_Number > 1:
        targets = np.where(states % 5 == 5 - 1, states, states + 1)
        pi_matrix[states, targets] += policy_matrix[:, 1]

    # Action 2: 向下 (DOWN)
    if Action_Number > 2:
        targets = np.where(states >= State_Number - 5, states, states + 5)
        pi_matrix[states, targets] += policy_matrix[:, 2]

    # Action 3: 向左 (LEFT)
    if Action_Number > 3:
        targets = np.where(states % 5 == 0, states, states - 1)
        pi_matrix[states, targets] += policy_matrix[:, 3]

    if Action_Number > 4:
        pi_matrix[states, states] += policy_matrix[:, 4]

    return pi_matrix

def Mov_Once_Pi(state_now, pi_matrix):
    probabilities = pi_matrix[state_now, :]
    return np.random.choice(State_Number, p=probabilities)

def Mov_Once_Policy(state_now, policy_matrix):
    probabilities = policy_matrix[state_now, :]
    action = np.random.choice(Action_Number, p=probabilities)
    reward = Reward_Matrix[state_now, action]

    row, col = state_now // 5, state_now % 5
    if action == 0:
        new_row, new_col = max(0, row - 1), col
    elif action == 1:
        new_row, new_col = row, min(4, col + 1)
    elif action == 2:
        new_row, new_col = min(4, row + 1), col
    elif action == 3:
        new_row, new_col = row, max(0, col - 1)
    else:
        new_row, new_col = row, col
    state_next = new_row * 5 + new_col
    return action, reward, state_next

def Calculate_Phi_Vector(state):
    x_val = (state // 5) / 4.0
    y_val = (state % 5) / 4.0

    phi_vector = np.array([
        1.0,

        x_val,
        y_val,

        x_val ** 2,
        x_val * y_val,
        y_val ** 2,

        x_val ** 3,
        x_val * y_val ** 2,
        x_val ** 2 * y_val,
        y_val ** 3,

        x_val ** 4,
        x_val ** 3 * y_val,
        x_val ** 2 * y_val ** 2,
        x_val * y_val ** 3,
        y_val ** 4,

        x_val ** 5,
        x_val ** 4 * y_val,
        x_val ** 3 * y_val ** 2,
        x_val ** 2 * y_val ** 3,
        x_val * y_val ** 4,
        y_val ** 5,

        x_val ** 6,
        x_val ** 5 * y_val,
        x_val ** 4 * y_val ** 2,
        x_val ** 2 * y_val ** 4,
        x_val ** 3 * y_val ** 3,
        x_val * y_val ** 5,
        y_val ** 6,
    ])

    return phi_vector

def Calculate_Q_MAtrix(q_matrix, value_matrix, ):
    for state in range(q_matrix.shape[0]):
        for action in range(q_matrix.shape[1]):

            row, col = state // 5, state % 5
            if action == 0:
                new_row, new_col = max(0, row - 1), col
            elif action == 1:
                new_row, new_col = row, min(4, col + 1)
            elif action == 2:
                new_row, new_col = min(4, row + 1), col
            elif action == 3:
                new_row, new_col = row, max(0, col - 1)
            else:
                new_row, new_col = row, col
            state_next = new_row * 5 + new_col

            q_matrix[state, action] = Reward_Matrix[state, action] + value_matrix[new_row,new_col]

    return q_matrix

def Update_Policy_Matrix(q_matrix):
    global  epsilon_start
    epsilon_start -= delta_epsilon

    p_sub = epsilon_start / Action_Number
    p_main = 1 - (Action_Number - 1) * p_sub

    policy_matrix = np.full((State_Number, Action_Number), p_sub)
    max_positions = np.argmax(q_matrix, axis=1)
    for i in range(0,25):
        policy_matrix[i, max_positions[i]] = p_main
    return policy_matrix

def Draw_Result(matrix,policy_matrix):

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

    max_indices = np.argmax(policy_matrix, axis=1)
    for state in range(State_Number):
        j = state % 5
        i = int(state / 5)
        a = max_indices[state]
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
    Make_World()
    Policy_Matrix = Init_Policy_Matrix()
    Pi_Matrix = Calculate_Pi_Matrix(Policy_Matrix)

    '''
    TD iteration    
    '''

    for iter1 in tqdm(range(iteration_time), desc="Learning", unit="times"):

        state1 = np.random.randint(0, 25)
        state2 = 0
        for iter2 in tqdm(range(policy_evaluate_time), desc="Learning", unit="times"):
            action, reward, state2 = Mov_Once_Policy(state1, Policy_Matrix)

            Phi_Vector1 = Calculate_Phi_Vector(state1)
            Phi_Vector2 = Calculate_Phi_Vector(state2)

            TD_Target = reward + gama * np.dot(Phi_Vector2, Omega_Vector)
            # TD_Target = Value_Pi[int(state2/5),int(state2%5)]
            TD_Error = np.dot(Phi_Vector1, Omega_Vector) - TD_Target

            # alpha = 1 / (iteration + 1)

            Omega_Vector = Omega_Vector - alpha * TD_Error * Phi_Vector1

            if state2 == 17:
                state1 = np.random.randint(0, 25)
            else:
                state1 = state2


        Value_Matrix = np.zeros((5,5))
        for i in range(5):
            for j in range(5):
                Phi_Vector = Calculate_Phi_Vector(i * 5 + j)
                Value_Matrix[i, j] = np.dot(Phi_Vector, Omega_Vector)

        Q_Matrix = Calculate_Q_MAtrix(Q_Matrix,Value_Matrix)
        Policy_Matrix = Update_Policy_Matrix(Q_Matrix)

    print(Value_Matrix)
    print(Policy_Matrix)
    Draw_Result(World_Matrix, Policy_Matrix)

