import numpy as np

# 해당 action 에 대해서 state를 update 해주는 함수
def get_state(state, action):
    action_grid = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    #  새로운 x 좌표 = 기존의 x 좌표 + action 을 취했을때의 delta x 변화량// 새로운 y 좌표 = 기존의 y 좌표 + action 을 취했을때의 delta y 변화량
    #  action = [0, 1, 2, 3] 의 원소인 0, 1, 2, 3 이 각각 up, down, left, right 에 대응됨을 알 수 있다.
    state[0] += action_grid[action][0]
    state[1] += action_grid[action][1]

    # x 좌표가 grid 를 벗어나게 되면 그 전의 state 를 유지.// x > 3 -> x = 3, x < 0 -> x = 0// y > 3 -> y = 3, y < 0 -> y = 0.
    if state[0] < 0:
        state[0] = 0
    elif state[0] > 3:
        state[0] = 3

    if state[1] < 0:
        state[1] = 0
    elif state[1] > 3:
        state[1] = 3

    return state[0], state[1]


def policy_evaluation(grid_width, grid_height, action, policy, iter_num, reward=-1, dis=1):
    # table initialize
    post_value_table = np.zeros([grid_height, grid_width], dtype=float)

    # iteration
    if iter_num == 0:
        print('Iteration: {} \n{}\n'.format(iter_num, post_value_table))
        return post_value_table

    for iteration in range(iter_num):
        next_value_table = np.zeros([grid_height, grid_width], dtype=float)
        for i in range(grid_height):
            for j in range(grid_width):
                # (i,j) = (0,0), (3,3) 의 value 값 => 0
                if i == j and ((i == 0) or (i == 3)):
                    value_t = 0

                # 그외의 (i,j) 좌표에 대해서
                else:
                    value_t = 0
                    # main code 에서 action = [0, 1, 2, 3] 이므로 action 0 ~ 3 에 대해서 loop 가 돌아가게 된다.
                    for act in action:
                        # state update
                        i_, j_ = get_state([i, j], act)
                        # value 값 계산
                        value = policy[i][j][act] * (reward + dis * post_value_table[i_][j_])
                        value_t += value
                # 소수 셋째 자리까지 -> 넷째 자리에서 반올림 해준다.
                next_value_table[i][j] = round(value_t, 3)

        # 위에서 iteration 은 0부터 시작하므로 실제로는 1을 더한것이 실제 iteration 횟수가 된다.
        iteration += 1
        # print result
        if (iteration % 10) != iter_num:
            # print result
            if iteration > 100:
                if (iteration % 20) == 0:
                    print('Iteration: {} \n{}\n'.format(iteration, next_value_table))
            else:
                if (iteration % 10) == 0:
                    print('Iteration: {} \n{}\n'.format(iteration, next_value_table))
        else:
            print('Iteration: {} \n{}\n'.format(iteration, next_value_table))

        post_value_table = next_value_table

    return next_value_table

grid_width = 4
grid_height = grid_width
action = [0, 1, 2, 3] # up, down, left, right
policy = np.empty([grid_height, grid_width, len(action)], dtype=float)
#print(policy)

""" policy -> 3차원으로 정의 """

# 해당 (i,j) 좌표 grid 에서
for i in range(grid_height):
    for j in range(grid_width):
        # action 에 따라 정의가 되는데,
        for k in range(len(action)):
            # 해당 action 에서 grid 의 (i,j) 좌표값이 (0,0), (3,3) 이면 해당 policy 는 0 값을 갖지고 나머지 좌표값에 대해서는 0.25를 가지게 된다.
            if i==j and ((i==0) or (i==3)):
                policy[i][j]=0.00
            else :
                policy[i][j]=0.25
"""
policy matrix -> height * width * depth = 3 * 3 * 4 인 이유. depth 가 len(action) 이므로.
[[[ -> height, width, action 순으로 행렬이 정의. 
예를들면, policy[0][0] 이면, 해당 i,j 좌표값인 (0,0) 에 대해서 action 이 0,1,2,3 일때이므로 [0. 0. 0. 0.] 이 출력됨.
해당 출력값.

[[[0.   0.   0.   0.  ]
  [0.25 0.25 0.25 0.25]
  [0.25 0.25 0.25 0.25]
  [0.25 0.25 0.25 0.25]]

 [[0.25 0.25 0.25 0.25]
  [0.25 0.25 0.25 0.25]
  [0.25 0.25 0.25 0.25]
  [0.25 0.25 0.25 0.25]]

 [[0.25 0.25 0.25 0.25]
  [0.25 0.25 0.25 0.25]
  [0.25 0.25 0.25 0.25]
  [0.25 0.25 0.25 0.25]]

 [[0.25 0.25 0.25 0.25]
  [0.25 0.25 0.25 0.25]
  [0.25 0.25 0.25 0.25]
  [0.   0.   0.   0.  ]]]
"""

#policy[0][0] = [0] * grid_width
#policy[3][3] = [0] * grid_width
#print(policy)

value = policy_evaluation(grid_width, grid_height, action, policy, 100)

print(value)