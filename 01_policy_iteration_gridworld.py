import numpy as np

"""Policy Evaluation"""
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
                        # value 값 계산 => Bellman expectation eqn.
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

"""Policy Improvement <- greedy policy improvement 적용"""
#policy evaluation에서는 state의 value func.을 iterative하게 계산하여 모든 state들에 대한 true value를 도출하는 과정이라면,
#policy improvement에서는 도출된 state value들을 토대로 현재의 policy를 action-value func.을 이용하여 더 좋은 action을 선택하는 policy로 만드는 과정이다.

def policy_improvement(value, action, policy, reward=-1, grid_width=4):
    grid_height = grid_width

    action_match = ['Up', 'Down', 'Left', 'Right']
    action_table = []

    # get Q-func.
    for i in range(grid_height):
        for j in range(grid_width):
            q_func_list = []
            if i == j and ((i == 0) or (i == 3)):
                action_table.append('T')
            else:
                # 해당 i,j 에 대해서 ('Up', 'Down', 'Left', 'Right') 에 대한 action 을 하게 되었을때의 q_function 을 각각 구해서 q_func_list에 추가 해준다.
                for k in range(len(action)):
                    i_, j_ = get_state([i, j], k)
                    q_func_list.append(value[i_][j_])
                # 해당 (i,j) 에서의 q_function의 최댓값이 가장 좋은 action 이다.
                # q_func_list 에서 (index, value) -> (action_v,x) 라고 하자. 만약 그때의 x 값이 q_func_list 의 최댓값이면 그때의 index 값을 반환한다.
                # 예를 들면 q_function_list 의 최댓값이 Left, Right 두개라면, max_actions = [2, 3]
                max_actions = [action_v for action_v, x in enumerate(q_func_list) if x == max(q_func_list)]

                # update policy
                policy[i][j] = [0] * len(action)  # initialize q-func_list
                # 위와 같이 max_actions = [2, 3] 이라면 y 가 2, 3 순서로 iteration 될 것 이다. 그러면, policy[i][j][2] 와 [3]에 = 0.5 저장
                for y in max_actions:
                    policy[i][j][y] = (1 / len(max_actions))

                # get action
                idx = np.argmax(policy[i][j])
                action_table.append(action_match[idx])
    action_table = np.asarray(action_table).reshape((grid_height, grid_width))

    print('Updated policy is :\n{}\n'.format(policy))
    print('at each state, chosen action is :\n{}'.format(action_table))

    return policy

updated_policy = policy_improvement(value, action, policy)