import gym
import gym_gvgai

env = gym.make('gvgai-ikaruga-lvl0-v0')
env.reset()

# Erase the test.txt file before writing new info to it
open("test.txt", "w").close()

score = 0
for i in range(100):
    # env.render()
    action_id = env.action_space.sample()
    state, reward, isOver, info = env.step(action_id)
    print(str(info["ascii"]))
    print(" " * 20)
    # print("--" * 20)
    # print(" " * 20)

    with open("test.txt", "a") as f:
        f.write(str(info["ascii"]))
        f.write("\n \n")
        # f.write(" " * 20)
        # f.write("\n")

    score += reward
    #print("Action " + str(action_id) + " played at game tick " + str(i+1) + ", reward=" + str(reward) + ", new score=" + str(score))
    test = info['grid'][0][0][0]
    #if(test):
    #    print(test.itypeKey)
    #else:
    #    print("None")
    if isOver:
        print("Game over at game tick " + str(i+1) + " with player " + info['winner'])
        break
