import argparse
import sys
from gym_mcts import *

#####################################################################
############################### UTIL ################################
#####################################################################

def stateToString(state):
    string = ''
    for x in state:
        for y in x:
            if sum(y) > 0:
                string += '1'
            string += str(y)
        string+='\n'
    return string

#####################################################################
############################### MAIN ################################
#####################################################################

if __name__ == '__main__':
    random.seed(time.time())

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?',
                        default='CartPole-v0',
                        help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    outdir = 'log/{}/'.format(str(random.randint(0, 10000000)))
    env.seed(random.randint(0, 10000000))
    mon = wrappers.Monitor(env, directory=outdir, force=True)
    mon.reset()
    env.reset()

    # logger
    mylog = mylogger.LOG(args.env_id, outdir)

    #start
    obs = env.step(1)[0]
    mon.step(1)
    agent = MCTSAgent( [ 0, 2 ,3 ] )

    episode_count = 10
    move_count = 1
    reward = 0
    done = False

    for i in range(1, episode_count):
        while True:
            t1 = time.time()
            action = agent.mctsaction(env)
            t2 = time.time()
            print('Running time for MCTS: ', t2 - t1)
            obs, _, done, _ = env.step(action)
            mon.step(action)
            mon.render()


            ## LOG and save image
            filename = str(i) +'_'+ str(move_count) +'_'+ str(action)
            mylog.write(str(i) +' '+ str(move_count) +' '+ str(action)+'\n')
            mylog.writePng(obs, filename)
            if done:
                print('Game done!')
                break
            move_count += 1

    # Close the env and write monitor result info to disk
    env.close()

#####################################################################
############################ END ####################################
#####################################################################
