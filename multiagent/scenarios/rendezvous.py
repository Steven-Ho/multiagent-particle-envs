import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from copy import deepcopy

class Scenario(BaseScenario):
    def make_world(self, na=4, nl=1, random=True, random_agent=True, init=None):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = na
        num_landmarks = nl
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            # agent.size = 0.15
            agent.size = 0.075
            agent.accel = 1.0
            agent.max_speed = 1.0       
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        if num_agents == 2:
            self.init_pos = np.array([[0.8, -0.8],[0.8, 0.8]]) if init is None else np.array(init)
        elif num_agents == 3:
            self.init_pos = np.array([[-0.8,-0.8],[0.8,-0.8],[0.8, 0.8]])
        elif num_agents == 4:
            self.init_pos = np.array([[0.8,0.8],[-0.8,0.8],[-0.8,-0.8],[0.8,-0.8]])
        else:
            self.init_pos = [np.random.uniform(-1, +1, world.dim_p) for i in range(num_agents)]
        self.init_landmark = np.array([[0.0, 0.0]])
        self.random = random
        self.random_agent = random_agent
        self.reset_world(world)
        return world

    def reset_world(self, world, pos=[]):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        self.agent_pos = []
        if len(pos) > 0:
            self.init_pos = np.array(pos)
        if self.random_agent:
            self.init_pos = [np.random.uniform(-1, +1, world.dim_p) for i in range(len(world.agents))]
        for i, agent in enumerate(world.agents):
            self.agent_pos.append(self.init_pos[i])
            agent.state.p_pos = deepcopy(self.agent_pos[-1])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c) 
        if self.random:
            self.landmark_pos = []
            for landmark in world.landmarks:
                self.landmark_pos.append(np.random.uniform(-0.5, +0.5, world.dim_p))
                landmark.state.p_pos = deepcopy(self.landmark_pos[-1])
                landmark.state.p_vel = np.zeros(world.dim_p)
        else:
            self.landmark_pos = []
            for i, landmark in enumerate(world.landmarks):
                self.landmark_pos.append(self.init_landmark[i])
                landmark.state.p_pos = deepcopy(self.landmark_pos[-1])
                landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # individual reward
        # rew = -np.linalg.norm(agent.state.p_pos)
        diff = agent.state.p_pos - world.landmarks[0].state.p_pos
        dist = np.linalg.norm(diff)
        rew = -0.1*dist*dist
        # if dist < 0.075:
        #     rew += 0.1
        return rew

    def done(self, agent, world):
        # dists = []
        # for agent in world.agents:
        #     diff = agent.state.p_pos - world.landmarks[0].state.p_pos
        #     dists.append(np.linalg.norm(diff))
        # if min(dists) < 0.075:
        #     done = True
        # else:
        #     done = False
        # return done
        return False

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)
